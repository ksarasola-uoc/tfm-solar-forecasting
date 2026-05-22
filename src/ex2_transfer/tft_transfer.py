"""
tft_transfer.py — Soporte para transferencia TFT entre plantas solares.

Estrategia de transferencia
────────────────────────────
1. Pre-entrenamiento: TFT multi-planta sobre 5 plantas fuente.
   TimeSeriesDataSet con EncoderNormalizer → normalización por ventana de encoder,
   sin estadísticas por grupo → habilita zero-shot a plantas no vistas.

2. Zero-shot: Se crea un nuevo TFT desde los datasets de TALASOL (misma
   arquitectura que el pre-entrenado) y se cargan los pesos via state_dict.
   El modelo nunca vio TALASOL durante el entrenamiento.

3. Fine-tuning progresivo: Se parte del state_dict pre-entrenado y se continúa
   el entrenamiento con cantidades crecientes de datos de TALASOL.

4. Layer freezing: Se congelan los componentes temporales del TFT
   (LSTM encoder, LSTM decoder, multihead attention) y solo se adaptan
   las capas de selección de variables y salida.

Diferencia respecto a nbeats_transfer.py
─────────────────────────────────────────
- TFT usa pytorch-forecasting (TimeSeriesDataSet, pytorch-lightning).
- La transferencia de pesos se hace con state_dict, no cargando checkpoints
  directamente, para evitar problemas de compatibilidad entre datasets.
- EncoderNormalizer (en lugar de GroupNormalizer) es la clave que permite
  zero-shot en plantas no vistas durante el pre-entrenamiento.
"""

import copy
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from evaluation import ForecastEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Callback para capturar curvas de aprendizaje
# ─────────────────────────────────────────────────────────────────────────────

class LossHistoryCallback(Callback):
    """Captura train_loss y val_loss al final de cada época de validación."""

    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        train_loss = m.get('train_loss', m.get('train_loss_epoch'))
        val_loss   = m.get('val_loss')
        if train_loss is not None:
            self.train_losses.append(float(train_loss))
        if val_loss is not None:
            self.val_losses.append(float(val_loss))


# ─────────────────────────────────────────────────────────────────────────────
# Preparación de DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variables categóricas a numéricas para TFT.

    Aplica la misma codificación que 05_baseline_tft.ipynb:
    - estacion → 0..3
    - zona_climatica → 0..2  (Norte=0, Centro=1, Sur=2)
    - seguimiento → 0/1

    Args:
        df: DataFrame con columnas 'estacion', 'zona_climatica', 'seguimiento'.

    Returns:
        Copia del DataFrame con columnas *_encoded añadidas.
    """
    df = df.copy()

    estacion_map = {'Invierno': 0, 'Primavera': 1, 'Verano': 2, 'Otoño': 3}
    df['estacion_encoded'] = (
        df['estacion'].map(estacion_map).fillna(0).astype(float)
    )

    zona_map = {'Norte': 0, 'Centro': 1, 'Sur': 2}
    df['zona_climatica_encoded'] = (
        df['zona_climatica'].map(zona_map).fillna(1).astype(float)
    )

    df['seguimiento_encoded'] = df['seguimiento'].astype(float)

    return df


def prepare_tft_df(df: pd.DataFrame, planta: str,
                   smoke_test: bool = False,
                   max_rows: int = 700) -> pd.DataFrame:
    """
    Filtra un split por planta y añade columnas requeridas por TimeSeriesDataSet.

    pytorch-forecasting requiere:
    - time_idx: entero incremental dentro de cada grupo (0..N-1).
    - series_id: identificador de la serie (nombre de la planta).

    Args:
        df:         DataFrame completo (todas las plantas).
        planta:     Planta a filtrar.
        smoke_test: Si True, limita a max_rows filas.
        max_rows:   Límite de filas en modo smoke test.

    Returns:
        DataFrame filtrado, ordenado y con time_idx / series_id.
    """
    df_p = df[df['planta_nombre'] == planta].copy()
    df_p = df_p.sort_values('timestamp_utc').reset_index(drop=True)
    if smoke_test and len(df_p) > max_rows:
        df_p = df_p.iloc[:max_rows]
    df_p['time_idx']  = np.arange(len(df_p), dtype=int)
    df_p['series_id'] = planta
    return df_p


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de TimeSeriesDataSets
# ─────────────────────────────────────────────────────────────────────────────

def _create_tft_dataset(
        df: pd.DataFrame,
        known_reals: List[str],
        unknown_reals: List[str],
        static_reals: List[str],
        max_encoder_length: int,
        min_encoder_length: int = None,
) -> TimeSeriesDataSet:
    """
    Crea un TimeSeriesDataSet con EncoderNormalizer.

    EncoderNormalizer normaliza el target por ventana de encoder (media/std
    calculadas on-the-fly), sin estadísticas almacenadas por grupo.
    Esto es fundamental para zero-shot: permite aplicar el modelo a plantas
    no vistas durante el pre-entrenamiento.

    Args:
        df:                 DataFrame preparado (con time_idx y series_id).
        known_reals:        Features deterministas en el futuro.
        unknown_reals:      Features meteorológicas (no perfectamente conocidas).
        static_reals:       Características estáticas de la planta.
        max_encoder_length: Horas de contexto histórico.
        min_encoder_length: Mínimo de contexto para una muestra válida.
                            Por defecto max_encoder_length // 2.

    Returns:
        TimeSeriesDataSet listo para to_dataloader().
    """
    if min_encoder_length is None:
        min_encoder_length = max_encoder_length // 2

    # Garantizar que 'produccion' está en unknown_reals
    unknown_reals_full = list(unknown_reals)
    if 'produccion' not in unknown_reals_full:
        unknown_reals_full.append('produccion')

    return TimeSeriesDataSet(
        df,
        time_idx               = 'time_idx',
        target                 = 'produccion',
        group_ids              = ['series_id'],
        max_encoder_length     = max_encoder_length,
        max_prediction_length  = 1,
        min_encoder_length     = min_encoder_length,
        time_varying_known_reals   = known_reals,
        time_varying_unknown_reals = unknown_reals_full,
        static_reals               = static_reals,
        # EncoderNormalizer: clave para zero-shot cross-plant
        target_normalizer      = EncoderNormalizer(),
        allow_missing_timesteps= True,
        add_relative_time_idx  = True,
        add_target_scales      = True,
        add_encoder_length     = True,
    )


def build_source_datasets(
        plantas_source: List[str],
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        known_reals: List[str],
        unknown_reals: List[str],
        static_reals: List[str],
        max_encoder_length: int,
        smoke_test: bool = False,
        max_rows: int = 700,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Crea datasets multi-planta para el pre-entrenamiento.

    Concatena todos los DataFrames de plantas fuente en un único DataFrame
    (pytorch-forecasting los separa por series_id internamente).

    Args:
        plantas_source:     Lista de plantas fuente.
        df_train:           DataFrame completo de train.
        df_val:             DataFrame completo de val.
        known_reals:        Features conocidas futuras.
        unknown_reals:      Features desconocidas futuras.
        static_reals:       Features estáticas de planta.
        max_encoder_length: Horas de contexto.
        smoke_test:         Si True, limita filas por planta.
        max_rows:           Límite en smoke test.

    Returns:
        (train_ds, val_ds) multi-planta.
    """
    frames_train, frames_val = [], []

    for planta in plantas_source:
        df_tr = prepare_tft_df(df_train, planta, smoke_test, max_rows)
        df_va = prepare_tft_df(df_val,   planta, smoke_test, max_rows)
        frames_train.append(df_tr)
        frames_val.append(df_va)
        print(f"  {planta}: train={len(df_tr):,}  val={len(df_va):,}")

    df_all_train = pd.concat(frames_train, ignore_index=True)
    df_all_val   = pd.concat(frames_val,   ignore_index=True)

    train_ds = _create_tft_dataset(
        df_all_train, known_reals, unknown_reals, static_reals, max_encoder_length
    )

    # Val usa el mismo esquema que train pero sin randomización
    min_enc_val = min(
        max_encoder_length,
        max(2, len(df_all_val) // max(len(plantas_source), 1) - 1)
    )
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, df_all_val,
        predict=False, stop_randomization=True,
        min_encoder_length=min_enc_val,
    )

    print(f"\nSource datasets: train={len(train_ds):,}  val={len(val_ds):,}")
    return train_ds, val_ds


def build_target_datasets(
        planta_target: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        known_reals: List[str],
        unknown_reals: List[str],
        static_reals: List[str],
        max_encoder_length: int,
        n_finetune_hours: Optional[int] = None,
        smoke_test: bool = False,
        max_rows: int = 700,
) -> Tuple[Optional[TimeSeriesDataSet], TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Crea datasets INDEPENDIENTES para la planta target (TALASOL).

    Los datasets se crean con _create_tft_dataset (mismo esquema que el source),
    lo que garantiza arquitecturas TFT idénticas y permite state_dict transfer.

    Args:
        planta_target:    Nombre de la planta objetivo.
        df_train/val/test: DataFrames completos de cada split.
        known_reals, unknown_reals, static_reals: Features (mismos que source).
        max_encoder_length: Horas de contexto.
        n_finetune_hours: None = todo el train (baseline), 0 = zero-shot, >0 = FT.
        smoke_test:       Si True, limita filas.
        max_rows:         Límite en smoke test.

    Returns:
        ft_ds:   Dataset de fine-tuning (None para zero-shot).
        val_ds:  Dataset de validación.
        test_ds: Dataset de test.
    """
    df_tr = prepare_tft_df(df_train, planta_target, smoke_test, max_rows)
    df_va = prepare_tft_df(df_val,   planta_target, smoke_test, max_rows)
    df_te = prepare_tft_df(df_test,  planta_target, smoke_test, max_rows)

    # Dataset de referencia para train completo (define el esquema)
    train_full_ds = _create_tft_dataset(
        df_tr, known_reals, unknown_reals, static_reals, max_encoder_length
    )

    min_enc_val  = min(max_encoder_length, max(2, len(df_va)  - 1))
    min_enc_test = min(max_encoder_length, max(2, len(df_te) - 1))

    val_ds  = TimeSeriesDataSet.from_dataset(
        train_full_ds, df_va,
        predict=False, stop_randomization=True,
        min_encoder_length=min_enc_val,
    )
    test_ds = TimeSeriesDataSet.from_dataset(
        train_full_ds, df_te,
        predict=False, stop_randomization=True,
        min_encoder_length=min_enc_test,
    )

    ft_ds = None
    if n_finetune_hours is None:
        # Baseline: todo el train
        ft_ds = train_full_ds
        print(f"  {planta_target} baseline: {len(ft_ds):,} seqs (todo el train)")
    elif n_finetune_hours > 0:
        n_ft  = min(n_finetune_hours + max_encoder_length, len(df_tr))
        df_ft = df_tr.iloc[:n_ft].copy()
        if len(df_ft) >= max_encoder_length + 1:
            ft_ds = TimeSeriesDataSet.from_dataset(
                train_full_ds, df_ft,
                predict=False, stop_randomization=False,
                min_encoder_length=min(max_encoder_length, max(2, len(df_ft) - 1)),
            )
            print(f"  {planta_target} fine-tune: {len(ft_ds):,} seqs ({n_ft} filas)")
        else:
            print(f"  ⚠ Datos insuficientes ({len(df_ft)} filas) para fine-tuning")
    else:
        print(f"  {planta_target}: zero-shot (sin fine-tuning)")

    print(f"  val={len(val_ds):,}  test={len(test_ds):,}")
    return ft_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# Predicción y evaluación
# ─────────────────────────────────────────────────────────────────────────────

def get_tft_predictions(
        model: TemporalFusionTransformer,
        test_ds: TimeSeriesDataSet,
        df_test_orig: pd.DataFrame,
        planta: str,
        batch_size: int = 64,
) -> pd.DataFrame:
    """
    Genera predicciones en escala original (MW) y las combina con los valores reales.

    Usa model.predict(mode='prediction') que aplica la inversa de EncoderNormalizer
    automáticamente → devuelve valores en escala original.
    Los actuals vienen del DataFrame original (no normalizado).

    Args:
        model:        TFT en modo eval.
        test_ds:      TimeSeriesDataSet de test.
        df_test_orig: DataFrame original del split de test (sin normalizar).
        planta:       Nombre de la planta.
        batch_size:   Tamaño de batch para inferencia.

    Returns:
        DataFrame con columnas: timestamp, y_true, y_pred, planta_nombre.
    """
    model.eval()
    test_dl = test_ds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, pin_memory=False
    )

    # Predicciones en escala original (EncoderNormalizer inverse aplicado)
    # pytorch-forecasting >=1.x devuelve un named tuple Prediction;
    # >=0.10.x devuelve una tupla (predictions, index). Compatibilidad con ambos.
    result = model.predict(
        test_dl, mode='prediction', return_index=True
    )
    predictions = result.prediction if hasattr(result, 'prediction') else result[0]
    pred_index  = result.index      if hasattr(result, 'index')      else result[1]

    y_pred = np.clip(predictions.cpu().numpy().flatten(), 0, None)

    # Actuals desde el DataFrame original (produccion en MW, no normalizada)
    df_te = prepare_tft_df(df_test_orig, planta)
    df_map = df_te.set_index('time_idx')

    time_idxs  = pred_index['time_idx'].values
    y_true     = df_map.loc[time_idxs, 'produccion'].values.flatten()
    timestamps = df_map.loc[time_idxs, 'timestamp_utc'].values

    return pd.DataFrame({
        'timestamp':     timestamps,
        'y_true':        y_true,
        'y_pred':        y_pred,
        'planta_nombre': planta,
    })


def evaluate_tft(
        model: TemporalFusionTransformer,
        test_ds: TimeSeriesDataSet,
        df_test_orig: pd.DataFrame,
        planta: str,
        batch_size: int = 64,
) -> Tuple[dict, pd.DataFrame]:
    """
    Calcula métricas en escala original (MAE, RMSE, MAPE, R²).

    Returns:
        (metrics_dict, df_pred) donde df_pred tiene y_true y y_pred en MW.
    """
    df_pred = get_tft_predictions(model, test_ds, df_test_orig, planta, batch_size)
    metrics = ForecastEvaluator.calculate_metrics(
        df_pred['y_true'].values,
        df_pred['y_pred'].values,
    )
    return metrics, df_pred


def save_results_row(results_dir: Path, scenario: str,
                     metrics: dict, n_ft_hours: int) -> None:
    """
    Append a CSV acumulativo results_dir/metrics.csv.

    Misma firma y formato que nbeats_transfer.save_results_row.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'metrics.csv'

    row = {
        'timestamp':  pd.Timestamp.now().isoformat(timespec='seconds'),
        'scenario':   scenario,
        'n_ft_hours': n_ft_hours,
        'MAE':        round(metrics.get('MAE',  float('nan')), 4),
        'RMSE':       round(metrics.get('RMSE', float('nan')), 4),
        'MAPE':       round(metrics.get('MAPE', float('nan')), 4),
        'R2':         round(metrics.get('R2',   float('nan')), 4),
        'n_samples':      metrics.get('n_samples', 0),
        'training_time_s': round(metrics.get('training_time_s', float('nan')), 1),
    }

    fieldnames = list(row.keys())
    if csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path)
            for col in fieldnames:
                if col not in df_old.columns:
                    df_old[col] = float('nan')
            df_old = df_old[fieldnames]
            df_old.to_csv(csv_path, index=False)
        except Exception:
            csv_path.unlink(missing_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  [guardado] {csv_path}  | scenario={scenario}  "
          f"MAE={row['MAE']}  RMSE={row['RMSE']}  R2={row['R2']}")


# ─────────────────────────────────────────────────────────────────────────────
# TFTTransferTrainer — wrapper de pre-entrenamiento y fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

class TFTTransferTrainer:
    """
    Encapsula pre-entrenamiento y fine-tuning de TFT para transfer learning.

    Internamente usa pytorch-lightning y pytorch-forecasting.

    La transferencia de pesos se realiza mediante state_dict:
    - pretrain() entrena en multi-planta y guarda el state_dict.
    - finetune() crea un nuevo modelo desde los datos target (misma arquitectura)
      y carga el state_dict pre-entrenado. Funciona porque EncoderNormalizer
      + mismos hiperparámetros → misma arquitectura para cualquier planta.

    Atributos públicos:
        _pretrain_state : dict con state_dict del pretraining.
        pretrain_hist   : dict con curvas de aprendizaje del pretraining.
        finetune_hist   : dict con curvas de fine-tuning (None hasta finetune()).
    """

    def __init__(self, model_cfg: dict, device: str, checkpoints_dir: Path):
        self.model_cfg        = model_cfg
        self.device           = device
        self.checkpoints_dir  = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._pretrain_state: Optional[dict] = None
        self.pretrain_hist:   Optional[dict] = None
        self.finetune_hist:   Optional[dict] = None

    def _build_tft(self, train_ds: TimeSeriesDataSet,
                   lr: float,
                   quantiles: list = None) -> TemporalFusionTransformer:
        """
        Crea un TemporalFusionTransformer desde un TimeSeriesDataSet.

        Args:
            train_ds  : dataset de entrenamiento
            lr        : learning rate
            quantiles : cuantiles para QuantileLoss (default: [0.5] para Exp 2)
        """
        quantiles = quantiles or [0.5]
        cfg = self.model_cfg
        model = TemporalFusionTransformer.from_dataset(
            train_ds,
            hidden_size             = cfg['hidden_size'],
            lstm_layers             = 1,
            attention_head_size     = cfg['attention_head_size'],
            dropout                 = cfg['dropout'],
            hidden_continuous_size  = cfg['hidden_continuous_size'],
            loss                    = QuantileLoss(quantiles=quantiles),
            learning_rate           = lr,
            reduce_on_plateau_patience = 3,
            log_interval            = -1,
            log_val_interval        = -1,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  TFT: {n_params:,} parámetros  |  cuantiles: {quantiles}")
        return model

    def _build_pl_trainer(self, cfg: dict,
                          ckpt_name: str) -> Tuple[pl.Trainer,
                                                    LossHistoryCallback,
                                                    ModelCheckpoint]:
        """Crea un pytorch-lightning Trainer con callbacks estándar."""
        loss_cb = LossHistoryCallback()

        ckpt_cb = ModelCheckpoint(
            monitor   = 'val_loss',
            mode      = 'min',
            save_top_k= 1,
            dirpath   = str(self.checkpoints_dir),
            filename  = ckpt_name,
        )
        early_stop_cb = EarlyStopping(
            monitor   = 'val_loss',
            min_delta = 1e-4,
            patience  = cfg['patience'],
            mode      = 'min',
            verbose   = False,
        )

        accelerator = 'gpu' if self.device == 'cuda' else 'cpu'

        trainer = pl.Trainer(
            max_epochs           = cfg['max_epochs'],
            accelerator          = accelerator,
            devices              = 1,
            gradient_clip_val    = 0.1,
            callbacks            = [loss_cb, ckpt_cb, early_stop_cb],
            enable_progress_bar  = True,
            enable_model_summary = False,
            logger               = False,
            enable_checkpointing = True,
            num_sanity_val_steps = 0,
        )
        return trainer, loss_cb, ckpt_cb

    def _load_best_checkpoint(self, ckpt_cb: ModelCheckpoint,
                              current_model: TemporalFusionTransformer
                              ) -> TemporalFusionTransformer:
        """Carga el mejor checkpoint si existe; si no, devuelve el modelo actual."""
        best_path = ckpt_cb.best_model_path
        if best_path and Path(best_path).exists():
            try:
                return TemporalFusionTransformer.load_from_checkpoint(best_path)
            except Exception as e:
                print(f"  ⚠ No se pudo cargar checkpoint: {e}. Usando modelo en memoria.")
        return current_model

    def _make_dataloaders(self, train_ds, val_ds, batch_size):
        """Crea DataLoaders de train y val."""
        train_dl = train_ds.to_dataloader(
            train=True,  batch_size=batch_size, num_workers=0, pin_memory=False
        )
        val_dl = val_ds.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=0, pin_memory=False
        )
        return train_dl, val_dl

    def _build_hist(self, loss_cb: LossHistoryCallback,
                    n_train: int, n_val: int) -> dict:
        best_val = min(loss_cb.val_losses) if loss_cb.val_losses else float('nan')
        return {
            'train_losses':  loss_cb.train_losses,
            'val_losses':    loss_cb.val_losses,
            'best_val_loss': best_val,
            'best_iter':     len(loss_cb.train_losses),
            'n_train':       n_train,
            'n_val':         n_val,
        }

    # ── API pública ───────────────────────────────────────────────────────────

    def pretrain(self, train_ds: TimeSeriesDataSet,
                 val_ds: TimeSeriesDataSet, cfg: dict) -> dict:
        """
        Pre-entrena el TFT sobre datos multi-planta (plantas fuente).

        Args:
            train_ds: TimeSeriesDataSet multi-planta de entrenamiento.
            val_ds:   TimeSeriesDataSet multi-planta de validación.
            cfg:      MODEL_CFG.

        Returns:
            Dict con curvas de aprendizaje del pretraining.
        """
        model = self._build_tft(train_ds, lr=cfg['lr'])
        train_dl, val_dl = self._make_dataloaders(train_ds, val_ds, cfg['batch_size'])
        trainer, loss_cb, ckpt_cb = self._build_pl_trainer(cfg, 'pretrained')

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        model = self._load_best_checkpoint(ckpt_cb, model)

        self._pretrain_state = copy.deepcopy(model.state_dict())
        self.pretrain_hist   = self._build_hist(loss_cb, len(train_ds), len(val_ds))

        print(f"✓ Pretraining completado | "
              f"best val_loss: {self.pretrain_hist['best_val_loss']:.4f}")
        return self.pretrain_hist

    def finetune(self,
                 ft_ds: TimeSeriesDataSet,
                 val_ds: TimeSeriesDataSet,
                 cfg: dict,
                 freeze_temporal: bool = False,
                 scenario_name: str = 'ft',
                 ) -> Tuple[TemporalFusionTransformer, dict]:
        """
        Fine-tunea el TFT sobre datos de la planta target.

        Crea un nuevo TFT desde ft_ds (misma arquitectura que el pre-entrenado)
        y carga el state_dict del pretraining. Opcionalmente congela los
        componentes temporales: LSTM encoder, LSTM decoder, multihead attention.

        El LR se reduce según cfg['lr_factor'] (más conservador para TFT).

        Args:
            ft_ds:           TimeSeriesDataSet de fine-tuning.
            val_ds:          TimeSeriesDataSet de validación.
            cfg:             FINETUNE_CFG con lr_factor.
            freeze_temporal: Si True, congela LSTM + attention.
            scenario_name:   Nombre del escenario (para el checkpoint).

        Returns:
            (model_finetuned, hist_dict)
        """
        if self._pretrain_state is None:
            raise RuntimeError("Llama a pretrain() antes de finetune().")

        lr_ft   = self.model_cfg['lr'] * cfg['lr_factor']
        model   = self._build_tft(ft_ds, lr=lr_ft)

        # Transferir pesos pre-entrenados
        missing, unexpected = model.load_state_dict(
            self._pretrain_state, strict=False
        )
        if missing:
            print(f"  ⚠ Keys no cargadas (se inicializarán aleatoriamente): {len(missing)}")
        if unexpected:
            print(f"  ⚠ Keys ignoradas del checkpoint: {len(unexpected)}")

        if freeze_temporal:
            # Congelar LSTM encoder, decoder y attention (core temporal)
            frozen_prefixes = ('encoder', 'decoder', 'multihead_attn')
            n_frozen = 0
            for name, param in model.named_parameters():
                if any(name.startswith(p) for p in frozen_prefixes):
                    param.requires_grad = False
                    n_frozen += param.numel()
            n_total   = sum(p.numel() for p in model.parameters())
            n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [fine-tune] Congelado: LSTM encoder+decoder+attention | "
                  f"entrenables: {n_train:,}/{n_total:,}")

        train_dl, val_dl = self._make_dataloaders(
            ft_ds, val_ds, self.model_cfg['batch_size']
        )
        trainer, loss_cb, ckpt_cb = self._build_pl_trainer(cfg, scenario_name)

        try:
            trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        except Exception as e:
            print(f"  ⚠ Error durante fine-tuning: {e}")

        model = self._load_best_checkpoint(ckpt_cb, model)

        hist = self._build_hist(loss_cb, len(ft_ds), len(val_ds))
        self.finetune_hist = hist

        print(f"✓ Fine-tuning completado | "
              f"best val_loss: {hist['best_val_loss']:.4f}")
        return model, hist
