"""
lstm_transfer.py — Soporte para transferencia LSTM entre plantas solares.

Funciones de soporte y LSTMTransferTrainer para el Experimento 2.
No duplica lógica de lstm_model.py ni data_loaders.py; los usa directamente.

Estrategia de normalización
────────────────────────────
- Plantas fuente: SequenceScaler + SolarScaler independientes por planta,
  fit en su train. ConcatDataset combina los datasets ya normalizados → el
  modelo ve producción normalizada [0,1] de cada planta (evita sesgo por
  capacidad instalada).
- TALASOL: SequenceScaler + SolarScaler independientes, fit en su train.
  Mismo scaler para zero-shot, fine-tuning y baseline → evaluación consistente.

Estrategia de layer freezing
────────────────────────────
- Componentes congelados (capturan patrones universales multi-planta):
    model.lstm   — encoder recurrente
- Componentes entrenables (se adaptan a TALASOL):
    model.attention, model.fc1, model.fc2  — decoder y atención temporal
"""

import copy
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data_loaders import SolarSequenceDataset, SequenceScaler, LagFeatureBuilder
from lstm_model import LSTMForecaster, LSTMTrainer
from models import SolarScaler
from evaluation import ForecastEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares de datos
# ─────────────────────────────────────────────────────────────────────────────

def load_plant_split(df: pd.DataFrame, planta: str,
                     smoke_test: bool = False,
                     max_rows: int = 700) -> pd.DataFrame:
    """
    Filtra un split (train/val/test) por planta con límite opcional.

    Args:
        df:         DataFrame con columna 'planta_nombre'.
        planta:     Nombre de la planta a filtrar.
        smoke_test: Si True, limita a max_rows filas.
        max_rows:   Límite de filas en modo smoke test.

    Returns:
        DataFrame filtrado y ordenado por timestamp_utc.
    """
    df_p = df[df['planta_nombre'] == planta].copy()
    df_p = df_p.sort_values('timestamp_utc').reset_index(drop=True)
    if smoke_test and len(df_p) > max_rows:
        df_p = df_p.iloc[:max_rows]
    return df_p


def _make_dataset(df_p: pd.DataFrame,
                  planta: str,
                  feature_cols: List[str],
                  future_feature_cols: Optional[List[str]],
                  lag_steps: Optional[List[int]],
                  seq_length: int,
                  scaler_x: Optional[SequenceScaler] = None,
                  scaler_y: Optional[SolarScaler] = None,
                  lag_builder: Optional[LagFeatureBuilder] = None,
                  ) -> SolarSequenceDataset:
    """Crea un SolarSequenceDataset para una planta con scalers opcionales."""
    return SolarSequenceDataset(
        df_p, planta,
        feature_cols        = feature_cols,
        future_feature_cols = future_feature_cols,
        lag_steps           = lag_steps,
        seq_length          = seq_length,
        scaler_x            = scaler_x,
        scaler_y            = scaler_y,
        lag_builder         = lag_builder,
    )


def build_source_loaders(
        plantas_source:     List[str],
        df_train:           pd.DataFrame,
        df_val:             pd.DataFrame,
        feature_cols:       List[str],
        future_feature_cols: Optional[List[str]],
        lag_steps:          Optional[List[int]],
        seq_length:         int,
        batch_size:         int,
        smoke_test:         bool = False,
        max_rows:           int = 700,
) -> Tuple[DataLoader, DataLoader, Dict[str, SolarScaler]]:
    """
    Crea DataLoaders multi-planta para el pre-entrenamiento.

    Para cada planta fuente:
      - Filtra el split con load_plant_split().
      - Crea un SolarSequenceDataset (fit de scalers en train).
      - Reutiliza scalers de train en val.

    Combina todos los datasets con ConcatDataset → un único train/val loader.

    Returns:
        train_loader:  DataLoader multi-planta para entrenamiento.
        val_loader:    DataLoader multi-planta para validación.
        scalers_y:     Dict {planta: SolarScaler} (fit en train de cada planta).
    """
    ds_trains, ds_vals = [], []
    scalers_y: Dict[str, SolarScaler] = {}

    for planta in plantas_source:
        df_tr_p = load_plant_split(df_train, planta, smoke_test, max_rows)
        df_va_p = load_plant_split(df_val,   planta, smoke_test, max_rows)

        # Dataset de train: crea todos los scalers
        ds_tr = _make_dataset(df_tr_p, planta, feature_cols,
                               future_feature_cols, lag_steps, seq_length)
        scalers_y[planta] = ds_tr.scaler_y

        # Dataset de val: reutiliza scalers de train
        ds_va = _make_dataset(df_va_p, planta, feature_cols,
                               future_feature_cols, lag_steps, seq_length,
                               scaler_x    = ds_tr.scaler_x,
                               scaler_y    = ds_tr.scaler_y,
                               lag_builder = ds_tr.lag_builder)

        ds_trains.append(ds_tr)
        ds_vals.append(ds_va)

        print(f"  {planta}: train={len(ds_tr):,}  val={len(ds_va):,}  "
              f"scaler_mean={ds_tr.scaler_y.mean_:.1f}")

    concat_train = ConcatDataset(ds_trains)
    concat_val   = ConcatDataset(ds_vals)

    train_loader = DataLoader(concat_train, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(concat_val, batch_size=batch_size,
                              shuffle=False, drop_last=False)

    print(f"\nLoaders multi-planta: train={len(concat_train):,}  val={len(concat_val):,}")
    return train_loader, val_loader, scalers_y


def build_target_loaders(
        planta_target:       str,
        df_train:            pd.DataFrame,
        df_val:              pd.DataFrame,
        df_test:             pd.DataFrame,
        feature_cols:        List[str],
        future_feature_cols: Optional[List[str]],
        lag_steps:           Optional[List[int]],
        seq_length:          int,
        batch_size:          int,
        n_finetune_hours:    Optional[int] = None,
        smoke_test:          bool = False,
        max_rows:            int = 700,
) -> Tuple[Optional[DataLoader], DataLoader, DataLoader,
           SolarSequenceDataset, SolarScaler]:
    """
    Crea DataLoaders para la planta target (TALASOL).

    SolarScaler fit en TODO el train de TALASOL (no limitado a n_finetune_hours).
    Esto garantiza evaluación consistente entre zero-shot, fine-tuning y baseline.

    Args:
        n_finetune_hours: Horas de datos de train usadas para fine-tuning.
                          None = todo el train.
                          0   = zero-shot (no hay fine-tuning).

    Returns:
        ft_loader:   DataLoader para fine-tuning (None si n_finetune_hours == 0).
        val_loader:  DataLoader de validación.
        test_loader: DataLoader de test.
        test_ds:     SolarSequenceDataset de test (para predict/evaluate).
        scaler_y:    SolarScaler fit en train de TALASOL.
    """
    df_tr = load_plant_split(df_train, planta_target, smoke_test, max_rows)
    df_va = load_plant_split(df_val,   planta_target, smoke_test, max_rows)
    df_te = load_plant_split(df_test,  planta_target, smoke_test, max_rows)

    # Dataset completo de train → fit de scalers
    ds_train_full = _make_dataset(df_tr, planta_target, feature_cols,
                                   future_feature_cols, lag_steps, seq_length)
    scaler_x    = ds_train_full.scaler_x
    scaler_y    = ds_train_full.scaler_y
    lag_builder = ds_train_full.lag_builder

    # Val y Test reutilizan scalers de train
    ds_val = _make_dataset(df_va, planta_target, feature_cols,
                            future_feature_cols, lag_steps, seq_length,
                            scaler_x=scaler_x, scaler_y=scaler_y,
                            lag_builder=lag_builder)
    ds_test = _make_dataset(df_te, planta_target, feature_cols,
                             future_feature_cols, lag_steps, seq_length,
                             scaler_x=scaler_x, scaler_y=scaler_y,
                             lag_builder=lag_builder)

    val_loader  = DataLoader(ds_val,  batch_size=batch_size,
                             shuffle=False, drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    # Fine-tuning loader
    ft_loader = None
    if n_finetune_hours is not None and n_finetune_hours > 0:
        n_ft = min(n_finetune_hours + seq_length, len(df_tr))
        n_ft = max(n_ft, seq_length + 1)
        if n_ft > len(df_tr):
            print(f"  ⚠ {planta_target}: datos insuficientes para fine-tuning "
                  f"({len(df_tr)} filas disponibles, necesita {seq_length + 1}). "
                  "ft_loader=None")
        else:
            df_ft  = df_tr.iloc[:n_ft]
            ds_ft  = _make_dataset(df_ft, planta_target, feature_cols,
                                    future_feature_cols, lag_steps, seq_length,
                                    scaler_x=scaler_x, scaler_y=scaler_y,
                                    lag_builder=lag_builder)
            ft_batch  = min(batch_size, len(ds_ft))
            ft_loader = DataLoader(ds_ft, batch_size=ft_batch,
                                   shuffle=True, drop_last=False)
            print(f"  {planta_target} fine-tune: {len(ds_ft):,} muestras "
                  f"({n_ft} filas de train, {n_finetune_hours}h solicitadas, "
                  f"batch_size={ft_batch})")
    elif n_finetune_hours == 0:
        print(f"  {planta_target}: zero-shot (sin fine-tuning)")
    else:
        print(f"  {planta_target} baseline: {len(ds_train_full):,} muestras "
              f"(todo el train)")
        ft_batch  = min(batch_size, len(ds_train_full))
        ft_loader = DataLoader(ds_train_full, batch_size=ft_batch,
                               shuffle=True, drop_last=True)

    print(f"  val={len(ds_val):,}  test={len(ds_test):,}  "
          f"scaler_mean={scaler_y.mean_:.1f}")
    return ft_loader, val_loader, test_loader, ds_test, scaler_y


# ─────────────────────────────────────────────────────────────────────────────
# Métricas y guardado de resultados
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: LSTMForecaster,
             test_ds: SolarSequenceDataset,
             device: str) -> dict:
    """
    Calcula MAE, RMSE, MAPE, R² en escala original (MW).

    Args:
        model:   LSTMForecaster en modo eval.
        test_ds: SolarSequenceDataset de la planta objetivo.
        device:  Dispositivo de cómputo.

    Returns:
        Dict con claves 'MAE', 'RMSE', 'MAPE', 'R2', 'n_samples'.
    """
    df_pred = test_ds.get_full_predictions(model, device=device)
    return ForecastEvaluator.calculate_metrics(
        df_pred['y_true'].values,
        df_pred['y_pred'].values,
    )


def save_results_row(results_dir: Path, scenario: str,
                     metrics: dict, n_ft_hours: int) -> None:
    """
    Append a CSV acumulativo results_dir/metrics.csv.

    Columnas: timestamp, scenario, n_ft_hours, MAE, RMSE, MAPE, R2, n_samples.
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
            df_old[fieldnames].to_csv(csv_path, index=False)
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
# LSTMTransferTrainer — wrapper de pre-entrenamiento y fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

class LSTMTransferTrainer:
    """
    Encapsula pre-entrenamiento y fine-tuning LSTM en una interfaz simple.

    Internamente usa LSTMTrainer de src/lstm_model.py.

    Estrategia de layer freezing
    ────────────────────────────
    freeze_lstm=True congela model.lstm (encoder recurrente) y deja
    entrenables: model.attention, model.fc1, model.fc2.
    Análogo a freeze_trend_seasonality en N-BEATS (capas universales → fijas;
    capas específicas de tarea → entrenables).
    """

    def __init__(self, model: LSTMForecaster, device: str):
        self.model         = model
        self.device        = device
        self.trainer       = None
        self._pretrain_state: Optional[dict] = None

    def pretrain(self, train_loader: DataLoader, val_loader: DataLoader,
                 cfg: dict) -> dict:
        """
        Pre-entrena el modelo sobre datos multi-planta (plantas fuente).

        Returns:
            Dict con 'train_losses', 'val_losses', 'best_val_loss', 'best_iter'.
        """
        self.trainer = LSTMTrainer(
            self.model,
            device        = self.device,
            learning_rate = cfg['lr'],
            loss_alpha    = cfg.get('loss_alpha', 0.7),
        )
        self.trainer.fit(
            train_loader, val_loader,
            epochs   = cfg['max_epochs'],
            patience = cfg['patience'],
            verbose  = True,
        )
        self._pretrain_state = copy.deepcopy(self.trainer.best_state)
        best_val = min(self.trainer.val_losses) if self.trainer.val_losses else float('nan')
        return {
            'fold':          1,
            'train_losses':  self.trainer.train_losses,
            'val_losses':    self.trainer.val_losses,
            'best_val_loss': best_val,
            'best_iter':     len(self.trainer.train_losses),
            'n_train':       len(train_loader.dataset),
            'n_val':         len(val_loader.dataset),
        }

    def finetune(self, ft_loader: DataLoader, val_loader: DataLoader,
                 cfg: dict, freeze_lstm: bool = False) -> dict:
        """
        Fine-tunea el modelo sobre datos de la planta objetivo.

        Restaura el mejor state del pretraining, reduce LR (lr * lr_factor)
        y opcionalmente congela el encoder LSTM.

        Args:
            ft_loader:   DataLoader de fine-tuning.
            val_loader:  DataLoader de validación.
            cfg:         FINETUNE_CFG con max_epochs, patience, lr_factor, lr.
            freeze_lstm: Si True, congela model.lstm (encoder recurrente).
                         Análogo a freeze_trend_seasonality en N-BEATS.

        Returns:
            Dict con curvas de entrenamiento.
        """
        # Restaurar mejor state del pretraining
        if self._pretrain_state is not None:
            self.model.load_state_dict(self._pretrain_state)

        # Congelación opcional del encoder LSTM
        if freeze_lstm:
            for param in self.model.lstm.parameters():
                param.requires_grad = False
            n_trainable = sum(p.numel() for p in self.model.parameters()
                              if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            print(f"  [fine-tune] Congelado: lstm | "
                  f"entrenables: {n_trainable:,}/{n_total:,}")

        lr_ft = cfg.get('lr', 1e-3) * cfg['lr_factor']

        self.trainer = LSTMTrainer(
            self.model,
            device        = self.device,
            learning_rate = lr_ft,
            loss_alpha    = cfg.get('loss_alpha', 0.7),
        )
        self.trainer.fit(
            ft_loader, val_loader,
            epochs   = cfg['max_epochs'],
            patience = cfg['patience'],
            verbose  = True,
        )
        best_val = min(self.trainer.val_losses) if self.trainer.val_losses else float('nan')
        return {
            'fold':          1,
            'train_losses':  self.trainer.train_losses,
            'val_losses':    self.trainer.val_losses,
            'best_val_loss': best_val,
            'best_iter':     len(self.trainer.train_losses),
            'n_train':       len(ft_loader.dataset),
            'n_val':         len(val_loader.dataset),
        }
