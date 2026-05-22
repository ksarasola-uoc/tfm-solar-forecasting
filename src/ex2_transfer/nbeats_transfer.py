"""
nbeats_transfer.py — Soporte para transferencia N-BEATS entre plantas solares.

Funciones de soporte y NBeatsTransferTrainer para el Experimento 2.
No duplica lógica de nbeats_model.py ni nbeats_data_loader.py; los usa directamente.

Estrategia de normalización
────────────────────────────
- Plantas fuente: SolarScaler independiente por planta, fit en su train.
  ConcatDataset combina los datasets ya normalizados → el modelo ve
  producción normalizada [0,1] de cada planta (evita sesgo por capacidad).
- TALASOL: SolarScaler independiente, fit en su train.
  Mismo scaler para zero-shot, fine-tuning y baseline → evaluación consistente.
"""

import copy
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from nbeats_data_loader import NBeatsDataset, ExogScaler
from nbeats_model import NBeatsForecaster, NBeatsTrainer
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


def build_source_loaders(
        plantas_source: List[str],
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        exog_cols: List[str],
        seq_length: int,
        batch_size: int,
        smoke_test: bool = False,
        max_rows: int = 700,
) -> Tuple[DataLoader, DataLoader, Dict[str, SolarScaler]]:
    """
    Crea DataLoaders multi-planta para el pre-entrenamiento.

    Para cada planta fuente:
      - Filtra el split con load_plant_split().
      - Crea un SolarScaler fit en su train.
      - Construye NBeatsDataset con ese scaler.

    Combina todos los datasets con ConcatDataset → un único train/val loader.

    Args:
        plantas_source: Lista de nombres de plantas fuente.
        df_train:       DataFrame completo de train (todas las plantas).
        df_val:         DataFrame completo de val (todas las plantas).
        exog_cols:      Columnas de features externas.
        seq_length:     Longitud del lookback.
        batch_size:     Tamaño de batch.
        smoke_test:     Si True, limita filas por planta.
        max_rows:       Límite de filas por planta en smoke test.

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

        # SolarScaler fit en train de esta planta
        ds_train_p = NBeatsDataset(
            df_tr_p, planta,
            exog_cols  = exog_cols,
            seq_length = seq_length,
        )
        scaler_y = ds_train_p.scaler_y
        scalers_y[planta] = scaler_y

        # Val reutiliza el scaler de train
        ds_val_p = NBeatsDataset(
            df_va_p, planta,
            exog_cols  = exog_cols,
            seq_length = seq_length,
            scaler_y   = scaler_y,
            exog_scaler= ds_train_p.exog_scaler,
        )

        ds_trains.append(ds_train_p)
        ds_vals.append(ds_val_p)

        print(f"  {planta}: train={len(ds_train_p):,}  val={len(ds_val_p):,}  "
              f"scaler_mean={scaler_y.mean_:.1f}")

    concat_train = ConcatDataset(ds_trains)
    concat_val   = ConcatDataset(ds_vals)

    train_loader = DataLoader(concat_train, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(concat_val, batch_size=batch_size,
                              shuffle=False, drop_last=False)

    print(f"\nLoaders multi-planta: train={len(concat_train):,}  val={len(concat_val):,}")
    return train_loader, val_loader, scalers_y


def build_target_loaders(
        planta_target: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        exog_cols: List[str],
        seq_length: int,
        batch_size: int,
        n_finetune_hours: Optional[int] = None,
        smoke_test: bool = False,
        max_rows: int = 700,
) -> Tuple[Optional[DataLoader], DataLoader, DataLoader, NBeatsDataset, SolarScaler]:
    """
    Crea DataLoaders para la planta target (TALASOL).

    SolarScaler fit en TODO el train de TALASOL (no limitado a n_finetune_hours).
    Esto garantiza evaluación consistente entre zero-shot, fine-tuning y baseline.

    Args:
        planta_target:    Nombre de la planta objetivo.
        df_train:         DataFrame completo de train.
        df_val:           DataFrame completo de val.
        df_test:          DataFrame completo de test.
        exog_cols:        Columnas de features externas.
        seq_length:       Longitud del lookback.
        batch_size:       Tamaño de batch.
        n_finetune_hours: Horas de datos de train usadas para fine-tuning.
                          None = todo el train.
                          0   = no hay fine-tuning (zero-shot).
        smoke_test:       Si True, limita filas.
        max_rows:         Límite de filas por planta en smoke test.

    Returns:
        ft_loader:   DataLoader para fine-tuning (None si n_finetune_hours == 0).
        val_loader:  DataLoader de validación.
        test_loader: DataLoader de test.
        test_ds:     NBeatsDataset de test (para predict/evaluate).
        scaler_y:    SolarScaler fit en train de TALASOL.
    """
    df_tr = load_plant_split(df_train, planta_target, smoke_test, max_rows)
    df_va = load_plant_split(df_val,   planta_target, smoke_test, max_rows)
    df_te = load_plant_split(df_test,  planta_target, smoke_test, max_rows)

    # Scaler fit en TODO el train (para evaluación consistente)
    ds_train_full = NBeatsDataset(
        df_tr, planta_target,
        exog_cols  = exog_cols,
        seq_length = seq_length,
    )
    scaler_y    = ds_train_full.scaler_y
    exog_scaler = ds_train_full.exog_scaler

    # Val y Test reutilizan el scaler
    ds_val = NBeatsDataset(
        df_va, planta_target,
        exog_cols   = exog_cols,
        seq_length  = seq_length,
        scaler_y    = scaler_y,
        exog_scaler = exog_scaler,
    )
    ds_test = NBeatsDataset(
        df_te, planta_target,
        exog_cols   = exog_cols,
        seq_length  = seq_length,
        scaler_y    = scaler_y,
        exog_scaler = exog_scaler,
    )

    val_loader  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

    # Fine-tuning loader
    ft_loader = None
    if n_finetune_hours is not None and n_finetune_hours > 0:
        # n_finetune_hours = horas de TARGET data para ventanas de entrenamiento.
        # Necesitamos n_finetune_hours + seq_length filas totales para generar
        # ~n_finetune_hours ventanas deslizantes válidas.
        n_ft = min(n_finetune_hours + seq_length, len(df_tr))
        n_ft = max(n_ft, seq_length + 1)
        if n_ft > len(df_tr):
            print(f"  ⚠ {planta_target}: datos insuficientes para fine-tuning "
                  f"({len(df_tr)} filas disponibles, necesita {seq_length + 1}). "
                  "ft_loader=None")
        else:
            df_ft = df_tr.iloc[:n_ft]
            ds_ft = NBeatsDataset(
                df_ft, planta_target,
                exog_cols   = exog_cols,
                seq_length  = seq_length,
                scaler_y    = scaler_y,
                exog_scaler = exog_scaler,
            )
            # Ajustar batch_size para datasets pequeños: si hay menos muestras
            # que batch_size, drop_last=True descartaría el único batch y el
            # fine-tuning no entrenaría nada (bug con 1_semana y seq_length=168).
            ft_batch = min(batch_size, len(ds_ft))
            ft_loader = DataLoader(ds_ft, batch_size=ft_batch,
                                   shuffle=True, drop_last=False)
            print(f"  {planta_target} fine-tune: {len(ds_ft):,} muestras "
                  f"({n_ft} filas de train, {n_finetune_hours}h solicitadas, "
                  f"batch_size={ft_batch})")
    elif n_finetune_hours == 0:
        print(f"  {planta_target}: zero-shot (sin fine-tuning)")
    else:
        print(f"  {planta_target} baseline: {len(ds_train_full):,} muestras (todo el train)")
        ft_loader = DataLoader(ds_train_full, batch_size=batch_size,
                               shuffle=True, drop_last=True)

    print(f"  val={len(ds_val):,}  test={len(ds_test):,}  "
          f"scaler_mean={scaler_y.mean_:.1f}")
    return ft_loader, val_loader, test_loader, ds_test, scaler_y


# ─────────────────────────────────────────────────────────────────────────────
# Métricas y guardado de resultados
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: NBeatsForecaster,
             test_ds: NBeatsDataset,
             scaler_y: SolarScaler,
             device: str) -> dict:
    """
    Calcula MAE, RMSE, MAPE, R² en escala original (MW).

    Args:
        model:    NBeatsForecaster en modo eval.
        test_ds:  NBeatsDataset de la planta objetivo.
        scaler_y: SolarScaler para invertir la normalización.
        device:   Dispositivo de cómputo.

    Returns:
        Dict con claves 'MAE', 'RMSE', 'MAPE', 'R2', 'n_samples'.
    """
    df_pred = test_ds.get_full_predictions(model, device=device)
    metrics = ForecastEvaluator.calculate_metrics(
        df_pred['y_true'].values,
        df_pred['y_pred'].values,
    )
    return metrics


def save_results_row(results_dir: Path, scenario: str,
                     metrics: dict, n_ft_hours: int) -> None:
    """
    Append a CSV acumulativo results_dir/metrics.csv.

    Columnas: timestamp, scenario, n_ft_hours, MAE, RMSE, MAPE, R2,
              n_samples, training_time_s.

    Si el CSV ya existe con menos columnas (versión anterior), lo repara
    añadiendo las columnas faltantes antes de hacer el append.

    Args:
        results_dir: Directorio donde se escribe el CSV.
        scenario:    Nombre del escenario (e.g. 'zero_shot', 'ft_1_semana', 'baseline').
        metrics:     Dict con métricas del evaluador.
        n_ft_hours:  Horas de datos de fine-tuning (0 para zero-shot/baseline).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'metrics.csv'

    fieldnames = [
        'timestamp', 'scenario', 'n_ft_hours',
        'MAE', 'RMSE', 'MAPE', 'R2', 'n_samples', 'training_time_s',
    ]

    row = {
        'timestamp':       pd.Timestamp.now().isoformat(timespec='seconds'),
        'scenario':        scenario,
        'n_ft_hours':      n_ft_hours,
        'MAE':             round(metrics.get('MAE',  float('nan')), 4),
        'RMSE':            round(metrics.get('RMSE', float('nan')), 4),
        'MAPE':            round(metrics.get('MAPE', float('nan')), 4),
        'R2':              round(metrics.get('R2',   float('nan')), 4),
        'n_samples':       metrics.get('n_samples', 0),
        'training_time_s': round(metrics.get('training_time_s', float('nan')), 1),
    }

    # Reparar CSV si existe con columnas desfasadas (e.g. training_time_s añadido
    # posteriormente). Leemos con nombres explícitos para tolerar el header antiguo.
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(
                csv_path, names=fieldnames, skiprows=1,
            )
            missing = [c for c in fieldnames if c not in df_existing.columns]
            if missing or list(df_existing.columns) != fieldnames:
                for col in missing:
                    df_existing[col] = float('nan')
                df_existing[fieldnames].to_csv(csv_path, index=False)
        except Exception:
            # CSV irrecuperable → sobreescribir (perderíamos filas antiguas,
            # pero es mejor que bloquear la ejecución).
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
# NBeatsTransferTrainer — wrapper de pre-entrenamiento y fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsTransferTrainer:
    """
    Encapsula pre-entrenamiento y fine-tuning en una interfaz simple.

    Internamente usa NBeatsTrainer de src/nbeats_model.py.

    Atributos públicos:
        .trainer       : NBeatsTrainer activo (pretraining o fine-tuning)
        .pretrain_hist : dict con curvas de pretraining
        .finetune_hist : dict con curvas de fine-tuning (None hasta llamar a finetune())
    """

    def __init__(self, model: NBeatsForecaster, device: str):
        self.model          = model
        self.device         = device
        self.trainer        = None
        self.pretrain_hist  = None
        self.finetune_hist  = None
        # Guardamos el best_state del pretraining para poder restaurarlo
        self._pretrain_state: Optional[dict] = None

    def pretrain(self, train_loader: DataLoader, val_loader: DataLoader,
                 cfg: dict) -> dict:
        """
        Pre-entrena el modelo sobre datos multi-planta (plantas fuente).

        Args:
            train_loader: DataLoader multi-planta de entrenamiento.
            val_loader:   DataLoader multi-planta de validación.
            cfg:          MODEL_CFG con claves max_epochs, patience, lr.

        Returns:
            Dict con 'train_losses', 'val_losses', 'best_val_loss', 'best_iter'.
        """
        self.trainer = NBeatsTrainer(
            self.model,
            device        = self.device,
            learning_rate = cfg['lr'],
        )
        self.trainer.fit(
            train_loader, val_loader,
            epochs  = cfg['max_epochs'],
            patience= cfg['patience'],
            verbose = True,
        )
        self._pretrain_state = copy.deepcopy(self.trainer.best_state)
        best_val = min(self.trainer.val_losses) if self.trainer.val_losses else float('nan')
        self.pretrain_hist = {
            'fold':          1,
            'train_losses':  self.trainer.train_losses,
            'val_losses':    self.trainer.val_losses,
            'best_val_loss': best_val,
            'best_iter':     len(self.trainer.train_losses),
            'n_train':       len(train_loader.dataset),
            'n_val':         len(val_loader.dataset),
        }
        return self.pretrain_hist

    def finetune(self, ft_loader: DataLoader, val_loader: DataLoader,
                 cfg: dict, freeze_trend_seasonality: bool = False) -> dict:
        """
        Fine-tunea el modelo sobre datos de la planta objetivo.

        Restaura el mejor state del pretraining, reduce el LR (lr * lr_factor)
        y opcionalmente congela los stacks de trend y seasonality.

        Args:
            ft_loader:                DataLoader de fine-tuning.
            val_loader:               DataLoader de validación.
            cfg:                      FINETUNE_CFG con claves max_epochs, patience, lr_factor.
            freeze_trend_seasonality: Si True, congela stacks[0] (trend) y stacks[1] (seasonality).

        Returns:
            Dict con 'train_losses', 'val_losses', 'best_val_loss', 'best_iter'.
        """
        # Restaurar mejor state del pretraining
        if self._pretrain_state is not None:
            self.model.load_state_dict(self._pretrain_state)

        # Congelación opcional de los primeros stacks (trend + seasonality)
        if freeze_trend_seasonality and len(self.model.stacks) >= 2:
            for stack in list(self.model.stacks)[:2]:
                for param in stack.parameters():
                    param.requires_grad = False
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total     = sum(p.numel() for p in self.model.parameters())
            print(f"  [fine-tune] Congelados stacks 0-1 | "
                  f"entrenables: {n_trainable:,}/{n_total:,}")

        lr_ft = (self.pretrain_hist['train_losses'][0]  # ignorado
                 if False else
                 cfg.get('lr', 1e-3) * cfg['lr_factor'])

        # Usar el lr original del MODEL_CFG * lr_factor
        # (el trainer del pretraining tiene el lr ya reducido por scheduler;
        #  queremos partir del lr nominal definido en MODEL_CFG)
        self.trainer = NBeatsTrainer(
            self.model,
            device        = self.device,
            learning_rate = lr_ft,
        )
        self.trainer.fit(
            ft_loader, val_loader,
            epochs  = cfg['max_epochs'],
            patience= cfg['patience'],
            verbose = True,
        )
        best_val = min(self.trainer.val_losses) if self.trainer.val_losses else float('nan')
        self.finetune_hist = {
            'fold':          1,
            'train_losses':  self.trainer.train_losses,
            'val_losses':    self.trainer.val_losses,
            'best_val_loss': best_val,
            'best_iter':     len(self.trainer.train_losses),
            'n_train':       len(ft_loader.dataset),
            'n_val':         len(val_loader.dataset),
        }
        return self.finetune_hist

    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicciones en escala normalizada.

        Returns:
            (actuals_scaled, preds_scaled) — arrays numpy 1D.
        """
        if self.trainer is None:
            raise RuntimeError("Llama a pretrain() o finetune() antes de predict().")
        return self.trainer.predict(test_loader)
