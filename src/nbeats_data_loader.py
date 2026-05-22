"""
nbeats_data_loader.py
=====================
Dataset y DataLoaders para N-BEATS.

Diferencia fundamental respecto a SolarSequenceDataset (LSTM)
─────────────────────────────────────────────────────────────
El LSTM recibe:
  X_seq (batch, seq_length, n_features) — matriz de features por timestep

N-BEATS recibe dos tensores separados:
  x_hist (batch, seq_length)  — solo el histórico del TARGET (1D)
  x_exog (batch, n_exog)      — features externas en t (meteo, hora, elevación)

Esta separación es fundamental: las bases de trend/seasonality de N-BEATS
operan directamente sobre el histórico del target. Si mezclásemos features
meteorológicas en x_hist, el stack de trend intentaría ajustar polinomios
a [produccion, cloud_cover, hora, ...] lo que no tiene sentido físico.

Las features externas condicionan el MLP de cada bloque de forma
independiente, permitiendo que la descomposición sea interpretable.

Compatibilidad con predict_ensemble() de models.py
────────────────────────────────────────────────────
predict_ensemble() accede directamente a:
  test_dataset.timestamps      ← array de timestamps
  test_dataset.seq_length      ← int, longitud del lookback
  test_dataset.scaler_y        ← SolarScaler, para invertir normalización
  test_dataset[i][-1].item()   ← último elemento del tuple = y escalado
  len(test_dataset)            ← número de muestras

NBeatsDataset expone exactamente estos atributos con los mismos nombres.

Reutiliza SolarScaler de models.py (idéntica al LSTM) para comparación justa.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from models import SolarScaler   # misma normalización que el LSTM


# ─────────────────────────────────────────────────────────────────────────────
# ExogScaler — StandardScaler para las features externas
# ─────────────────────────────────────────────────────────────────────────────

class ExogScaler:
    """
    StandardScaler por planta para features externas (x_exog).

    Análogo a SequenceScaler de data_loaders.py pero para vectores 1D
    en lugar de matrices 2D. Se ajusta en train y se reutiliza en val/test.
    """

    def __init__(self):
        self.scalers_      = {}
        self.feature_names = []

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> 'ExogScaler':
        self.feature_names = feature_cols
        for planta in df['planta_nombre'].unique():
            df_p = df[df['planta_nombre'] == planta]
            sc   = StandardScaler()
            sc.fit(df_p[feature_cols])
            self.scalers_[planta] = sc
        return self

    def transform(self, df: pd.DataFrame,
                  planta: str, feature_cols: List[str]) -> np.ndarray:
        if planta not in self.scalers_:
            raise ValueError(f"Planta '{planta}' no vista durante fit.")
        df_p = df[df['planta_nombre'] == planta]
        return self.scalers_[planta].transform(df_p[feature_cols]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# NBeatsDataset — ventanas deslizantes para N-BEATS
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsDataset(Dataset):
    """
    Dataset de ventanas deslizantes para N-BEATS.

    Por cada índice i devuelve:
      Con exog:  (x_hist, x_exog, y)
      Sin exog:  (x_hist, y)

      x_hist : FloatTensor (seq_length,)   — histórico del target escalado
      x_exog : FloatTensor (n_exog,)       — features externas en t=i+L
      y      : FloatTensor (1,)            — target escalado en t=i+L

    Atributos públicos compatibles con predict_ensemble():
      .timestamps  : array de timestamps (mismos nombres que SolarSequenceDataset)
      .seq_length  : int
      .scaler_y    : SolarScaler
      .planta_nombre : str

    Args:
        df             : DataFrame de UNA planta, ordenado por timestamp_utc
        planta_nombre  : nombre de la planta
        exog_cols      : columnas de features externas (None = univariado)
        target_col     : columna de producción
        seq_length     : longitud del lookback
        scaler_y       : SolarScaler (None = crear y ajustar sobre df)
        exog_scaler    : ExogScaler  (None = crear y ajustar sobre df)
    """

    def __init__(self,
                 df:            pd.DataFrame,
                 planta_nombre: str,
                 exog_cols:     Optional[List[str]] = None,
                 target_col:    str  = 'produccion',
                 seq_length:    int  = 48,
                 scaler_y:      Optional[SolarScaler] = None,
                 exog_scaler:   Optional[ExogScaler]  = None):

        self.planta_nombre = planta_nombre
        self.seq_length    = seq_length   # REQUERIDO por predict_ensemble
        self.exog_cols     = exog_cols or []
        self.target_col    = target_col

        # Filtrar y ordenar
        df_p = df[df['planta_nombre'] == planta_nombre].copy()
        df_p = df_p.sort_values('timestamp_utc').reset_index(drop=True)

        if len(df_p) == 0:
            raise ValueError(f"No hay datos para '{planta_nombre}'")

        # ── SolarScaler del target — idéntico al LSTM ─────────────────────────
        if scaler_y is None:
            self.scaler_y = SolarScaler()
            self.scaler_y.fit(df_p[target_col].values)
        else:
            self.scaler_y = scaler_y

        self.y_scaled  = self.scaler_y.transform(df_p[target_col].values).astype(np.float32)
        self.timestamps = df_p['timestamp_utc'].values   # REQUERIDO por predict_ensemble

        # ── ExogScaler para features externas ─────────────────────────────────
        if self.exog_cols:
            if exog_scaler is None:
                self.exog_scaler = ExogScaler()
                self.exog_scaler.fit(df_p, self.exog_cols)
            else:
                self.exog_scaler = exog_scaler
            self.X_exog = self.exog_scaler.transform(df_p, planta_nombre, self.exog_cols)
        else:
            self.exog_scaler = None
            self.X_exog      = None

        # ── Máscara valid_data: excluye ventanas con horas no válidas ─────────
        if 'valid_data' in df_p.columns:
            valid_mask = df_p['valid_data'].fillna(True).values.astype(bool)
        else:
            valid_mask = np.ones(len(self.y_scaled), dtype=bool)

        N = len(self.y_scaled)
        self.valid_indices = [
            i for i in range(N - seq_length)
            if valid_mask[i: i + seq_length + 1].all()
        ]
        self.n_samples = len(self.valid_indices)
        if self.n_samples <= 0:
            raise ValueError(
                f"Dataset demasiado pequeño o sin ventanas válidas: {N} filas "
                f"para seq_length={seq_length}, planta={planta_nombre}"
            )

    @property
    def n_exog(self) -> int:
        return len(self.exog_cols)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        IMPORTANTE: el último elemento del tuple siempre es y (FloatTensor (1,)).
        predict_ensemble() accede a test_dataset[i][-1].item() para el actual.
        """
        real_idx = self.valid_indices[idx]
        x_hist = self.y_scaled[real_idx: real_idx + self.seq_length]   # (L,)
        y_next = self.y_scaled[real_idx + self.seq_length]             # escalar

        if self.X_exog is not None:
            x_exog = self.X_exog[real_idx + self.seq_length]           # (n_exog,)
            return (
                torch.FloatTensor(x_hist),
                torch.FloatTensor(x_exog),
                torch.FloatTensor([y_next])
            )
        return (
            torch.FloatTensor(x_hist),
            torch.FloatTensor([y_next])
        )

    def get_full_predictions(self, model, device: str = 'cpu') -> pd.DataFrame:
        """
        Predicciones completas con inversión de SolarScaler.
        Misma firma que SolarSequenceDataset.get_full_predictions.
        Para uso con split fijo (sin K-Fold).
        """
        model.eval()
        preds_sc, acts_sc, timestamps = [], [], []

        with torch.no_grad():
            for i in range(len(self)):
                real_idx = self.valid_indices[i]
                item = self[i]
                if len(item) == 3:
                    xh, xe, y = item
                    xh = xh.unsqueeze(0).to(device)
                    xe = xe.unsqueeze(0).to(device)
                    p  = model(xh, xe).cpu().item()
                else:
                    xh, y = item
                    xh = xh.unsqueeze(0).to(device)
                    p  = model(xh, None).cpu().item()

                preds_sc.append(p)
                acts_sc.append(y[-1].item())
                timestamps.append(self.timestamps[real_idx + self.seq_length])

        y_pred = self.scaler_y.inverse_transform(np.array(preds_sc))
        y_true = self.scaler_y.inverse_transform(np.array(acts_sc))

        return pd.DataFrame({
            'timestamp':     timestamps,
            'y_true':        y_true,
            'y_pred':        np.clip(y_pred, 0, None),
            'planta_nombre': self.planta_nombre
        })


# ─────────────────────────────────────────────────────────────────────────────
# create_nbeats_dataloaders — función principal para el notebook
# ─────────────────────────────────────────────────────────────────────────────

def create_nbeats_dataloaders(
        df_train:      pd.DataFrame,
        df_val:        pd.DataFrame,
        df_test:       pd.DataFrame,
        planta_nombre: str,
        exog_cols:     Optional[List[str]] = None,
        target_col:    str = 'produccion',
        seq_length:    int = 48,
        batch_size:    int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader,
           None, SolarScaler, None, int]:
    """
    Crea DataLoaders train/val/test para N-BEATS.

    Devuelve la misma tupla de 7 elementos que create_dataloaders del LSTM
    para que el notebook sea lo más paralelo posible:
        (train_loader, val_loader, test_loader,
         None,         ← no hay SequenceScaler
         scaler_y,     ← SolarScaler del target
         None,         ← no hay LagFeatureBuilder
         n_exog)       ← equivale a n_features_total del LSTM

    Los scalers se ajustan en df_train y se reusan en val y test.
    """
    exog_cols = exog_cols or []

    # Train: ajusta ambos scalers
    ds_train    = NBeatsDataset(df_train, planta_nombre, exog_cols,
                                target_col, seq_length)
    scaler_y    = ds_train.scaler_y
    exog_scaler = ds_train.exog_scaler

    # Val y Test: reusan scalers de train
    ds_val  = NBeatsDataset(df_val,  planta_nombre, exog_cols, target_col, seq_length,
                            scaler_y=scaler_y, exog_scaler=exog_scaler)
    ds_test = NBeatsDataset(df_test, planta_nombre, exog_cols, target_col, seq_length,
                            scaler_y=scaler_y, exog_scaler=exog_scaler)

    train_loader = DataLoader(ds_train, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    n_exog = ds_train.n_exog
    print(f"\nNBeats DataLoaders — {planta_nombre}:")
    print(f"  Train: {len(ds_train):,}  Val: {len(ds_val):,}  Test: {len(ds_test):,}")
    print(f"  seq_length={seq_length}  n_exog={n_exog}")
    print(f"  SolarScaler → mean={scaler_y.mean_:.1f} MW  std={scaler_y.std_:.1f} MW")
    if exog_cols:
        print(f"  Exog cols: {exog_cols}")

    return train_loader, val_loader, test_loader, None, scaler_y, None, n_exog
