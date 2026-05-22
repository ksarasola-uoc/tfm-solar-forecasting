"""
Data loaders para modelos LSTM.
Prepara secuencias temporales (ventanas deslizantes).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# FIX 3: importar el nuevo scaler robusto para producción solar
from models import SolarScaler

# ============================================================================
# SequenceScaler — se mantiene para features de entrada (meteorología, etc.)
# Solo SolarScaler reemplaza al scaler del TARGET (produccion).
# ============================================================================
class SequenceScaler:
    """
    Escalador StandardScaler para las features de entrada (X).
    El target (produccion) usa SolarScaler, no este.
    """

    def __init__(self):
        self.feature_scalers = {}   # {planta: StandardScaler}
        self.feature_names = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        self.feature_names = feature_cols
        for planta in df['planta_nombre'].unique():
            df_planta = df[df['planta_nombre'] == planta]
            scaler_x = StandardScaler()
            scaler_x.fit(df_planta[feature_cols])
            self.feature_scalers[planta] = scaler_x

    def transform(self, df: pd.DataFrame, planta: str,
                  feature_cols: List[str]) -> np.ndarray:
        if planta not in self.feature_scalers:
            raise ValueError(f"Planta {planta} no vista durante fit")
        df_planta = df[df['planta_nombre'] == planta]
        return self.feature_scalers[planta].transform(df_planta[feature_cols])
    
    def inverse_transform_target(self, y_scaled: np.ndarray, planta: str) -> np.ndarray:
        """
        Desescalar predicciones.
        """
        if planta not in self.target_scalers:
            raise ValueError(f"Planta {planta} no vista durante fit")
        
        y_scaled_2d = y_scaled.reshape(-1, 1)
        y_original = self.target_scalers[planta].inverse_transform(y_scaled_2d)
        
        return y_original.flatten()

# ─────────────────────────────────────────────────────────────────────────────
# MEJORA 3 — LagFeatureBuilder
# ─────────────────────────────────────────────────────────────────────────────

class LagFeatureBuilder:
    """
    Calcula features de lag del target (produccion escalada) y las normaliza.

    Los lags se calculan DESPUÉS de que SolarScaler escala el target, para
    que los valores de lag estén en el mismo espacio que el target escalado
    y sean directamente comparables entre sí.

    Uso:
        builder = LagFeatureBuilder(lag_steps=[1, 2, 3, 24, 48])
        lag_matrix = builder.compute(y_scaled_array)
        # lag_matrix shape: (N, len(lag_steps))
        # lag_matrix[i, j] = y_scaled[i - lag_steps[j]]
        # Las primeras max(lag_steps) filas tienen NaN → se rellenan con 0
    """

    def __init__(self, lag_steps: List[int] = None):
        """
        Args:
            lag_steps: pasos de retardo a calcular. Por defecto [1, 2, 3, 24, 48].
                       Deben estar ordenados de menor a mayor.
        """
        self.lag_steps = sorted(lag_steps or [1, 2, 3, 24, 48])

    @property
    def n_lags(self) -> int:
        return len(self.lag_steps)

    def compute(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de lags.

        Args:
            y_scaled: array 1D de valores escalados (N,)

        Returns:
            lag_matrix: array 2D (N, n_lags)
                        Posiciones sin lag disponible → 0.0
        """
        N          = len(y_scaled)
        lag_matrix = np.zeros((N, self.n_lags), dtype=np.float32)

        for j, lag in enumerate(self.lag_steps):
            if lag < N:
                lag_matrix[lag:, j] = y_scaled[:-lag] if lag > 0 else y_scaled

        return lag_matrix




# ─────────────────────────────────────────────────────────────────────────────
# SolarSequenceDataset v3 — con lag features
# ─────────────────────────────────────────────────────────────────────────────

class SolarSequenceDataset(Dataset):
    """
    Dataset de ventanas deslizantes con lag features del target.

    Cambios respecto a v2:
      - Acepta lag_steps para añadir lag features a la secuencia de entrada.
        Las lags se añaden como columnas adicionales en X, por lo que
        n_features_total = len(feature_cols) + len(lag_steps).
      - El LSTMForecaster debe inicializarse con ese n_features_total.

    Compatibilidad hacia atrás:
      - lag_steps=None → sin lags, comportamiento idéntico a v2.
      - future_feature_cols=None → sin future features, comportamiento de v2.

    Args:
        df                  : DataFrame con datos de la planta
        planta_nombre       : planta a filtrar
        feature_cols        : features meteorológicas y temporales
        future_feature_cols : features del timestep t a predecir (Fix 2)
        lag_steps           : pasos de retardo del target a incluir (Mejora 3)
        target_col          : columna de producción
        seq_length          : longitud de la ventana de contexto
        scaler_x            : SequenceScaler (None = crear nuevo)
        scaler_y            : SolarScaler (None = crear nuevo)
        lag_builder         : LagFeatureBuilder (None = crear nuevo)
    """

    def __init__(self,
                 df:                   pd.DataFrame,
                 planta_nombre:        str,
                 feature_cols:         List[str],
                 future_feature_cols:  Optional[List[str]] = None,
                 lag_steps:            Optional[List[int]] = None,
                 target_col:           str = 'produccion',
                 seq_length:           int = 48,
                 scaler_x:             Optional[SequenceScaler]   = None,
                 scaler_y:             Optional[SolarScaler]      = None,
                 lag_builder:          Optional[LagFeatureBuilder] = None):

        self.planta_nombre      = planta_nombre
        self.seq_length         = seq_length
        self.feature_cols       = feature_cols
        self.future_feature_cols = future_feature_cols
        self.target_col         = target_col

        # Filtrar y ordenar
        df_p = df[df['planta_nombre'] == planta_nombre].copy()
        df_p = df_p.sort_values('timestamp_utc').reset_index(drop=True)

        # ── Scaler de features (StandardScaler) ──────────────────────────────
        if scaler_x is None:
            self.scaler_x = SequenceScaler()
            self.scaler_x.fit(df_p, feature_cols)
        else:
            self.scaler_x = scaler_x

        X_base = self.scaler_x.transform(df_p, planta_nombre, feature_cols)
        # X_base: (N, n_features)

        # ── SolarScaler del target ────────────────────────────────────────────
        if scaler_y is None:
            self.scaler_y = SolarScaler()
            self.scaler_y.fit(df_p[target_col].values)
        else:
            self.scaler_y = scaler_y

        y_scaled = self.scaler_y.transform(df_p[target_col].values)
        # y_scaled: (N,)

        # ── MEJORA 3: Lag features ────────────────────────────────────────────
        if lag_steps is not None:
            if lag_builder is None:
                self.lag_builder = LagFeatureBuilder(lag_steps)
            else:
                self.lag_builder = lag_builder

            lag_matrix = self.lag_builder.compute(y_scaled)
            # lag_matrix: (N, n_lags)

            # Concatenar lags a X: ahora X tiene n_features + n_lags columnas
            self.X = np.concatenate([X_base, lag_matrix], axis=1).astype(np.float32)
        else:
            self.lag_builder = None
            self.X = X_base.astype(np.float32)

        self.y          = y_scaled.astype(np.float32)
        self.timestamps = df_p['timestamp_utc'].values

        # ── Índices de future features en X_base (no en lags) ────────────────
        if future_feature_cols is not None:
            self.future_indices = [
                feature_cols.index(col) for col in future_feature_cols
            ]
        else:
            self.future_indices = None

        # ── Máscara valid_data: excluye ventanas con horas no válidas ─────────
        # Una ventana [i, i+seq_length] es válida solo si TODAS sus filas
        # (contexto + target) tienen valid_data=True.
        if 'valid_data' in df_p.columns:
            valid_mask = df_p['valid_data'].fillna(True).values.astype(bool)
        else:
            valid_mask = np.ones(len(df_p), dtype=bool)

        N = len(self.X)
        self.valid_indices = [
            i for i in range(N - seq_length)
            if valid_mask[i: i + seq_length + 1].all()
        ]
        self.n_samples = len(self.valid_indices)
        if self.n_samples <= 0:
            raise ValueError(
                f"Dataset muy pequeño o sin ventanas válidas: {N} filas para "
                f"seq_length={seq_length}, planta={planta_nombre}"
            )

    @property
    def n_features_total(self) -> int:
        """Número total de features incluyendo lags. Usar para LSTMForecaster."""
        return self.X.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
          Sin future features: (X_seq, y_next)
          Con future features: (X_seq, X_future, y_next)

          X_seq    : (seq_length, n_features_total)
          X_future : (n_future_features,)
          y_next   : (1,)
        """
        real_idx = self.valid_indices[idx]
        X_seq  = self.X[real_idx: real_idx + self.seq_length]
        y_next = self.y[real_idx + self.seq_length]

        if self.future_indices is not None:
            # Las future features se extraen de X_base (las primeras columnas)
            # no de las columnas de lag
            X_future = self.X[real_idx + self.seq_length][self.future_indices]
            return (
                torch.FloatTensor(X_seq),
                torch.FloatTensor(X_future),
                torch.FloatTensor([y_next])
            )
        else:
            return (
                torch.FloatTensor(X_seq),
                torch.FloatTensor([y_next])
            )

    def get_full_predictions(self, model, device: str = 'cpu') -> pd.DataFrame:
        """Predicciones completas con inversión de SolarScaler."""
        model.eval()
        preds_scaled   = []
        actuals_scaled = []
        timestamps     = []

        with torch.no_grad():
            for i in range(len(self)):
                real_idx = self.valid_indices[i]
                item = self[i]
                if len(item) == 3:
                    X_seq, X_fut, y_true = item
                    X_seq = X_seq.unsqueeze(0).to(device)
                    X_fut = X_fut.unsqueeze(0).to(device)
                    y_pred_s = model(X_seq, X_fut).cpu().item()
                else:
                    X_seq, y_true = item
                    X_seq = X_seq.unsqueeze(0).to(device)
                    y_pred_s = model(X_seq, None).cpu().item()

                preds_scaled.append(y_pred_s)
                actuals_scaled.append(y_true.item())
                timestamps.append(self.timestamps[real_idx + self.seq_length])

        y_pred = self.scaler_y.inverse_transform(np.array(preds_scaled))
        y_true = self.scaler_y.inverse_transform(np.array(actuals_scaled))

        df_res = pd.DataFrame({
            'timestamp':     timestamps,
            'y_true':        y_true,
            'y_pred':        np.clip(y_pred, 0, None),
            'planta_nombre': self.planta_nombre
        })
        return df_res


# ─────────────────────────────────────────────────────────────────────────────
# create_dataloaders v3
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(df_train:            pd.DataFrame,
                       df_val:              pd.DataFrame,
                       df_test:             pd.DataFrame,
                       planta_nombre:       str,
                       feature_cols:        List[str],
                       future_feature_cols: Optional[List[str]] = None,
                       lag_steps:           Optional[List[int]] = None,
                       seq_length:          int = 48,
                       batch_size:          int = 64
                       ) -> Tuple[DataLoader, DataLoader, DataLoader,
                                  SequenceScaler, SolarScaler,
                                  Optional[LagFeatureBuilder], int]:
    """
    Crea DataLoaders train/val/test para una planta.

    MEJORA 3: acepta lag_steps. El LagFeatureBuilder se crea una vez en train
    y se reutiliza en val y test para garantizar consistencia.

    Returns:
        (train_loader, val_loader, test_loader,
         scaler_x, scaler_y, lag_builder, n_features_total)

        n_features_total: len(feature_cols) + n_lags
                          Usar este valor en LSTMForecaster(n_features=...)
    """
    # ── Train: ajusta todos los scalers y el lag builder ─────────────────────
    train_dataset = SolarSequenceDataset(
        df_train, planta_nombre, feature_cols,
        future_feature_cols=future_feature_cols,
        lag_steps=lag_steps,
        seq_length=seq_length,
        scaler_x=None,
        scaler_y=None,
        lag_builder=None
    )

    scaler_x    = train_dataset.scaler_x
    scaler_y    = train_dataset.scaler_y
    lag_builder = train_dataset.lag_builder    # None si lag_steps=None
    n_feat_total = train_dataset.n_features_total

    # ── Val: usa scalers y lag_builder de train ───────────────────────────────
    val_dataset = SolarSequenceDataset(
        df_val, planta_nombre, feature_cols,
        future_feature_cols=future_feature_cols,
        lag_steps=lag_steps,
        seq_length=seq_length,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        lag_builder=lag_builder
    )

    # ── Test: usa scalers y lag_builder de train ──────────────────────────────
    test_dataset = SolarSequenceDataset(
        df_test, planta_nombre, feature_cols,
        future_feature_cols=future_feature_cols,
        lag_steps=lag_steps,
        seq_length=seq_length,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        lag_builder=lag_builder
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    n_lags = lag_builder.n_lags if lag_builder is not None else 0
    print(f"\nDataLoaders — {planta_nombre}:")
    print(f"  Train : {len(train_dataset):,} secuencias")
    print(f"  Val   : {len(val_dataset):,} secuencias")
    print(f"  Test  : {len(test_dataset):,} secuencias")
    print(f"  Features base: {len(feature_cols)}  |  Lags: {n_lags}  |  Total: {n_feat_total}")
    print(f"  SolarScaler → mean={scaler_y.mean_:.1f} MW, std={scaler_y.std_:.1f} MW")

    return (train_loader, val_loader, test_loader,
            scaler_x, scaler_y, lag_builder, n_feat_total)