"""
lgbm_data_loader.py
====================
Preparación de features tabulares para LightGBM.

LightGBM NO necesita secuencias ni DataLoaders: trabaja con una tabla
donde cada fila es un timestep y cada columna es una feature.
La información temporal se inyecta explícitamente como:
  - Lag features del target (produccion en t-1, t-2, t-3, t-24, t-48)
  - Rolling statistics (media móvil y std de las últimas N horas)
  - Features temporales cíclicas (sin/cos de hora y día del año)
  - Variables meteorológicas del instante t (conocidas en forecast)

Por qué LightGBM puede superar a Prophet con los mismos datos:
  - Captura interacciones no lineales entre variables (ej: nubosidad × irradiación)
  - No asume ninguna forma funcional de la estacionalidad
  - Los lag features le dan la autocorrelación que Prophet modela con Fourier
  - Es muy eficiente con datasets pequeños (<5000 filas por planta)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Constantes — features base del dataset
# ─────────────────────────────────────────────────────────────────────────────

BASE_METEO_COLS = [
    'shortwave_radiation',
    'direct_normal_irradiance',
    'global_tilted_irradiance',
    'cloud_cover',
    'temperature_2m',
]

BASE_SOLAR_COLS = [
    'elevacion_solar',
    'azimut_solar',
]

BASE_TIME_COLS = [
    'hora',
    'dia_semana',
    'mes',
    'dia_año',
    'es_dia',
]

LAG_STEPS_DEFAULT    = [1, 2, 3, 24, 48, 168]   # horas: tendencia, día, semana
ROLLING_WINDOWS_DEFAULT = [3, 6, 24]             # ventanas para stats móviles


# ─────────────────────────────────────────────────────────────────────────────
# LGBMFeatureBuilder — toda la ingeniería de features en un solo lugar
# ─────────────────────────────────────────────────────────────────────────────

class LGBMFeatureBuilder:
    """
    Construye la tabla de features para LightGBM a partir del DataFrame procesado.

    Uso:
        builder = LGBMFeatureBuilder()
        df_feat = builder.build(df_planta)   # df de una sola planta, ordenado por tiempo
        feature_names = builder.feature_names_

    El builder guarda los nombres de features generados para usarlos en
    la interpretación de importancias y en la validación de consistencia
    entre train y test.
    """

    def __init__(self,
                 lag_steps:      List[int] = None,
                 rolling_windows: List[int] = None,
                 target_col:     str = 'produccion',
                 add_cyclical:   bool = True):
        """
        Args:
            lag_steps       : retardos en horas del target. Por defecto [1,2,3,24,48,168].
            rolling_windows : ventanas de rolling mean/std en horas. Por defecto [3,6,24].
            target_col      : columna de producción (target).
            add_cyclical    : si True, añade sin/cos de hora y dia_año.
                              Mejora la captura de estacionalidad frente a valores enteros.
        """
        self.lag_steps       = lag_steps      or LAG_STEPS_DEFAULT
        self.rolling_windows  = rolling_windows or ROLLING_WINDOWS_DEFAULT
        self.target_col      = target_col
        self.add_cyclical    = add_cyclical
        self.feature_names_  = []   # rellenado tras build()

    def build(self, df: pd.DataFrame,
              is_train: bool = True) -> pd.DataFrame:
        """
        Construye el DataFrame de features.

        Args:
            df       : DataFrame de UNA planta, ordenado por timestamp_utc.
                       Debe contener: timestamp_utc, produccion, todas las
                       columnas de BASE_METEO/SOLAR/TIME_COLS.
            is_train : si True, elimina las filas iniciales con NaN en lags.
                       si False (test), rellena NaN con 0 para no perder filas.

        Returns:
            df_feat  : DataFrame listo para LightGBM.
                       Columnas: feature_names_ + target_col + timestamp_utc
        """
        df = df.copy().sort_values('timestamp_utc').reset_index(drop=True)
        feat_cols = []

        # ── 1. Features base (meteo, solar, temporales) ───────────────────────
        for col in BASE_METEO_COLS + BASE_SOLAR_COLS + BASE_TIME_COLS:
            if col in df.columns:
                feat_cols.append(col)

        # ── 2. Features cíclicas — sin/cos de hora y día del año ─────────────
        # Por qué: hora=23 y hora=0 son consecutivas pero numéricamente distantes.
        # sin/cos mapea el ciclo correctamente: sin(23h)≈sin(0h) en el ciclo de 24h.
        if self.add_cyclical:
            df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
            df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
            df['dia_sin']  = np.sin(2 * np.pi * df['dia_año'] / 365)
            df['dia_cos']  = np.cos(2 * np.pi * df['dia_año'] / 365)
            feat_cols += ['hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']

        # ── 3. Lag features del target ────────────────────────────────────────
        # Cada lag le dice al modelo cuánto produjo la planta N horas atrás.
        # Son las features más informativas para series con alta autocorrelación.
        for lag in self.lag_steps:
            col_name = f'lag_{lag}h'
            df[col_name] = df[self.target_col].shift(lag)
            feat_cols.append(col_name)

        # ── 4. Rolling statistics del target ─────────────────────────────────
        # Media móvil: suaviza el ruido y captura la tendencia reciente.
        # Std móvil: mide la variabilidad reciente (nubes intermitentes = std alta).
        for w in self.rolling_windows:
            col_mean = f'rolling_mean_{w}h'
            col_std  = f'rolling_std_{w}h'
            df[col_mean] = df[self.target_col].shift(1).rolling(w, min_periods=1).mean()
            df[col_std]  = df[self.target_col].shift(1).rolling(w, min_periods=1).std().fillna(0)
            feat_cols += [col_mean, col_std]

        # ── 5. Interacción irradiación × nubosidad ───────────────────────────
        # LightGBM puede aprenderlo con árboles, pero dárselo explícitamente
        # acelera la convergencia y mejora la interpretabilidad.
        if 'shortwave_radiation' in df.columns and 'cloud_cover' in df.columns:
            df['irrad_x_cloud'] = (
                df['shortwave_radiation'] * (1 - df['cloud_cover'] / 100)
            )
            feat_cols.append('irrad_x_cloud')

        # ── 6. Ratio GTI / DNI (eficiencia angular de los paneles) ──────────
        if 'global_tilted_irradiance' in df.columns and 'direct_normal_irradiance' in df.columns:
            dni_safe = df['direct_normal_irradiance'].replace(0, np.nan)
            df['gti_dni_ratio'] = (df['global_tilted_irradiance'] / dni_safe).fillna(0)
            feat_cols.append('gti_dni_ratio')

        # ── Gestión de NaN ────────────────────────────────────────────────────
        if is_train:
            # En train: eliminar filas con NaN en lags (las primeras max_lag filas)
            max_lag   = max(self.lag_steps) if self.lag_steps else 0
            max_roll  = max(self.rolling_windows) if self.rolling_windows else 0
            n_drop    = max(max_lag, max_roll)
            df = df.iloc[n_drop:].reset_index(drop=True)
        else:
            # En test: rellenar NaN con 0 para no perder filas de predicción
            for col in feat_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        self.feature_names_ = [c for c in feat_cols if c in df.columns]
        return df


# ─────────────────────────────────────────────────────────────────────────────
# prepare_lgbm_data — función de conveniencia para el notebook
# ─────────────────────────────────────────────────────────────────────────────

def prepare_lgbm_data(
        df_train:    pd.DataFrame,
        df_val:      pd.DataFrame,
        df_test:     pd.DataFrame,
        planta:      str,
        builder:     Optional[LGBMFeatureBuilder] = None,
        target_col:  str = 'produccion',
        solo_diurno: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           List[str], LGBMFeatureBuilder]:
    """
    Prepara los tres splits para una planta con el mismo builder.

    Args:
        df_train, df_val, df_test : DataFrames completos (todas las plantas)
        planta        : nombre de la planta a filtrar
        builder       : LGBMFeatureBuilder (None = crear uno nuevo)
        target_col    : columna de producción
        solo_diurno   : si True, filtra horas nocturnas (es_dia==0).
                        Recomendado para métricas comparables con Prophet.

    Returns:
        (df_tr, df_va, df_te, feature_names, builder)
    """
    if builder is None:
        builder = LGBMFeatureBuilder(target_col=target_col)

    def _filter(df):
        d = df[df['planta_nombre'] == planta].copy()
        if solo_diurno:
            d = d[d['es_dia'] == 1]
        return d.sort_values('timestamp_utc').reset_index(drop=True)

    df_tr_raw = _filter(df_train)
    df_va_raw = _filter(df_val)
    df_te_raw = _filter(df_test)

    # Build: train elimina NaN, val/test los rellena con 0
    df_tr = builder.build(df_tr_raw, is_train=True)
    df_va = builder.build(df_va_raw, is_train=False)
    df_te = builder.build(df_te_raw, is_train=False)

    feat = builder.feature_names_

    print(f"\nFeatures preparadas para {planta}:")
    print(f"  Train: {len(df_tr):,} filas × {len(feat)} features")
    print(f"  Val:   {len(df_va):,} filas × {len(feat)} features")
    print(f"  Test:  {len(df_te):,} filas × {len(feat)} features")
    print(f"  Features: {feat}")

    return df_tr, df_va, df_te, feat, builder
