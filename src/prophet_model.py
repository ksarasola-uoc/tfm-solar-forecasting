"""
Prophet forecaster para produccion solar.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class ProphetForecaster:
    """
    Wrapper para Prophet adaptado a forecasting solar.

    Configuracion optimizada para dataset de 1 anio (2025).
    """

    def __init__(self, regressors: Optional[List[str]] = None):
        """
        Inicializar forecaster.

        Args:
            regressors: Lista de variables exogenas a incluir
        """
        self.regressors = regressors or [
            'shortwave_radiation',
            'direct_normal_irradiance',
            'global_tilted_irradiance',
            'cloud_cover',
            'temperature_2m'
        ]

        self.model = None
        self.planta_nombre = None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preparar datos en formato Prophet (ds, y, regressors).

        Args:
            df: DataFrame con timestamp y variables

        Returns:
            DataFrame en formato Prophet
        """
        df_prophet = pd.DataFrame()

        # Prophet requiere columnas 'ds' (timestamp) y 'y' (target)
        # IMPORTANTE: Prophet NO soporta timestamps con zona horaria
        ds = pd.to_datetime(df['timestamp_utc'])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert('UTC').dt.tz_localize(None)
        df_prophet['ds'] = ds
        df_prophet['y'] = df['produccion']  # Produccion

        # Anadir regresores
        for reg in self.regressors:
            if reg in df.columns:
                df_prophet[reg] = df[reg]
            else:
                raise ValueError(f"Regresor {reg} no encontrado en DataFrame")

        return df_prophet

    def fit(self, df_train: pd.DataFrame, planta_nombre: str):
        """
        Entrenar modelo Prophet.

        Args:
            df_train: Datos de entrenamiento
            planta_nombre: Nombre de la planta
        """
        self.planta_nombre = planta_nombre

        # Filtrar datos de esta planta
        df_planta = df_train[df_train['planta_nombre'] == planta_nombre].copy()

        # Preparar datos
        df_prophet = self.prepare_data(df_planta)

        print(f"\nEntrenando Prophet para {planta_nombre}...")
        print(f"  Registros: {len(df_prophet):,}")
        print(f"  Periodo: {df_prophet['ds'].min().date()} a {df_prophet['ds'].max().date()}")

        # Configurar modelo
        # NOTA: Con solo ~8 meses de datos, ajustamos parametros
        self.model = Prophet(
            daily_seasonality=True,           # Critico para solar
            weekly_seasonality=True,          # Captura patrones semanales
            yearly_seasonality=True,          # Con 8 meses, aprende tendencia anual
            seasonality_mode='multiplicative', # Mejor para datos solares
            changepoint_prior_scale=0.05,     # Regularizacion (evita overfitting)
            seasonality_prior_scale=10.0,     # Permite estacionalidad fuerte
            interval_width=0.95,              # Intervalos 95%
        )

        # Anadir regresores
        for reg in self.regressors:
            self.model.add_regressor(reg)

        # Entrenar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df_prophet)

        print(f"✓ Modelo entrenado")

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Hacer predicciones.

        Args:
            df_test: Datos de test

        Returns:
            DataFrame con predicciones
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")

        # Filtrar datos de esta planta
        df_planta = df_test[df_test['planta_nombre'] == self.planta_nombre].copy()

        # Preparar datos
        df_prophet = self.prepare_data(df_planta)

        print(f"\nPrediciendo para {self.planta_nombre}...")
        print(f"  Registros: {len(df_prophet):,}")
        print(f"  Periodo: {df_prophet['ds'].min().date()} a {df_prophet['ds'].max().date()}")

        # Predecir
        forecast = self.model.predict(df_prophet)

        # Extraer predicciones
        results = pd.DataFrame({
            'timestamp': df_prophet['ds'],
            'y_true': df_prophet['y'].values,
            'y_pred': forecast['yhat'].values,
            'y_pred_lower': forecast['yhat_lower'].values,
            'y_pred_upper': forecast['yhat_upper'].values,
            'planta_nombre': self.planta_nombre
        })

        # Clip predicciones negativas
        results['y_pred'] = results['y_pred'].clip(lower=0)
        results['y_pred_lower'] = results['y_pred_lower'].clip(lower=0)

        print(f"✓ Predicciones completadas")

        return results

    def get_component_importance(self) -> pd.DataFrame:
        """
        Obtener importancia de componentes del modelo.

        Returns:
            DataFrame con importancia de features
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        extra_reg_names = list(self.model.extra_regressors.keys())
        if not extra_reg_names:
            return pd.DataFrame()

        try:
            # params['beta'] contiene [fourier_seasonality_betas..., regressor_betas...]
            # Los regresores extra ocupan las últimas n entradas del array
            beta = self.model.params['beta']  # shape: (n_samples, n_betas)
            beta_mean = beta.mean(axis=0) if beta.ndim == 2 else beta.mean(axis=(0, 1))
            n_extra = len(extra_reg_names)
            reg_betas = beta_mean[-n_extra:]
            regressor_coeffs = dict(zip(extra_reg_names, reg_betas))
        except Exception as e:
            print(f"⚠️  No se pudieron extraer coeficientes de regresores: {e}")
            return pd.DataFrame()

        df_importance = pd.DataFrame({
            'feature': list(regressor_coeffs.keys()),
            'coefficient': list(regressor_coeffs.values())
        })

        df_importance['abs_coefficient'] = df_importance['coefficient'].abs()
        df_importance = df_importance.sort_values('abs_coefficient', ascending=False)

        return df_importance
