"""
Data loader para Temporal Fusion Transformer.
Crea TimeSeriesDataSet de pytorch-forecasting.
"""

import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import warnings
warnings.filterwarnings('ignore')


class TFTDatasetCreator:
    """
    Creador de datasets para Temporal Fusion Transformer.

    pytorch-forecasting requiere un formato especifico diferente al de LSTM.
    """

    def __init__(self,
                 max_encoder_length: int = 168,
                 max_prediction_length: int = 1,
                 target_col: str = 'produccion'):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.target_col = target_col

    def prepare_dataframe(self, df: pd.DataFrame, planta_nombre: str) -> pd.DataFrame:
        """
        Preparar DataFrame para pytorch-forecasting.

        pytorch-forecasting requiere:
        - time_idx: indice temporal entero incremental
        - group_ids: identificador de series (planta)

        Args:
            df: DataFrame procesado
            planta_nombre: Planta a filtrar

        Returns:
            DataFrame preparado
        """
        # Filtrar planta
        df_planta = df[df['planta_nombre'] == planta_nombre].copy()

        # Verificar que no este vacio
        if len(df_planta) == 0:
            raise ValueError(f"No hay datos para la planta {planta_nombre}")

        # Ordenar por timestamp
        df_planta = df_planta.sort_values('timestamp_utc').reset_index(drop=True)

        # Rellenar NaN en el target con 0 (pytorch-forecasting no tolera NaN en el target
        # ni en los lags derivados de él; las filas con valid_data=False se excluyen de
        # las métricas en el notebook, por lo que rellenar con 0 es seguro)
        if df_planta[self.target_col].isna().any():
            n_nan = df_planta[self.target_col].isna().sum()
            df_planta[self.target_col] = df_planta[self.target_col].fillna(0.0)
            print(f"  ⚠ {n_nan} NaN en '{self.target_col}' rellenados con 0 "
                  f"(valid_data=False → excluidos de métricas)")

        # Crear time_idx (indice temporal entero)
        df_planta['time_idx'] = np.arange(len(df_planta))

        # group_ids (identificador de serie temporal)
        df_planta['series_id'] = planta_nombre

        # Convertir timestamp a datetime si no lo es
        df_planta['timestamp'] = pd.to_datetime(df_planta['timestamp_utc'])

        return df_planta

    def create_dataset(self,
                       df: pd.DataFrame,
                       planta_nombre: str,
                       time_varying_known_reals: list,
                       time_varying_unknown_reals: list,
                       static_reals: list = None,
                       training: bool = True,
                       min_encoder_length: int = None) -> TimeSeriesDataSet:
        """
        Crea un TimeSeriesDataSet de PyTorch Forecasting.
        """

        df_prep = self.prepare_dataframe(df, planta_nombre)

        if static_reals is None:
            static_reals = []

        all_features = (time_varying_known_reals +
                        time_varying_unknown_reals +
                        static_reals +
                        [self.target_col])

        missing = set(all_features) - set(df_prep.columns)
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

        # Si no se pasa min_encoder_length, usamos max_encoder_length
        if min_encoder_length is None:
            min_encoder_length = self.max_encoder_length

        if self.target_col not in time_varying_unknown_reals:
            time_varying_unknown_reals = time_varying_unknown_reals + [self.target_col]

        dataset = TimeSeriesDataSet(
            df_prep,
            time_idx='time_idx',
            target=self.target_col,
            group_ids=['series_id'],

            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            static_reals=static_reals,

            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            min_encoder_length=min_encoder_length,

            # lags para target (1 hora atras y 24 horas atras)
            lags={
                self.target_col: [1, 24]
            },

            target_normalizer=GroupNormalizer(
                groups=['series_id'],
                transformation=None,
                center=True
            ),

            allow_missing_timesteps=True,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        return dataset

    def create_train_val_test(self,
                              df_train: pd.DataFrame,
                              df_val: pd.DataFrame,
                              df_test: pd.DataFrame,
                              planta_nombre: str,
                              time_varying_known_reals: list,
                              time_varying_unknown_reals: list,
                              static_reals: list = None):
        """
        Crea datasets de train/val/test para una planta, ajustando min_encoder_length dinamicamente.
        """

        print(f"\nCreando datasets TFT para {planta_nombre}...")

        # Train dataset (min_encoder_length fijo)
        train_dataset = self.create_dataset(
            df_train, planta_nombre,
            time_varying_known_reals,
            time_varying_unknown_reals,
            static_reals,
            training=True
        )
        print(f"  Train: {len(train_dataset):,} secuencias")

        # Calcular min_encoder_length para val/test
        val_len = len(df_val[df_val['planta_nombre'] == planta_nombre])
        test_len = len(df_test[df_test['planta_nombre'] == planta_nombre])

        min_encoder_val = min(self.max_encoder_length, max(2, val_len - 1))
        min_encoder_test = min(self.max_encoder_length, max(2, test_len - 1))

        # Val dataset
        val_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset,
            self.prepare_dataframe(df_val, planta_nombre),
            predict=False,
            stop_randomization=True,
            min_encoder_length=min_encoder_val
        )
        print(f"  Val:   {len(val_dataset):,} secuencias")

        # Test dataset
        test_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset,
            self.prepare_dataframe(df_test, planta_nombre),
            predict=False,
            stop_randomization=True,
            min_encoder_length=min_encoder_test
        )
        print(f"  Test:  {len(test_dataset):,} secuencias")

        return train_dataset, val_dataset, test_dataset
