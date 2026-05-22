"""
lgbm_model.py
=============
LightGBM para forecasting de producción solar.

Por qué LightGBM puede superar a Prophet:
  - GBDT captura interacciones no lineales complejas sin que las programemos
  - Entrenamiento nativo con early stopping sobre validación
  - Feature importance nativa: sabemos exactamente qué features importan
  - Muy eficiente con pocos datos (<5000 filas): no sobreajusta como el LSTM
  - El lag t-24 le da la estacionalidad diaria que Prophet modela con Fourier

Arquitectura del pipeline:
  LGBMForecaster.fit(X_train, y_train, X_val, y_val)
    → entrena con early stopping sobre val_loss
    → guarda feature importance
  LGBMForecaster.predict(X_test)
    → predicción directa, sin inversión de scaler (trabaja en escala original)
    → clip a 0 para eliminar predicciones negativas
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Hiperparámetros por defecto — optimizados para <5000 filas por planta
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # Tipo de tarea
    'objective':    'regression_l1',  # MAE directo como objetivo de optimización
                                       # Más robusto a outliers que MSE (regression_l2)

    'metric':       'mae',            # métrica de early stopping = lo que medimos

    # Arquitectura del árbol
    'num_leaves':    31,    # árboles no muy profundos con pocos datos
    'max_depth':     -1,    # sin límite explícito (controlado por num_leaves)
    'min_data_in_leaf': 20, # mínimo de muestras por hoja — evita overfitting

    # Regularización (crítico con <5000 filas)
    'lambda_l1':    0.1,    # L1 — sparsidad en pesos de features
    'lambda_l2':    0.1,    # L2 — pesos pequeños
    'feature_fraction': 0.8,  # subsampling de features por árbol (como Random Forest)
    'bagging_fraction': 0.8,  # subsampling de filas por árbol
    'bagging_freq':     5,    # cada cuántos árboles hacer bagging

    # Velocidad de aprendizaje
    'learning_rate': 0.05,  # bajo → más árboles pero mejor generalización
    'n_estimators':  1000,  # máximo de árboles (early stopping lo reduce)

    # Reproducibilidad
    'seed':         42,
    'verbose':      -1,     # silenciar output interno de LightGBM

    # CPU
    'n_jobs':       -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: clave real de la métrica MAE en evals_result_
# ─────────────────────────────────────────────────────────────────────────────

def _get_metric_key(evals_result: dict, split: str = 'val') -> str:
    """
    Devuelve la clave con la que LightGBM registró la métrica MAE.

    LightGBM nombra MAE de forma distinta según la versión y el objective:
      - 'l1'                  → nombre interno, el más habitual con regression_l1
      - 'mae'                 → alias explícito cuando metric='mae' en versiones recientes
      - 'mean_absolute_error' → versiones antiguas (<3.x)

    Si ninguna coincide, devuelve la primera clave disponible como fallback.
    """
    if split not in evals_result:
        return 'l1'   # fallback seguro
    keys = list(evals_result[split].keys())
    for candidate in ['l1', 'mae', 'mean_absolute_error']:
        if candidate in keys:
            return candidate
    return keys[0] if keys else 'l1'


# ─────────────────────────────────────────────────────────────────────────────
# LGBMForecaster
# ─────────────────────────────────────────────────────────────────────────────

class LGBMForecaster:
    """
    Wrapper de LightGBM para forecasting de producción solar por planta.

    Diferencias respecto a Prophet y LSTM:
      - Trabaja en escala ORIGINAL (no necesita SolarScaler)
      - Predicción directa (no autoregresiva): predice t dado el estado en t
      - Feature importance nativa para análisis de qué variables importan
      - Early stopping nativo sobre val_loss

    Args:
        params        : hiperparámetros de LightGBM (None = DEFAULT_PARAMS)
        early_stopping: rondas sin mejora antes de parar
        planta_nombre : se asigna en fit()
    """

    def __init__(self,
                 params:          Optional[Dict] = None,
                 early_stopping:  int = 50):

        self.params         = {**DEFAULT_PARAMS, **(params or {})}
        self.early_stopping = early_stopping
        self.model          = None
        self.planta_nombre  = None
        self.feature_names_ = []
        self.best_iteration_ = 0

        # Historial de entrenamiento para curvas de aprendizaje
        self.evals_result_  = {}

    def fit(self,
            X_train:       pd.DataFrame,
            y_train:       pd.Series,
            X_val:         pd.DataFrame,
            y_val:         pd.Series,
            feature_names: List[str],
            planta_nombre: str,
            verbose_eval:  int = 100) -> 'LGBMForecaster':
        """
        Entrena el modelo con early stopping sobre validación.

        Args:
            X_train, y_train : features y target de entrenamiento
            X_val, y_val     : features y target de validación
            feature_names    : lista de nombres de features (para importance)
            planta_nombre    : nombre de la planta
            verbose_eval     : cada cuántas rondas mostrar progreso (0 = silencio)

        Returns:
            self (para encadenamiento)
        """
        self.planta_nombre  = planta_nombre
        self.feature_names_ = feature_names

        print(f"\nEntrenando LightGBM para {planta_nombre}...")
        print(f"  Train: {len(X_train):,} filas × {len(feature_names)} features")
        print(f"  Val:   {len(X_val):,} filas")

        dtrain = lgb.Dataset(
            X_train[feature_names], label=y_train,
            feature_name=feature_names
        )
        dval = lgb.Dataset(
            X_val[feature_names], label=y_val,
            feature_name=feature_names,
            reference=dtrain
        )

        callbacks = [
            lgb.early_stopping(self.early_stopping, verbose=False),
            lgb.log_evaluation(period=verbose_eval if verbose_eval > 0 else 0),
            lgb.record_evaluation(self.evals_result_),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 1000),
            valid_sets=[dtrain, dval],
            valid_names=['train', 'val'],
            callbacks=callbacks,
        )

        self.best_iteration_ = self.model.best_iteration
        # LightGBM nombra MAE como 'l1', 'mae' o 'mean_absolute_error' según versión
        _mkey    = _get_metric_key(self.evals_result_, split='val')
        best_val = self.evals_result_['val'][_mkey][self.best_iteration_ - 1]

        print(f"  ✓ Entrenado  |  best_iter={self.best_iteration_}  "
              f"best_val_MAE={best_val:.4f}  (metric_key='{_mkey}')")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicciones en escala original (MWh/h).

        Args:
            X : DataFrame con las mismas features que en fit()

        Returns:
            array 1D de predicciones, clippeadas a 0
        """
        if self.model is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")

        preds = self.model.predict(
            X[self.feature_names_],
            num_iteration=self.best_iteration_
        )
        return np.clip(preds, 0, None)

    def get_feature_importance(self,
                               importance_type: str = 'gain') -> pd.DataFrame:
        """
        Importancia de features del modelo entrenado.

        Args:
            importance_type : 'gain' (reducción de error por split, más informativo)
                              'split' (número de veces que se usa la feature)

        Returns:
            DataFrame ordenado de mayor a menor importancia.
        """
        if self.model is None:
            raise RuntimeError("Modelo no entrenado.")

        importances = self.model.feature_importance(importance_type=importance_type)
        df_imp = pd.DataFrame({
            'feature':    self.feature_names_,
            'importance': importances,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        df_imp['importance_pct'] = (
            df_imp['importance'] / df_imp['importance'].sum() * 100
        ).round(2)

        return df_imp


# ─────────────────────────────────────────────────────────────────────────────
# LGBMKFoldForecaster — ensemble con TimeSeriesSplit
# ─────────────────────────────────────────────────────────────────────────────

class LGBMKFoldForecaster:
    """
    Ensemble de modelos LightGBM entrenados con TimeSeriesSplit.

    Misma filosofía que train_with_kfold para LSTM:
      - K folds temporales (sin leakage futuro→pasado)
      - Ensemble ponderado por val_MAE inverso
      - Usa train+val combinados para aprovechar todos los datos

    Ventaja respecto al LSTM KFold:
      - Cada fold entrena en segundos (no minutos)
      - El early stopping de LGBM es más preciso que el de LSTM
      - Con pocos datos, LGBM generaliza mejor que LSTM por defecto

    Args:
        n_splits      : número de folds (recomendado 5 con LGBM, es rápido)
        lgbm_params   : hiperparámetros LightGBM
        early_stopping: rondas sin mejora
    """

    def __init__(self,
                 n_splits:      int  = 5,
                 lgbm_params:   Optional[Dict] = None,
                 early_stopping: int = 50):

        self.n_splits       = n_splits
        self.lgbm_params    = lgbm_params
        self.early_stopping = early_stopping
        self.models_        = []
        self.fold_histories_ = []
        self.feature_names_ = []
        self.planta_nombre  = None

    def fit(self,
            df_trainval:   pd.DataFrame,
            feature_names: List[str],
            planta_nombre: str,
            target_col:    str = 'produccion',
            verbose_eval:  int = 0) -> 'LGBMKFoldForecaster':
        """
        Entrena K modelos sobre train+val con TimeSeriesSplit.

        Args:
            df_trainval   : DataFrame con train+val ya con features construidas
            feature_names : lista de nombres de features
            planta_nombre : nombre de la planta
            target_col    : columna target

        Returns:
            self
        """
        self.feature_names_ = feature_names
        self.planta_nombre  = planta_nombre

        tss     = TimeSeriesSplit(n_splits=self.n_splits)
        indices = np.arange(len(df_trainval))

        print(f"\nLGBM K-Fold ({self.n_splits} folds) para {planta_nombre}...")
        print(f"  Total secuencias: {len(df_trainval):,} × {len(feature_names)} features")

        for fold, (tr_idx, va_idx) in enumerate(tss.split(indices)):
            df_tr = df_trainval.iloc[tr_idx]
            df_va = df_trainval.iloc[va_idx]

            X_tr = df_tr[feature_names]
            y_tr = df_tr[target_col]
            X_va = df_va[feature_names]
            y_va = df_va[target_col]

            print(f"\n── Fold {fold+1}/{self.n_splits}  "
                  f"train={len(df_tr):,}  val={len(df_va):,} ──")

            model = LGBMForecaster(
                params=self.lgbm_params,
                early_stopping=self.early_stopping
            )
            model.fit(X_tr, y_tr, X_va, y_va,
                      feature_names=feature_names,
                      planta_nombre=planta_nombre,
                      verbose_eval=verbose_eval)

            val_mae = np.mean(np.abs(
                model.predict(X_va) - y_va.values
            ))

            self.models_.append(model)
            _mkey = _get_metric_key(model.evals_result_, split='val')
            self.fold_histories_.append({
                'fold':          fold + 1,
                'n_train':       len(df_tr),
                'n_val':         len(df_va),
                'best_iter':     model.best_iteration_,
                'best_val_mae':  val_mae,
                'train_mae_curve': model.evals_result_.get('train', {}).get(_mkey, []),
                'val_mae_curve':   model.evals_result_.get('val',   {}).get(_mkey, []),
            })
            print(f"  Fold {fold+1}: best_iter={model.best_iteration_}  "
                  f"val_MAE={val_mae:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicción ensemble ponderada por val_MAE inverso.

        Returns:
            array 1D de predicciones clippeadas a 0
        """
        val_maes = np.array([h['best_val_mae'] for h in self.fold_histories_])
        val_maes = np.clip(val_maes, 1e-10, None)
        weights  = 1.0 / val_maes
        weights  = weights / weights.sum()

        print(f"\n  Pesos ensemble {self.planta_nombre}:")
        for h, w in zip(self.fold_histories_, weights):
            print(f"    Fold {h['fold']}: val_MAE={h['best_val_mae']:.3f}  peso={w:.3f}")

        preds_all = np.array([m.predict(X) for m in self.models_])
        return np.clip(np.average(preds_all, axis=0, weights=weights), 0, None)

    def get_feature_importance(self,
                               importance_type: str = 'gain') -> pd.DataFrame:
        """
        Importancia media de features sobre todos los folds.
        """
        dfs = [m.get_feature_importance(importance_type) for m in self.models_]
        df_merged = dfs[0][['feature']].copy()
        df_merged['importance'] = np.mean(
            [d['importance'].values for d in dfs], axis=0
        )
        df_merged = df_merged.sort_values('importance', ascending=False).reset_index(drop=True)
        df_merged['importance_pct'] = (
            df_merged['importance'] / df_merged['importance'].sum() * 100
        ).round(2)
        return df_merged
