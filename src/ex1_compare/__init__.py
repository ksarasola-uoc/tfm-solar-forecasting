"""
ex1_compare — Módulo de armonización para el Experimento 1 (comparación de modelos).

Proporciona configuración centralizada, visualizaciones unificadas y gestión
de resultados consistente para los 5 modelos: Prophet, LightGBM, LSTM,
N-BEATS y TFT.
"""

from pathlib import Path

from .ex1_config import PLANTAS, SMOKE_TEST, SMOKE_MAX_ROWS, DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, DEVICE
from .visualization import plot_diagnostic, plot_timeseries, plot_model_comparison, plot_learning_curves
from .results_manager import ResultsManager

# Garantizar que los paths son siempre objetos Path, independientemente
# de si ex1_config los devuelve como str o como Path.
DATA_DIR        = Path(DATA_DIR)
RESULTS_DIR     = Path(RESULTS_DIR)
CHECKPOINTS_DIR = Path(CHECKPOINTS_DIR)
