"""
ex3_lstm_config.py — Configuración del Experimento 3: LSTM-Quantile.

Predicción probabilística single-plant sobre TALASOL.
Sin preentrenamiento multi-planta (foco en calibración, no transferencia).

Basado en ex2_lstm_config.py; elimina configuración de fuente/fine-tuning
y añade QUANTILES para regresión por cuantiles.
"""

import os
import sys
from pathlib import Path

# Raíz del repositorio: src/ex3_probabilistic/ex3_lstm_config.py → src/ex3_probabilistic → src → repo_root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _is_colab() -> bool:
    """Detecta si el código se ejecuta en Google Colab."""
    return 'google.colab' in sys.modules


def _resolve_device() -> str:
    """
    Selecciona el dispositivo de cómputo según el entorno de ejecución.

    - Google Colab: 'cuda' si hay GPU disponible, 'cpu' en caso contrario.
    - Local:        siempre 'cpu'.
    """
    if _is_colab():
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[ex3_lstm] Entorno: Google Colab | Device: cuda ({gpu_name})")
                return 'cuda'
        except ImportError:
            pass
        print("[ex3_lstm] Entorno: Google Colab | Device: cpu (CUDA no disponible)")
        return 'cpu'

    print("[ex3_lstm] Entorno: local | Device: cpu")
    return 'cpu'


# ── Planta objetivo ─────────────────────────────────────────────────────────
PLANTA_TARGET = 'TALASOL'

# ── Cuantiles de predicción ─────────────────────────────────────────────────
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# ── Smoke test ───────────────────────────────────────────────────────────────
SMOKE_TEST     = os.environ.get('SMOKE_TEST', '0') == '1'
SMOKE_MAX_ROWS = 700  # filas en modo smoke

# ── Paths absolutos ──────────────────────────────────────────────────────────
DATA_DIR        = _REPO_ROOT / 'data' / 'processed'
RESULTS_DIR     = _REPO_ROOT / 'results' / 'ex3_probabilistic' / 'lstm'
CHECKPOINTS_DIR = _REPO_ROOT / 'checkpoints' / 'ex3_probabilistic' / 'lstm'

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = _resolve_device()

# ── Secuencia y features ─────────────────────────────────────────────────────
SEQ_LENGTH = 48  # ventana de contexto = 48 h (consistente con Exp 1/2)

# Features históricas (entradas al encoder LSTM)
FEATURE_COLS = [
    'shortwave_radiation', 'direct_normal_irradiance',
    'global_tilted_irradiance', 'cloud_cover', 'temperature_2m',
    'elevacion_solar', 'azimut_solar', 'hora', 'dia_semana',
    'mes', 'dia_año', 'es_dia',
]

# Features del timestep objetivo (entradas al decoder)
FUTURE_FEATURE_COLS = [
    'shortwave_radiation', 'direct_normal_irradiance',
    'global_tilted_irradiance', 'cloud_cover',
    'elevacion_solar', 'hora', 'es_dia',
]

# Lags del target incluidos en la secuencia de entrada
LAG_STEPS = [1, 2, 3, 24, 48]

# n_features_total = len(FEATURE_COLS) + len(LAG_STEPS) = 12 + 5 = 17
N_FEATURES_TOTAL = len(FEATURE_COLS) + len(LAG_STEPS)

# ── Configuración del modelo ──────────────────────────────────────────────────
MODEL_CFG = dict(
    hidden_size = 16  if SMOKE_TEST else 32,
    num_layers  = 2,
    dropout     = 0.4,
    noise_std   = 0.02,
    n_quantiles = len(QUANTILES),
    batch_size  = 32  if SMOKE_TEST else 64,
    max_epochs  = 2   if SMOKE_TEST else 80,
    patience    = 999 if SMOKE_TEST else 15,
    lr          = 1e-3,
)
