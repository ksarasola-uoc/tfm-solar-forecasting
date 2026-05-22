"""
ex3_tft_config.py — Configuración del Experimento 3: TFT-Quantile.

Predicción probabilística single-plant sobre TALASOL.
TFT ya soporta cuantiles nativamente via QuantileLoss de pytorch-forecasting.

Basado en ex2_tft_config.py; elimina configuración de fuente/fine-tuning
y añade QUANTILES para regresión por cuantiles.
"""

import os
import sys
from pathlib import Path

# Raíz del repositorio: src/ex3_probabilistic/ex3_tft_config.py → src/ex3_probabilistic → src → repo_root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _is_colab() -> bool:
    return 'google.colab' in sys.modules


def _resolve_device() -> str:
    if _is_colab():
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[ex3_tft] Entorno: Google Colab | Device: cuda ({gpu_name})")
                return 'cuda'
        except ImportError:
            pass
        print("[ex3_tft] Entorno: Google Colab | Device: cpu (CUDA no disponible)")
        return 'cpu'

    print("[ex3_tft] Entorno: local | Device: cpu")
    return 'cpu'


# ── Planta objetivo ──────────────────────────────────────────────────────────
PLANTA_TARGET = 'TALASOL'

# ── Cuantiles de predicción ──────────────────────────────────────────────────
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# ── Smoke test ────────────────────────────────────────────────────────────────
SMOKE_TEST     = os.environ.get('SMOKE_TEST', '0') == '1'
SMOKE_MAX_ROWS = 700

# ── Paths absolutos ───────────────────────────────────────────────────────────
DATA_DIR        = _REPO_ROOT / 'data' / 'processed'
RESULTS_DIR     = _REPO_ROOT / 'results'  / 'ex3_probabilistic' / 'tft'
CHECKPOINTS_DIR = _REPO_ROOT / 'checkpoints' / 'ex3_probabilistic' / 'tft'

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = _resolve_device()

# ── Longitud del encoder (contexto) ──────────────────────────────────────────
MAX_ENCODER_LENGTH = 48  # horas de contexto

# ── Features ──────────────────────────────────────────────────────────────────
# Alineadas con ex2_tft_config.py

TIME_VARYING_KNOWN_REALS = [
    'hora',
    'mes',
    'elevacion_solar',
    'es_dia',
    'estacion_encoded',
]

TIME_VARYING_UNKNOWN_REALS = [
    'shortwave_radiation',
    'direct_normal_irradiance',
    'global_tilted_irradiance',
    'cloud_cover',
    'temperature_2m',
]

STATIC_REALS = [
    'instalada_mw',
    'zona_climatica_encoded',
    'seguimiento_encoded',
]

# ── Configuración del modelo ──────────────────────────────────────────────────
MODEL_CFG = dict(
    hidden_size            = 32  if SMOKE_TEST else 64,
    attention_head_size    = 2   if SMOKE_TEST else 4,
    dropout                = 0.1,
    hidden_continuous_size = 8   if SMOKE_TEST else 16,
    lr                     = 1e-3,
    batch_size             = 32  if SMOKE_TEST else 64,
    max_epochs             = 2   if SMOKE_TEST else 40,
    patience               = 999 if SMOKE_TEST else 10,
)
