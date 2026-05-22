"""
ex2_nbeats_config.py — Configuración del Experimento 2: Transferencia N-BEATS.

Pre-entrenamiento sobre 5 plantas fuente y evaluación de transferibilidad
a TALASOL mediante zero-shot y fine-tuning progresivo.

Mismo patrón que src/ex1_compare/ex1_config.py.

Detección automática de entorno:
  - Google Colab: usa GPU (cuda) si está disponible
  - Local:        siempre CPU
"""

import os
import sys
from pathlib import Path

# Raíz del repositorio: src/ex2_transfer/ex2_nbeats_config.py → src/ex2_transfer → src → repo_root
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
                print(f"[ex2_nbeats] Entorno: Google Colab | Device: cuda ({gpu_name})")
                return 'cuda'
        except ImportError:
            pass
        print("[ex2_nbeats] Entorno: Google Colab | Device: cpu (CUDA no disponible)")
        return 'cpu'

    print("[ex2_nbeats] Entorno: local | Device: cpu")
    return 'cpu'


# ── Plantas ────────────────────────────────────────────────────────────────────
PLANTAS_SOURCE = [
    'UPGARNACHA',   # 149 MW  tracker  Norte   (2025 only)
    'FVARENA',      # 150 MW  tracker  Centro
    'NBALBOA',      # 499 MW  fijo     Centro
    'RODRI',        # 175 MW  tracker  Sur
    'UFMULA',       # 494 MW  fijo     Sur
    #'TALASOL',      # 300 MW  fijo     Centro  (target de transferencia)
]
PLANTA_TARGET  = 'TALASOL'

# ── Smoke test ─────────────────────────────────────────────────────────────────
SMOKE_TEST     = os.environ.get('SMOKE_TEST', '0') == '1'
SMOKE_MAX_ROWS = 700  # filas por planta en modo smoke

# ── Paths absolutos ────────────────────────────────────────────────────────────
DATA_DIR        = _REPO_ROOT / 'data' / 'processed'
RESULTS_DIR     = _REPO_ROOT / 'results' / 'ex2_transfer' / 'nbeats'
CHECKPOINTS_DIR = _REPO_ROOT / 'checkpoints' / 'ex2_transfer' / 'nbeats'

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = _resolve_device()

# ── Arquitectura y entrenamiento ───────────────────────────────────────────────
SEQ_LENGTH = 168  # 1 semana de contexto horario

# Features externas en t — se pasan como x_exog a cada bloque N-BEATS.
# Mismas columnas que en ex1 N-BEATS notebook.
EXOG_COLS = [
    'shortwave_radiation',
    'direct_normal_irradiance',
    'global_tilted_irradiance',
    'cloud_cover',
    'temperature_2m',
    'elevacion_solar',
    'hora',
    'es_dia',
]

# Configuración del modelo (smoke vs. full)
MODEL_CFG = dict(
    n_units     = 64  if SMOKE_TEST else 256,
    n_layers    = 2   if SMOKE_TEST else 4,
    n_blocks    = 3,
    batch_size  = 32  if SMOKE_TEST else 128,
    max_epochs  = 2   if SMOKE_TEST else 50,
    patience    = 999 if SMOKE_TEST else 10,
    lr          = 1e-3,
)

# Configuración del fine-tuning
FINETUNE_CFG = dict(
    max_epochs = 2   if SMOKE_TEST else 30,
    patience   = 999 if SMOKE_TEST else 5,
    lr_factor  = 0.1,
)

# Tamaños de fine-tuning (en horas de datos de entrenamiento de TALASOL).
# En smoke solo zero_shot + 1 semana para que sea rápido.
FINETUNE_SIZES = (
    {'zero_shot': 0, '1_semana': 168}
    if SMOKE_TEST else
    {'zero_shot': 0, '1_semana': 168, '1_mes': 720, '3_meses': 2160}
)
