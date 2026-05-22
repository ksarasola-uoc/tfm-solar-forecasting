"""
ex2_tft_config.py — Configuración del Experimento 2: Transferencia TFT.

Pre-entrenamiento sobre 5 plantas fuente y evaluación de transferibilidad
a TALASOL mediante zero-shot y fine-tuning progresivo.

Mismo patrón que ex2_nbeats_config.py.
"""

import os
import sys
from pathlib import Path

# Raíz del repositorio: src/ex2_transfer/ex2_tft_config.py → src/ex2_transfer → src → repo_root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _is_colab() -> bool:
    return 'google.colab' in sys.modules


def _resolve_device() -> str:
    if _is_colab():
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[ex2_tft] Entorno: Google Colab | Device: cuda ({gpu_name})")
                return 'cuda'
        except ImportError:
            pass
        print("[ex2_tft] Entorno: Google Colab | Device: cpu (CUDA no disponible)")
        return 'cpu'

    print("[ex2_tft] Entorno: local | Device: cpu")
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
SMOKE_MAX_ROWS = 700

# ── Paths absolutos ────────────────────────────────────────────────────────────
DATA_DIR        = _REPO_ROOT / 'data' / 'processed'
RESULTS_DIR     = _REPO_ROOT / 'results'  / 'ex2_transfer' / 'tft'
CHECKPOINTS_DIR = _REPO_ROOT / 'checkpoints' / 'ex2_transfer' / 'tft'

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = _resolve_device()

# ── Longitud del encoder (contexto) ────────────────────────────────────────────
# TFT es más costoso computacionalmente que N-BEATS; 48h (2 días) es un buen
# compromiso entre contexto y velocidad en CPU/GPU moderada.
MAX_ENCODER_LENGTH = 48  # horas de contexto

# ── Features ───────────────────────────────────────────────────────────────────
# Alineadas con 05_baseline_tft.ipynb, adaptadas para transfer learning.
#
# KNOWN REALS: deterministas dado el tiempo y la ubicación (sin incertidumbre
#              en horizonte futuro).
# UNKNOWN REALS: variables meteorológicas no conocidas con exactitud en el
#                futuro (asumimos disponibilidad perfecta en evaluación histórica).
# STATIC REALS: características físicas inmutables de cada planta.

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

# ── Configuración del modelo (smoke vs. full) ───────────────────────────────────
MODEL_CFG = dict(
    hidden_size            = 32  if SMOKE_TEST else 64,
    attention_head_size    = 2   if SMOKE_TEST else 4,
    dropout                = 0.1,
    hidden_continuous_size = 8   if SMOKE_TEST else 16,
    lr                     = 1e-3,
    batch_size             = 32  if SMOKE_TEST else 64,
    max_epochs             = 2   if SMOKE_TEST else 30,
    patience               = 999 if SMOKE_TEST else 7,
)

# ── Configuración del fine-tuning ───────────────────────────────────────────────
FINETUNE_CFG = dict(
    max_epochs = 2   if SMOKE_TEST else 20,
    patience   = 999 if SMOKE_TEST else 5,
    lr_factor  = 0.3,  # TFT más sensible al LR que N-BEATS
)

# ── Tamaños de fine-tuning (horas de datos de TALASOL) ─────────────────────────
FINETUNE_SIZES = (
    {'zero_shot': 0, '1_semana': 168}
    if SMOKE_TEST else
    {'zero_shot': 0, '1_semana': 168, '1_mes': 720, '3_meses': 2160}
)
