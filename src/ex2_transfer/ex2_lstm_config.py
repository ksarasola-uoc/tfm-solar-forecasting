"""
ex2_lstm_config.py — Configuración del Experimento 2: Transferencia LSTM.

Pre-entrenamiento sobre 5 plantas fuente y evaluación de transferibilidad
a TALASOL mediante zero-shot y fine-tuning progresivo.

Mismo patrón que ex2_nbeats_config.py.
"""

import os
import sys
from pathlib import Path

# Raíz del repositorio: src/ex2_transfer/ex2_lstm_config.py → src/ex2_transfer → src → repo_root
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
                print(f"[ex2_lstm] Entorno: Google Colab | Device: cuda ({gpu_name})")
                return 'cuda'
        except ImportError:
            pass
        print("[ex2_lstm] Entorno: Google Colab | Device: cpu (CUDA no disponible)")
        return 'cpu'

    print("[ex2_lstm] Entorno: local | Device: cpu")
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
RESULTS_DIR     = _REPO_ROOT / 'results' / 'ex2_transfer' / 'lstm'
CHECKPOINTS_DIR = _REPO_ROOT / 'checkpoints' / 'ex2_transfer' / 'lstm'

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = _resolve_device()

# ── Secuencia y features ───────────────────────────────────────────────────────
SEQ_LENGTH = 48   # ventana de contexto = baseline LSTM (48 h)

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

# ── Configuración del modelo ───────────────────────────────────────────────────
MODEL_CFG = dict(
    hidden_size = 16  if SMOKE_TEST else 32,
    num_layers  = 2,
    dropout     = 0.4,
    noise_std   = 0.02,
    batch_size  = 32  if SMOKE_TEST else 64,
    max_epochs  = 2   if SMOKE_TEST else 50,
    patience    = 999 if SMOKE_TEST else 10,
    lr          = 1e-3,
    loss_alpha  = 0.7,
)

# ── Configuración del fine-tuning ──────────────────────────────────────────────
FINETUNE_CFG = dict(
    max_epochs = 2   if SMOKE_TEST else 30,
    patience   = 999 if SMOKE_TEST else 5,
    lr_factor  = 0.1,
)

# ── Tamaños de fine-tuning ─────────────────────────────────────────────────────
# En smoke: solo zero_shot + 1 semana para que sea rápido.
FINETUNE_SIZES = (
    {'zero_shot': 0, '1_semana': 168}
    if SMOKE_TEST else
    {'zero_shot': 0, '1_semana': 168, '1_mes': 720, '3_meses': 2160}
)
