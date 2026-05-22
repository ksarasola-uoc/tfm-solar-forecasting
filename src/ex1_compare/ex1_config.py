"""
ex1_config.py — Configuración manual del Experimento 1 (comparación de modelos).

Solo contiene parámetros del experimento (plantas, paths, smoke test).
Los hiperparámetros de cada modelo se definen en su notebook correspondiente.

Detección automática de entorno:
  - Google Colab: usa GPU (cuda) si está disponible
  - Local:        siempre CPU (GPU local no compatible con PyTorch actual)
"""

import sys
from pathlib import Path

# Raíz del repositorio: src/ex1_compare/ex1_config.py → src/ex1_compare → src → repo_root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _is_colab() -> bool:
    """Detecta si el código se ejecuta en Google Colab."""
    return 'google.colab' in sys.modules


def _resolve_device() -> str:
    """
    Selecciona el dispositivo de cómputo según el entorno de ejecución.

    - Google Colab: 'cuda' si hay GPU disponible, 'cpu' en caso contrario.
    - Local:        siempre 'cpu' (la GPU local no es compatible con PyTorch).
    """
    if _is_colab():
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[ex1] Entorno: Google Colab | Device: cuda ({gpu_name})")
                return 'cuda'
        except ImportError:
            pass
        print("[ex1] Entorno: Google Colab | Device: cpu (CUDA no disponible)")
        return 'cpu'

    print("[ex1] Entorno: local | Device: cpu")
    return 'cpu'


# Device de cómputo para el Experimento 1 (se detecta automáticamente)
DEVICE = _resolve_device()

# Plantas del experimento: 5 fuente + TALASOL (destino de transferencia)
# UPGARNACHA: solo 2025 (limitación documentada — sin datos Sep-Dic 2024)
# FVARENA sustituye a PIZARRO (590 MW) por calidad de datos: PIZARRO acumulaba
# ~540h de paradas planificadas en verano (training), inutilizando el 37% de sus
# ventanas de entrenamiento (18% anomaly rate). FVARENA tiene <2% anomalías reales.
PLANTAS = [
    'UPGARNACHA',   # 149 MW  tracker  Norte   (2025 only)
    'FVARENA',      # 150 MW  tracker  Centro
    'NBALBOA',      # 499 MW  fijo     Centro
    'RODRI',        # 175 MW  tracker  Sur
    'UFMULA',       # 494 MW  fijo     Sur
    'TALASOL',      # 300 MW  fijo     Centro  (target de transferencia)
]

# Modo smoke test (datos reducidos para validación rápida)
SMOKE_TEST = False
SMOKE_MAX_ROWS = 700  # filas por planta en modo smoke

# Paths absolutos (independientes del directorio de trabajo actual)
DATA_DIR        = _REPO_ROOT / 'data' / 'processed'
RESULTS_DIR     = _REPO_ROOT / 'results' / 'ex1_compare'
CHECKPOINTS_DIR = _REPO_ROOT / 'checkpoints' / 'ex1_compare'
