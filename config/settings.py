"""
config/settings.py
==================
Configuración central del TFM. Detecta automáticamente si estamos
en local (CPU) o en Google Colab (GPU) y ajusta todos los paths
y parámetros de entrenamiento en consecuencia.

USO:
    from config.settings import CFG
    print(CFG.DEVICE)        # "cuda" o "cpu"
    print(CFG.DATA_PATH)     # path correcto según entorno
"""

import os
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. DETECCIÓN DE ENTORNO
# ─────────────────────────────────────────────────────────────
def _is_colab() -> bool:
    """Detecta si estamos ejecutando en Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

def _is_smoke_test() -> bool:
    """
    Activa el modo smoke test con datos reducidos.
    Controla con la variable de entorno:
        SMOKE_TEST=1 python train.py
    O en Colab: os.environ["SMOKE_TEST"] = "1"
    """
    return os.environ.get("SMOKE_TEST", "0") == "1"

def _resolve_device() -> str:
    """
    Detecta el device disponible de forma robusta.

    Distingue tres situaciones:
      1. Colab con GPU  → "cuda"   (caso ideal para entrenar)
      2. Local con GPU compatible con PyTorch instalado → "cuda"
      3. Local con GPU incompatible (GPU antigua, CUDA version mismatch)
         o sin GPU → "cpu"

    También acepta override manual via variable de entorno:
        FORCE_DEVICE=cpu  python train.py   ← fuerza CPU aunque haya GPU
        FORCE_DEVICE=cuda python train.py   ← fuerza CUDA (fallará si no hay)

    Por qué no basta con torch.cuda.is_available():
        En equipos con GPU antigua (compute capability < 3.7 aprox.),
        PyTorch moderno compila para arquitecturas más recientes y
        is_available() devuelve False aunque nvidia-smi muestre la GPU.
        Esta función informa del motivo exacto en lugar de fallar
        silenciosamente quedándose en CPU sin avisar.
    """
    # Override manual — útil para forzar CPU en pruebas locales
    forced = os.environ.get("FORCE_DEVICE", "").lower()
    if forced in ("cpu", "cuda", "mps"):
        print(f"  [device] Override manual: FORCE_DEVICE={forced}")
        return forced

    # Comprobar CUDA
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  [device] GPU detectada: {name} ({vram:.1f} GB VRAM) → usando CUDA")
        return "cuda"

    # CUDA no disponible — diagnosticar por qué
    # (solo informativo, el device seguirá siendo "cpu")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip()
            print(
                f"  [device] ⚠  GPU detectada por nvidia-smi ({gpu_info}) "
                f"pero NO compatible con PyTorch instalado.\n"
                f"           Posibles causas:\n"
                f"             · GPU antigua (compute capability < 3.7)\n"
                f"             · Versión de CUDA del driver < versión requerida por PyTorch\n"
                f"           → Usando CPU en local. En Colab se usará GPU automáticamente."
            )
        else:
            print("  [device] Sin GPU NVIDIA detectada → usando CPU")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi no instalado → definitivamente sin GPU NVIDIA
        # Comprobar Apple Silicon como alternativa
        if torch.backends.mps.is_available():
            print("  [device] Apple Silicon (MPS) detectado → usando MPS")
            return "mps"
        print("  [device] Sin GPU detectada → usando CPU")

    return "cpu"


IS_COLAB    = _is_colab()
IS_SMOKE    = _is_smoke_test()
DEVICE_STR  = _resolve_device()    # ← sustituye la línea problemática

# ─────────────────────────────────────────────────────────────
# 2. PATHS  (cambia solo DRIVE_ROOT si tu Drive está en otra ruta)
# ─────────────────────────────────────────────────────────────
if IS_COLAB:
    # Google Drive debe estar montado: drive.mount('/content/drive')
    DRIVE_ROOT   = Path("/content/drive/MyDrive/tfm")
    PROJECT_ROOT = DRIVE_ROOT
else:
    # Raíz del repositorio local (dos niveles arriba de este archivo)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
SMOKE_DATA_DIR  = PROJECT_ROOT / "data" / "smoke_test"
RESULTS_DIR     = PROJECT_ROOT / "results"
CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"

# En Colab los checkpoints van a Drive para sobrevivir desconexiones
if IS_COLAB:
    CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"

# ─────────────────────────────────────────────────────────────
# 3. NOMBRES DE PLANTAS Y FEATURES
# ─────────────────────────────────────────────────────────────
PLANTAS = ["BVNIDA1", "FCENTUR", "FVARENA"]          # ajusta si añades más

TARGET_COL = "quantity_sum"
TIME_COL   = "timestamp_utc"

# Features usadas como entrada al modelo
# Orden: temporales primero, meteorológicas después
FEATURE_COLS = [
    "hour_number",
    "temperature_2m (°C)",
    "wind_speed_10m (km/h)",
    "cloud_cover (%)",
    "cloud_cover_low (%)",
    "cloud_cover_mid (%)",
    "cloud_cover_high (%)",
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "shortwave_radiation (W/m²)",
    "direct_normal_irradiance (W/m²)",
    "global_tilted_irradiance (W/m²)",
]

ALL_COLS = [TIME_COL, TARGET_COL] + FEATURE_COLS

# ─────────────────────────────────────────────────────────────
# 4. PARÁMETROS DE SERIES TEMPORALES
# ─────────────────────────────────────────────────────────────
WINDOW_SIZE = 168     # 1 semana de histórico como entrada
HORIZON     = 1       # predecir 1 hora adelante

# ─────────────────────────────────────────────────────────────
# 5. DIVISIÓN TEMPORAL  (fracciones, se aplican a cada planta)
# ─────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15     # = 1 - TRAIN_FRAC - VAL_FRAC

# ─────────────────────────────────────────────────────────────
# 6. PARÁMETROS DE ENTRENAMIENTO
#    Dos perfiles: SMOKE (rápido, para validar pipeline)
#                  FULL  (producción, para Colab GPU)
# ─────────────────────────────────────────────────────────────
if IS_SMOKE:
    # ── Smoke test: termina en <60 segundos en CPU ──────────
    SMOKE_ROWS_PER_PLANT = 700      # ~29 días de datos

    NBEATS_CFG = dict(
        input_size          = WINDOW_SIZE,
        output_size         = HORIZON,
        num_stacks          = 2,          # mínimo funcional (prod: 30)
        num_blocks          = 1,
        num_layers          = 2,
        layer_widths        = 64,
        batch_size          = 32,
        max_epochs          = 2,          # solo para verificar el pipeline
        learning_rate       = 1e-3,
        early_stopping_patience = 999,   # desactivado en smoke
    )

    TFT_CFG = dict(
        max_encoder_length   = WINDOW_SIZE,
        max_prediction_length= HORIZON,
        hidden_size          = 16,        # mínimo funcional (prod: 64)
        lstm_layers          = 1,
        attention_head_size  = 1,
        dropout              = 0.1,
        hidden_continuous_size = 8,
        batch_size           = 32,
        max_epochs           = 2,
        learning_rate        = 1e-3,
        early_stopping_patience = 999,
    )

    TRAINER_CFG = dict(
        accelerator  = DEVICE_STR,
        devices      = 1,
        max_epochs   = 2,
        enable_progress_bar = True,
        logger       = False,
        enable_checkpointing = False,     # no guardar checkpoints en smoke
    )

else:
    # ── Full training: producción en Colab GPU ───────────────
    SMOKE_ROWS_PER_PLANT = None           # usa todos los datos

    NBEATS_CFG = dict(
        input_size          = WINDOW_SIZE,
        output_size         = HORIZON,
        num_stacks          = 30,
        num_blocks          = 1,
        num_layers          = 4,
        layer_widths        = 256,
        batch_size          = 128,
        max_epochs          = 50,
        learning_rate       = 1e-3,
        early_stopping_patience = 10,
    )

    TFT_CFG = dict(
        max_encoder_length    = WINDOW_SIZE,
        max_prediction_length = HORIZON,
        hidden_size           = 64,
        lstm_layers           = 2,
        attention_head_size   = 4,
        dropout               = 0.1,
        hidden_continuous_size= 16,
        batch_size            = 128,
        max_epochs            = 50,
        learning_rate         = 1e-3,
        early_stopping_patience = 15,
    )

    TRAINER_CFG = dict(
        accelerator  = DEVICE_STR,
        devices      = 1,
        max_epochs   = 50,
        enable_progress_bar = True,
        logger       = False,
        enable_checkpointing = True,
    )

# ─────────────────────────────────────────────────────────────
# 7. PARÁMETROS DE FINE-TUNING (EX2)
# ─────────────────────────────────────────────────────────────
FINETUNE_LR_FACTOR   = 0.1      # lr_finetune = lr_original * 0.1
FINETUNE_MAX_EPOCHS  = 30 if not IS_SMOKE else 2
FINETUNE_PATIENCE    = 5  if not IS_SMOKE else 999
FINETUNE_DATA_SIZES  = {        # horas de datos de la planta destino
    "zero_shot" : 0,
    "1_semana"  : 168,
    "1_mes"     : 720,
    "3_meses"   : 2160,
}

# ─────────────────────────────────────────────────────────────
# 8. DataLoader
# ─────────────────────────────────────────────────────────────
import platform
NUM_WORKERS = 0 if platform.system() == "Windows" else 2

# ─────────────────────────────────────────────────────────────
# 9. REPRODUCIBILIDAD
# ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# 10. OBJETO CFG — punto de entrada único
# ─────────────────────────────────────────────────────────────
class _Config:
    """Namespace único para toda la configuración."""
    # Entorno
    IS_COLAB     = IS_COLAB
    IS_SMOKE     = IS_SMOKE
    DEVICE       = DEVICE_STR

    # Paths
    PROJECT_ROOT    = PROJECT_ROOT
    DATA_DIR        = DATA_DIR
    SMOKE_DATA_DIR  = SMOKE_DATA_DIR
    RESULTS_DIR     = RESULTS_DIR
    CHECKPOINT_DIR  = CHECKPOINT_DIR

    # Dataset
    PLANTAS         = PLANTAS
    TARGET_COL      = TARGET_COL
    TIME_COL        = TIME_COL
    FEATURE_COLS    = FEATURE_COLS
    ALL_COLS        = ALL_COLS
    WINDOW_SIZE     = WINDOW_SIZE
    HORIZON         = HORIZON
    TRAIN_FRAC      = TRAIN_FRAC
    VAL_FRAC        = VAL_FRAC
    TEST_FRAC       = TEST_FRAC

    # Modelos
    NBEATS_CFG      = NBEATS_CFG
    TFT_CFG         = TFT_CFG
    TRAINER_CFG     = TRAINER_CFG

    # Smoke test
    SMOKE_ROWS_PER_PLANT = SMOKE_ROWS_PER_PLANT

    # Fine-tuning
    FINETUNE_LR_FACTOR  = FINETUNE_LR_FACTOR
    FINETUNE_MAX_EPOCHS = FINETUNE_MAX_EPOCHS
    FINETUNE_PATIENCE   = FINETUNE_PATIENCE
    FINETUNE_DATA_SIZES = FINETUNE_DATA_SIZES

    # DataLoader
    NUM_WORKERS  = NUM_WORKERS
    RANDOM_SEED  = RANDOM_SEED

    def __repr__(self):
        mode = "SMOKE TEST" if self.IS_SMOKE else "FULL TRAINING"
        env  = "Google Colab" if self.IS_COLAB else "Local"
        return (
            f"Config [{mode} | {env} | device={self.DEVICE}]\n"
            f"  DATA_DIR        : {self.DATA_DIR}\n"
            f"  CHECKPOINT_DIR  : {self.CHECKPOINT_DIR}\n"
            f"  PLANTAS         : {self.PLANTAS}\n"
            f"  WINDOW/HORIZON  : {self.WINDOW_SIZE}/{self.HORIZON}\n"
            f"  NBEATS epochs   : {self.NBEATS_CFG['max_epochs']}\n"
            f"  TFT epochs      : {self.TFT_CFG['max_epochs']}\n"
        )

CFG = _Config()

# ── Imprimir config al importar (útil para logs) ───────────
if __name__ == "__main__":
    print(CFG)
