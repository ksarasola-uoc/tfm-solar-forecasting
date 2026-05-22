# TFM — Forecasting Solar con Transfer Learning

## Estructura del proyecto

```
tfm/
├── config/
│   └── settings.py          ← Config central (detecta local vs Colab automáticamente)
├── data/
│   ├── generate_smoke_test.py  ← Genera datos reducidos para validación
│   ├── loader.py               ← Carga, split y normalización
│   └── smoke_test/             ← CSVs reducidos (700 filas, en Git)
├── models/
│   ├── nbeats.py               ← Arquitectura N-BEATS
│   └── tft_wrapper.py          ← Wrapper TFT (pytorch-forecasting)
├── experiments/
│   └── ex2_transfer/
│       ├── smoke_test_pipeline.py  ← Validación end-to-end (<90s en CPU)
│       ├── train_multiplanta.py    ← Entrenamiento LOO completo
│       ├── evaluate_zeroshot.py    ← Evaluación zero-shot
│       └── finetune.py             ← Fine-tuning progresivo
├── results/
│   └── ex2_metrics.csv         ← CSV acumulativo de métricas (en Git)
├── checkpoints/                ← .gitignore (solo en Drive/local)
└── notebooks/
    ├── EX2_colab.ipynb         ← Versión Colab (llama a los scripts)
    └── EX2_local.ipynb         ← Versión local
```

---

## Flujo de trabajo recomendado

### 1. Setup local (una sola vez)

```bash
pip install torch pytorch-forecasting pytorch-lightning scikit-learn joblib pandas
```

### 2. Generar smoke test

Copia tus CSVs reales en `data/` con el formato `{PLANTA}-merged-2025.csv`
y ejecuta:

```bash
python data/generate_smoke_test.py
```

Esto genera archivos de 700 filas en `data/smoke_test/` que se incluyen en Git.

### 3. Validar el pipeline completo en local

```bash
SMOKE_TEST=1 python experiments/ex2_transfer/smoke_test_pipeline.py
```

**Si termina sin errores → el código está listo para Colab.**
Tiempo esperado: < 90 segundos en CPU.

### 4. Subir a Colab

```python
# Celda 1: montar Drive y clonar repo
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/TU_USUARIO/tfm.git
%cd tfm
!pip install pytorch-forecasting pytorch-lightning -q
```

```python
# Celda 2: verificar que GPU está disponible
import torch
print(torch.cuda.is_available())   # debe ser True
print(torch.cuda.get_device_name(0))
```

```python
# Celda 3: smoke test en GPU (verifica que todo funciona antes del entrenamiento real)
import os
os.environ["SMOKE_TEST"] = "1"
%run experiments/ex2_transfer/smoke_test_pipeline.py
```

```python
# Celda 4: entrenamiento completo
# (eliminar SMOKE_TEST del entorno antes de ejecutar)
if "SMOKE_TEST" in os.environ:
    del os.environ["SMOKE_TEST"]

%run experiments/ex2_transfer/train_multiplanta.py
```

---

## Control del modo de ejecución

```bash
# Smoke test (datos reducidos, 2 epochs, <90s)
SMOKE_TEST=1 python experiments/ex2_transfer/smoke_test_pipeline.py

# Entrenamiento completo
python experiments/ex2_transfer/train_multiplanta.py
```

En Colab:

```python
import os
os.environ["SMOKE_TEST"] = "1"   # activar smoke
del os.environ["SMOKE_TEST"]      # desactivar (entrenamiento real)
```

---

## Añadir una planta nueva

1. Copia el CSV a `data/{PLANTA}-merged-2025.csv`
2. Añade el nombre a `CFG.PLANTAS` en `config/settings.py`
3. Regenera el smoke test: `python data/generate_smoke_test.py`
4. Verifica: `SMOKE_TEST=1 python experiments/ex2_transfer/smoke_test_pipeline.py`

---

## Métricas y resultados

Todos los experimentos añaden filas al CSV acumulativo:

```
results/ex2_metrics.csv
```

Si la sesión de Colab se corta, las métricas ya guardadas no se pierden.
Puedes reanudar el LOO desde el escenario donde se cortó.
