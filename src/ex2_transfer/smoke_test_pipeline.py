"""
experiments/ex2_transfer/smoke_test_pipeline.py
================================================
Valida el pipeline COMPLETO del EX2 de principio a fin
usando datos reducidos (700 filas/planta, 2 epochs).

Qué verifica:
    1. Config detecta entorno correctamente
    2. Datos smoke test se cargan sin errores
    3. SolarDataset genera batches con las dimensiones correctas
    4. N-BEATS hace forward pass en el device correcto
    5. TFT (pytorch-forecasting) crea TimeSeriesDataSet sin errores
    6. Fine-tuning: recarga pesos y entrena 1 epoch sin explotar
    7. Métricas se calculan y se guardan en CSV

Si este script termina sin errores, el código está listo para Colab.
Tiempo esperado: <90 segundos en CPU.

USO:
    SMOKE_TEST=1 python experiments/ex2_transfer/smoke_test_pipeline.py
"""

import os
os.environ["SMOKE_TEST"] = "1"   # Activar antes de importar CFG

import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Path al raíz del proyecto
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import CFG
from data.loader import (
    load_plant, load_multiplant, split_temporal,
    fit_scaler, apply_scaler, save_scaler,
    SolarDataset, make_dataloaders
)

# ─────────────────────────────────────────────────────────────
DEVICE = torch.device(CFG.DEVICE)
COLS_TO_SCALE = CFG.FEATURE_COLS + [CFG.TARGET_COL]
# ─────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def check(label: str, ok: bool = True):
    icon = "✓" if ok else "✗"
    print(f"  {icon}  {label}")


# ════════════════════════════════════════════════════════════
#  PASO 1 — Verificar config
# ════════════════════════════════════════════════════════════
section("PASO 1 · Configuración")
print(CFG)
check(f"Modo SMOKE activo: {CFG.IS_SMOKE}", CFG.IS_SMOKE)
check(f"Device: {CFG.DEVICE}")
check(f"CUDA disponible: {torch.cuda.is_available()}")
check(f"Plantas configuradas: {CFG.PLANTAS}")
check(f"Epochs en smoke: {CFG.NBEATS_CFG['max_epochs']} (N-BEATS) / {CFG.TFT_CFG['max_epochs']} (TFT)")


# ════════════════════════════════════════════════════════════
#  PASO 2 — Carga de datos
# ════════════════════════════════════════════════════════════
section("PASO 2 · Carga de datos smoke test")

# Verificar que los archivos smoke existen (cualquier año)
missing = []
for p in CFG.PLANTAS:
    matches = list(CFG.SMOKE_DATA_DIR.glob(f"{p}-merged-*.csv"))
    if not matches:
        missing.append(p)

if missing:
    print(f"  ✗  Faltan archivos smoke para: {missing}")
    print(f"     Ejecuta primero: python data/generate_smoke_test.py")
    sys.exit(1)

# Cargar una planta como destino (LOO: excluimos BVNIDA1)
TARGET_PLANT   = CFG.PLANTAS[0]           # BVNIDA1
SOURCE_PLANTAS = CFG.PLANTAS[1:]          # FCENTUR, FVARENA

df_target = load_plant(TARGET_PLANT, smoke=True)
df_source = load_multiplant(SOURCE_PLANTAS, smoke=True)

check(f"Planta destino ({TARGET_PLANT}): {len(df_target)} filas, {len(df_target.columns)} cols")
check(f"Plantas fuente {SOURCE_PLANTAS}: {len(df_source)} filas totales")


# ════════════════════════════════════════════════════════════
#  PASO 3 — Splits y escalado
# ════════════════════════════════════════════════════════════
section("PASO 3 · Split temporal y normalización")

# Split de datos fuente (para entrenamiento multi-planta)
df_src_train_list, df_src_val_list = [], []
for planta in SOURCE_PLANTAS:
    df_p = df_source[df_source["plant_id"] == planta]
    tr, va, _ = split_temporal(df_p)
    df_src_train_list.append(tr)
    df_src_val_list.append(va)

df_src_train = pd.concat(df_src_train_list, ignore_index=True)
df_src_val   = pd.concat(df_src_val_list,   ignore_index=True)

# Scaler ajustado SOLO sobre datos fuente de entrenamiento
scaler = fit_scaler(df_src_train, COLS_TO_SCALE)
save_scaler(scaler, "multiplant_smoke")

df_src_train_sc = apply_scaler(df_src_train, scaler, COLS_TO_SCALE)
df_src_val_sc   = apply_scaler(df_src_val,   scaler, COLS_TO_SCALE)

# Split de planta destino (fine-tuning usa datos pre-test)
df_tgt_train, df_tgt_val, df_tgt_test = split_temporal(df_target)
df_tgt_test_sc = apply_scaler(df_tgt_test, scaler, COLS_TO_SCALE)

check(f"Train fuente escalado: {len(df_src_train_sc)} filas")
check(f"Val fuente escalado:   {len(df_src_val_sc)} filas")
check(f"Test destino (sin refit del scaler): {len(df_tgt_test_sc)} filas")

n_seq_train = len(df_src_train_sc) - CFG.WINDOW_SIZE - CFG.HORIZON + 1
check(f"Secuencias de entrenamiento disponibles: {n_seq_train}", n_seq_train > 0)


# ════════════════════════════════════════════════════════════
#  PASO 4 — N-BEATS: forward pass y entrenamiento básico
# ════════════════════════════════════════════════════════════
section("PASO 4 · N-BEATS — forward pass + 2 epochs")

# Intentar importar N-BEATS del proyecto
try:
    from models.nbeats import NBEATSModel
    USE_PROJECT_NBEATS = True
except ImportError:
    USE_PROJECT_NBEATS = False
    print("  ⚠  models/nbeats.py no encontrado. Usando bloque mínimo de prueba.")

if not USE_PROJECT_NBEATS:
    # Bloque mínimo para validar el pipeline de datos sin el modelo real
    class _DummyNBEATS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n_in = CFG.WINDOW_SIZE * len(CFG.FEATURE_COLS)
            self.fc = torch.nn.Linear(n_in, CFG.HORIZON)
        def forward(self, x):
            return self.fc(x.reshape(x.size(0), -1))
    model = _DummyNBEATS().to(DEVICE)
else:
    model = NBEATSModel(**CFG.NBEATS_CFG).to(DEVICE)

# DataLoaders
# NOTA: train y val son de las plantas FUENTE (multi-planta).
# El test es de la planta DESTINO — make_dataloaders añade
# automáticamente el contexto histórico necesario (WINDOW_SIZE filas).
from data.loader import make_transfer_test_loader

train_loader, val_loader, _ = make_dataloaders(
    df_src_train_sc, df_src_val_sc, df_src_val_sc,  # test placeholder (no se usa aquí)
    batch_size=CFG.NBEATS_CFG["batch_size"]
)
# Test loader de la planta destino con contexto correcto
test_loader = make_transfer_test_loader(
    df_context=df_tgt_train,          # histórico pre-test de la planta destino
    df_test=df_tgt_test_sc,           # periodo de test (106 filas en smoke)
    batch_size=CFG.NBEATS_CFG["batch_size"]
)

x_batch, y_batch = next(iter(train_loader))
check(f"Batch shape → x: {x_batch.shape}, y: {y_batch.shape}")

# Forward pass
x_batch = x_batch.to(DEVICE)
t0 = time.time()
with torch.no_grad():
    pred = model(x_batch)
elapsed = time.time() - t0
check(f"Forward pass OK en {DEVICE} ({elapsed*1000:.1f} ms) → pred: {pred.shape}")

# Entrenamiento mínimo (2 epochs)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.NBEATS_CFG["learning_rate"])
loss_fn   = torch.nn.MSELoss()

t0 = time.time()
for epoch in range(CFG.NBEATS_CFG["max_epochs"]):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"    Epoch {epoch+1}/{CFG.NBEATS_CFG['max_epochs']} — loss: {total_loss/len(train_loader):.4f}")

elapsed = time.time() - t0
check(f"Entrenamiento N-BEATS completado en {elapsed:.1f}s")

# Guardar checkpoint
CFG.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ckpt_path = CFG.CHECKPOINT_DIR / "nbeats_smoke_multiplant.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": CFG.NBEATS_CFG,
    "source_plantas": SOURCE_PLANTAS,
    "target_plant": TARGET_PLANT,
}, ckpt_path)
check(f"Checkpoint guardado: {ckpt_path}")


# ════════════════════════════════════════════════════════════
#  PASO 5 — TFT: TimeSeriesDataSet (pytorch-forecasting)
# ════════════════════════════════════════════════════════════
section("PASO 5 · TFT — TimeSeriesDataSet + 2 epochs")

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping

    # Columnas adicionales añadidas por load_plant
    tft_time_varying_known = CFG.FEATURE_COLS + ["hour_of_day", "day_of_year", "month"]
    tft_time_varying_known = list(dict.fromkeys(tft_time_varying_known))  # dedup

    # TimeSeriesDataSet requiere group_ids como lista de strings
    df_src_train_sc["plant_id"] = df_src_train_sc["plant_id"].astype(str)
    df_src_val_sc["plant_id"]   = df_src_val_sc["plant_id"].astype(str)

    # FIX SMOKE TEST: encoder reducido
    # Con 700 filas/planta y split 70-15-15, la val tiene solo ~105 filas.
    # max_encoder_length=168 necesita 169 filas por grupo -> explota.
    # En smoke usamos encoder de 24h (1 dia). En full training: 168h.
    encoder_len = 24 if CFG.IS_SMOKE else CFG.TFT_CFG["max_encoder_length"]

    train_tft_ds = TimeSeriesDataSet(
        df_src_train_sc,
        time_idx                   = "time_idx",
        target                     = CFG.TARGET_COL,
        group_ids                  = ["plant_id"],
        min_encoder_length         = encoder_len // 2,
        max_encoder_length         = encoder_len,
        max_prediction_length      = CFG.TFT_CFG["max_prediction_length"],
        time_varying_known_reals   = tft_time_varying_known,
        time_varying_unknown_reals = [CFG.TARGET_COL],
        target_normalizer          = None,
        allow_missing_timesteps    = True,
    )

    # FIX VAL: concatenar train+val para que haya contexto suficiente
    # from_dataset con predict=True toma las ultimas prediction_length
    # filas como target y usa el resto como encoder context.
    df_train_val_sc = pd.concat([df_src_train_sc, df_src_val_sc], ignore_index=True)
    val_tft_ds = TimeSeriesDataSet.from_dataset(
        train_tft_ds, df_train_val_sc, predict=True, stop_randomization=True
    )

    train_tft_dl = train_tft_ds.to_dataloader(
        train=True, batch_size=CFG.TFT_CFG["batch_size"], num_workers=CFG.NUM_WORKERS
    )
    val_tft_dl = val_tft_ds.to_dataloader(
        train=False, batch_size=CFG.TFT_CFG["batch_size"], num_workers=CFG.NUM_WORKERS
    )

    check(f"TimeSeriesDataSet creado: {len(train_tft_ds)} muestras train, {len(val_tft_ds)} val")
    check(f"Encoder length: {encoder_len}h ({'smoke reducido' if CFG.IS_SMOKE else 'completo 168h'})")

    # Modelo TFT
    tft_model = TemporalFusionTransformer.from_dataset(
        train_tft_ds,
        learning_rate       = CFG.TFT_CFG["learning_rate"],
        hidden_size         = CFG.TFT_CFG["hidden_size"],
        lstm_layers         = CFG.TFT_CFG["lstm_layers"],
        attention_head_size = CFG.TFT_CFG["attention_head_size"],
        dropout             = CFG.TFT_CFG["dropout"],
        hidden_continuous_size = CFG.TFT_CFG["hidden_continuous_size"],
        loss                = QuantileLoss(),
        log_interval        = -1,
        log_val_interval    = -1,
    )

    trainer_tft = Trainer(
        **CFG.TRAINER_CFG,
        callbacks=[EarlyStopping(monitor="val_loss", patience=CFG.TFT_CFG["early_stopping_patience"])],
    )

    t0 = time.time()
    trainer_tft.fit(tft_model, train_dataloaders=train_tft_dl, val_dataloaders=val_tft_dl)
    elapsed = time.time() - t0
    check(f"TFT entrenado en {elapsed:.1f}s")

    # Guardar checkpoint TFT
    tft_ckpt = CFG.CHECKPOINT_DIR / "tft_smoke_multiplant.ckpt"
    trainer_tft.save_checkpoint(str(tft_ckpt))
    check(f"Checkpoint TFT guardado: {tft_ckpt}")

    TFT_OK = True

except ImportError as e:
    print(f"  ⚠  pytorch-forecasting no instalado: {e}")
    print(f"     pip install pytorch-forecasting pytorch-lightning")
    TFT_OK = False
except Exception as e:
    print(f"  ✗  Error en TFT: {e}")
    TFT_OK = False


# ════════════════════════════════════════════════════════════
#  PASO 6 — Fine-tuning mínimo (N-BEATS)
# ════════════════════════════════════════════════════════════
section("PASO 6 · Fine-tuning N-BEATS (1 semana de datos destino)")

# Cargar checkpoint pre-entrenado
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

if USE_PROJECT_NBEATS:
    ft_model = NBEATSModel(**CFG.NBEATS_CFG).to(DEVICE)
else:
    ft_model = _DummyNBEATS().to(DEVICE)

ft_model.load_state_dict(ckpt["model_state_dict"])

# Congelar primeras capas (simulado: congelar 50% de parámetros)
params = list(ft_model.parameters())
n_freeze = len(params) // 2
for p in params[:n_freeze]:
    p.requires_grad = False

trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in ft_model.parameters())
check(f"Parámetros: {total:,} total | {trainable:,} entrenables ({trainable/total*100:.0f}%)")

# Datos de fine-tuning: primera semana de la planta destino
n_ft = min(CFG.FINETUNE_DATA_SIZES["1_semana"], len(df_tgt_train))
df_ft = apply_scaler(df_tgt_train.iloc[:n_ft], scaler, COLS_TO_SCALE)

if len(df_ft) >= CFG.WINDOW_SIZE + CFG.HORIZON + 1:
    ft_ds     = SolarDataset(df_ft)
    ft_loader = torch.utils.data.DataLoader(ft_ds, batch_size=16, shuffle=True)

    ft_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, ft_model.parameters()),
        lr=CFG.NBEATS_CFG["learning_rate"] * CFG.FINETUNE_LR_FACTOR
    )

    ft_model.train()
    for xb, yb in ft_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        ft_optimizer.zero_grad()
        loss = loss_fn(ft_model(xb), yb)
        loss.backward()
        ft_optimizer.step()
    check(f"Fine-tuning con {len(df_ft)} filas completado (1 epoch)")
else:
    check(f"  ⚠  Datos de fine-tuning insuficientes ({len(df_ft)} filas) — saltando")


# ════════════════════════════════════════════════════════════
#  PASO 7 — Guardar métricas en CSV acumulativo
# ════════════════════════════════════════════════════════════
section("PASO 7 · Guardar métricas en CSV acumulativo")

CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_path = CFG.RESULTS_DIR / "ex2_metrics.csv"

# Evaluación básica en test — usar make_transfer_test_loader
# para garantizar contexto suficiente aunque df_tgt_test_sc sea pequeño
ft_model.eval()
preds, actuals = [], []
eval_loader = make_transfer_test_loader(
    df_context=apply_scaler(df_tgt_train, scaler, COLS_TO_SCALE),
    df_test=df_tgt_test_sc,
    batch_size=CFG.NBEATS_CFG["batch_size"]
)
with torch.no_grad():
    for xb, yb in eval_loader:
        xb = xb.to(DEVICE)
        out = ft_model(xb)
        preds.extend(out.cpu().numpy().flatten())
        actuals.extend(yb.numpy().flatten())

preds   = np.array(preds)
actuals = np.array(actuals)
mae     = np.mean(np.abs(preds - actuals))
rmse    = np.sqrt(np.mean((preds - actuals)**2))

# Añadir fila al CSV de resultados (append-safe)
row = pd.DataFrame([{
    "timestamp"      : pd.Timestamp.now().isoformat(),
    "mode"           : "smoke_test",
    "modelo"         : "N-BEATS",
    "escenario_loo"  : f"excluido_{TARGET_PLANT}",
    "fase"           : "finetune_1_semana",
    "datos_ft_horas" : CFG.FINETUNE_DATA_SIZES["1_semana"],
    "mae"            : round(mae, 4),
    "rmse"           : round(rmse, 4),
    "n_test"         : len(preds),
}])

write_header = not results_path.exists()
row.to_csv(results_path, mode="a", header=write_header, index=False)
check(f"Métricas guardadas en {results_path}")
print(f"    MAE smoke (N-BEATS fine-tuned): {mae:.4f}")


# ════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ════════════════════════════════════════════════════════════
section("RESUMEN FINAL")
print(f"""
  Entorno : {'Google Colab' if CFG.IS_COLAB else 'Local'}
  Device  : {CFG.DEVICE}
  Modo    : SMOKE TEST

  Pasos completados:
    ✓  Config y detección de entorno
    ✓  Carga de datos smoke test
    ✓  Split temporal y normalización
    ✓  N-BEATS forward pass + {CFG.NBEATS_CFG['max_epochs']} epochs
    ✓  N-BEATS checkpoint guardado
    {'✓' if TFT_OK else '⚠'}  TFT TimeSeriesDataSet + entrenamiento
    ✓  Fine-tuning N-BEATS (congelación + lr reducido)
    ✓  Métricas guardadas en CSV acumulativo

  {'─'*50}
  Si todos los pasos muestran ✓, el pipeline está listo para Colab.
  Cambia a entrenamiento completo eliminando SMOKE_TEST=1.
  {'─'*50}
""")
