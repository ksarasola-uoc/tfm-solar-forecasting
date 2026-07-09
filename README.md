# TFM: Forecasting Solar Fotovoltaico

Predicción horaria de producción de plantas solares fotovoltaicas mediante distintas
familias de modelos (estadísticos, gradient boosting, deep learning), con evaluación
de transferibilidad entre plantas, cuantificación de incertidumbre y traducción a
valor económico.

## Dataset

- **Plantas:** 5 instalaciones solares en España (FVARENA, RODRI, TALASOL, UFMULA, UPGARNACHA)
- **Periodo:** Sep 2024 – Dic 2025 (FVARENA, RODRI, TALASOL y UFMULA incluyen Sep-Dic 2024; UPGARNACHA solo 2025)
- **Resolución:** Horaria
- **Registros totales:** ~55,000
- **Variables:** producción (`quantity_sum`) + variables meteorológicas (temperatura, irradiancia, nubosidad, viento, humedad, precipitación)

## División temporal

División temporal fija (no aleatoria, para respetar la naturaleza secuencial de los datos):

- **Train:** Sep 2024 – 12 Sep 2025 (todo el histórico disponible hasta el corte)
- **Val:** 13 Sep – 31 Oct 2025 (~1.5 meses)
- **Test:** 01 Nov – 31 Dic 2025 (~2 meses)

## Fases del proyecto

El trabajo está organizado en cuatro bloques, cada uno respondiendo a una pregunta distinta:

1. **`ex1_compare`** — Comparativa de modelos: EDA/preprocesado, y baselines de Prophet, LightGBM, LSTM, N-BEATS y TFT (Temporal Fusion Transformer), con comparación final de resultados.
2. **`ex2_transfer`** — Transferibilidad espacial: ¿un modelo entrenado en unas plantas generaliza a una planta no vista? Fine-tuning con distintas cantidades de datos objetivo (zero-shot, 1 semana, 1 mes, 3 meses).
3. **`ex3_probabilistic`** — Forecasting probabilístico: predicciones por cuantiles (LSTM y TFT) en vez de un único valor puntual, para capturar incertidumbre.
4. **`ex4_valuation`** — Valoración económica: traducción de la precisión (y la incertidumbre) del forecast a valor económico real.

## Estructura

```
tfm-solar-forecasting/
├── config/
│   └── settings.py          # Configuración central (paths, device, hiperparámetros)
├── data/
│   └── [PLANTA]-merged-[AÑO].csv   # 9 archivos, una por planta y año disponible
├── src/
│   ├── data_processing.py, data_loaders.py, evaluation.py, models.py
│   ├── prophet_model.py, lgbm_model.py, lstm_model.py, nbeats_model.py, tft_model.py
│   ├── ex1_compare/, ex2_transfer/, ex3_probabilistic/   # utilidades específicas de cada fase
├── notebooks/
│   ├── ex1_compare/      # 00-06: EDA y comparativa de modelos
│   ├── ex2_transfer/     # 01-04: transferencia entre plantas
│   ├── ex3_probabilistic/ # 01-02: forecasting por cuantiles
│   └── ex4_valuation/    # 01-02: valoración económica
└── requirements.txt
```

## Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebooks en orden dentro de cada fase, por ejemplo:
jupyter notebook notebooks/ex1_compare/00_eda_preprocessing.ipynb
```

`config/settings.py` detecta automáticamente si se ejecuta en local o en Google Colab
y ajusta paths, device (CPU/CUDA/MPS) e hiperparámetros en consecuencia. Para una
verificación rápida del pipeline sin entrenar en el dataset completo:

```bash
SMOKE_TEST=1 jupyter notebook notebooks/ex1_compare/02_lightgbm_forecasting.ipynb
```

## Limitaciones reconocidas

1. **Validación temporal:** el test cubre solo 2 meses (nov-dic 2025)
2. **Generalización:** no se evalúa en un año futuro independiente
3. **Estacionalidad interanual:** como máximo se observa 1 ciclo anual completo (y solo en 4 de las 5 plantas)

Estas limitaciones se compensan parcialmente con:

- 5 plantas con ubicaciones distintas (diversidad espacial)
- Análisis de transferibilidad espacial (evaluación en plantas no vistas durante el entrenamiento)
- Forecasting probabilístico, que cuantifica la incertidumbre en vez de ocultarla tras un único valor puntual
