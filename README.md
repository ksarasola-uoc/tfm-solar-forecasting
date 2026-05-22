# TFM: Forecasting Solar Fotovoltaico

## Dataset

- **Plantas:** 8 instalaciones solares en España
- **Periodo:** 2025 (01 Ene - 31 Dic)
- **Zonas climáticas:** Norte, Centro, Sur
- **Resolución:** Horaria
- **Registros totales:** ~70,080

## División temporal

Dado que solo disponemos de datos de 2025, implementamos división conservadora:

- **Train:** 01 Ene - 12 Sep 2025 (~70%, 8.5 meses)
- **Val:** 13 Sep - 31 Oct 2025 (~15%, 1.5 meses)
- **Test:** 01 Nov - 31 Dic 2025 (~15%, 2 meses)

## Estructura

```
tfm-solar-forecasting/
├── data/
│   ├── [PLANTA]-merged-2025.csv  (8 archivos)
│   ├── datos_plantas.csv
│   └── processed/
├── src/
│   ├── data_processing.py
│   ├── models.py
│   └── evaluation.py
├── notebooks/
│   └── 01_baseline_prophet.ipynb
├── results/
│   ├── figures/
│   └── metrics/
└── requirements.txt
```

## Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook
jupyter notebook notebooks/01_baseline_prophet.ipynb
```

## Limitaciones reconocidas

1. **Validación temporal:** Test cubre solo 2 meses (nov-dic)
2. **Generalización:** No se evalúa en año futuro independiente
3. **Estacionalidad interanual:** Solo 1 ciclo anual observado

Estas limitaciones se compensan parcialmente con:

- 8 plantas (mayor diversidad espacial)

- 3 zonas climáticas (robustez geográfica)

- Análisis de transferibilidad espacial (plantas no vistas)
