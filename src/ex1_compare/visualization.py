"""
visualization.py — Funciones unificadas de visualización para el Experimento 1.

Basadas en el patrón diagnóstico del TFT (4 paneles) y el formato de
serie temporal del LightGBM/N-BEATS (panel con error absoluto).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Color fijo: Real=steelblue, Predicción=darkorange (como en TFT referencia)
COLOR_REAL = 'steelblue'
COLOR_PRED = 'darkorange'

# Colores por modelo (solo para gráficos de comparación multi-modelo)
MODEL_COLORS = {
    'prophet':  '#5C85D6',
    'lightgbm': '#FF7043',
    'lstm':     '#43A047',
    'nbeats':   '#9C27B0',
    'tft':      '#E91E63',
}


def plot_diagnostic(df_pred, planta, model_name, save_dir=None):
    """
    Gráfico diagnóstico de 4 paneles para una planta y modelo.

    Paneles:
      1. Serie temporal (primeros 200 puntos)
      2. Scatter real vs predicción
      3. Distribuciones (real vs predicción)
      4. Perfil horario medio

    Args:
        df_pred: DataFrame con columnas 'y_true', 'y_pred' y opcionalmente
                 'timestamp', 'hora'.
        planta: Nombre de la planta.
        model_name: Nombre del modelo (e.g. 'prophet', 'lstm').
        save_dir: Directorio donde guardar la figura. Si None, no guarda.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # --- Panel 1: Serie temporal (primeros 200 puntos) ---
    ax1 = axes[0, 0]
    sample = df_pred.head(200)
    x_axis = sample['timestamp'] if 'timestamp' in sample.columns else sample.index
    ax1.plot(x_axis, sample['y_true'], label='Real', alpha=0.7, color=COLOR_REAL)
    ax1.plot(x_axis, sample['y_pred'], label=model_name.upper(), alpha=0.7, color=COLOR_PRED)
    ax1.set_title(f'{planta} - Primeros 200 puntos', fontweight='bold')
    ax1.set_ylabel('Produccion (MW)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Scatter ---
    ax2 = axes[0, 1]
    max_val = max(df_pred['y_true'].max(), df_pred['y_pred'].max()) * 1.05
    ax2.scatter(df_pred['y_true'], df_pred['y_pred'], alpha=0.25, s=8, color=COLOR_PRED)
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=1.2, label='Ideal')
    # Metrics text box
    from evaluation import ForecastEvaluator
    m = ForecastEvaluator.calculate_metrics(
        df_pred['y_true'].values, df_pred['y_pred'].values
    )
    textstr = f"MAE={m['MAE']:.2f}\nRMSE={m['RMSE']:.2f}\nR2={m['R2']:.4f}"
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title('Prediccion vs Real', fontweight='bold')
    ax2.set_xlabel('y_true')
    ax2.set_ylabel('y_pred')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Distribuciones ---
    ax3 = axes[1, 0]
    ax3.hist(df_pred['y_true'], bins=50, alpha=0.5, label='Real',
             density=True, color=COLOR_REAL)
    ax3.hist(df_pred['y_pred'], bins=50, alpha=0.5, label=model_name.upper(),
             density=True, color=COLOR_PRED)
    ax3.set_title('Distribucion de valores', fontweight='bold')
    ax3.set_xlabel('Produccion (MW)')
    ax3.set_ylabel('Densidad')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Perfil horario medio ---
    ax4 = axes[1, 1]
    if 'hora' in df_pred.columns:
        hora_col = 'hora'
    elif 'timestamp' in df_pred.columns:
        hora_col = '_hora_tmp'
        df_pred = df_pred.copy()
        df_pred[hora_col] = pd.to_datetime(df_pred['timestamp']).dt.hour
    else:
        hora_col = None

    if hora_col is not None:
        hourly = df_pred.groupby(hora_col).agg({'y_true': 'mean', 'y_pred': 'mean'})
        ax4.plot(hourly.index, hourly['y_true'], marker='o', label='Real',
                 color=COLOR_REAL)
        ax4.plot(hourly.index, hourly['y_pred'], marker='o', label=model_name.upper(),
                 color=COLOR_PRED)
        ax4.set_title('Produccion media por hora', fontweight='bold')
        ax4.set_xlabel('Hora del dia')
        ax4.set_ylabel('Produccion media (MW)')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Sin columna hora', ha='center', va='center',
                 transform=ax4.transAxes)

    fig.suptitle(f'{model_name.upper()} - {planta}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f'{planta}_diagnostic.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Guardado: {path}')

    plt.show()


def plot_timeseries(df_pred, planta, model_name, save_dir=None, days=14):
    """
    Serie temporal extendida con panel de error absoluto.

    Args:
        df_pred: DataFrame con 'timestamp', 'y_true', 'y_pred'.
        planta: Nombre de la planta.
        model_name: Nombre del modelo.
        save_dir: Directorio donde guardar. Si None, no guarda.
        days: Numero de dias a mostrar (desde el final del test).
    """
    df = df_pred.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    n_horas = 24 * days
    df_viz = df.tail(n_horas)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                             gridspec_kw={'height_ratios': [3, 1]})

    # Panel superior: serie temporal
    axes[0].plot(df_viz['timestamp'], df_viz['y_true'],
                 label='Real', linewidth=1.2, color=COLOR_REAL, alpha=0.9)
    axes[0].plot(df_viz['timestamp'], df_viz['y_pred'],
                 label=model_name.upper(), linewidth=1.2, color=COLOR_PRED, alpha=0.85)
    mae_total = df['y_true'].sub(df['y_pred']).abs().mean()
    axes[0].set_ylabel('Produccion (MWh/h)', fontsize=10)
    axes[0].set_title(
        f'{planta} - Ultimos {days} dias de test  (MAE={mae_total:.2f})',
        fontsize=11, fontweight='bold'
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    # Panel inferior: error absoluto
    error = (df_viz['y_pred'] - df_viz['y_true']).abs()
    axes[1].fill_between(df_viz['timestamp'], error,
                         alpha=0.5, color='tomato', label='Error absoluto')
    axes[1].axhline(error.mean(), color='darkred', linestyle='--',
                    linewidth=1, label=f'MAE medio={error.mean():.2f}')
    axes[1].set_ylabel('|Error|', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f'{planta}_timeseries.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Guardado: {path}')

    plt.show()


def plot_model_comparison(df_metrics_current, model_name, save_dir=None):
    """
    Barras agrupadas de MAE comparando todos los modelos disponibles.
    Si los CSVs de métricas contienen la columna 'training_time_s', añade
    un segundo panel con el tiempo de entrenamiento total por modelo.

    Carga automaticamente los metrics_summary.csv de modelos previos
    desde el directorio de metricas.

    Args:
        df_metrics_current: DataFrame con metricas del modelo actual
                            (columnas: 'planta', 'MAE', ...).
        model_name: Nombre del modelo actual (e.g. 'lstm').
        save_dir: Directorio de figuras. Si None, no guarda.
    """
    if save_dir:
        metrics_dir = Path(save_dir).parent / 'metrics'
    else:
        metrics_dir = Path('../results/ex1_compare/metrics')

    all_models = {model_name: df_metrics_current}
    model_order = ['prophet', 'lightgbm', 'lstm', 'nbeats', 'tft']

    # Cargar metricas de otros modelos
    for m_name in model_order:
        if m_name == model_name:
            continue
        path = metrics_dir / f'{m_name}_metrics_summary.csv'
        if path.exists():
            try:
                df_m = pd.read_csv(path)
                if 'planta' in df_m.columns and 'MAE' in df_m.columns:
                    all_models[m_name] = df_m
            except Exception:
                pass

    if len(all_models) < 2:
        print(f'  Solo hay metricas de {model_name} — mostrando resultado individual.')

    # Plantas comunes (o todas las del modelo actual si es el único)
    if len(all_models) >= 2:
        plantas_sets = [set(df['planta'].values) for df in all_models.values()]
        plantas_common = sorted(set.intersection(*plantas_sets))
    else:
        plantas_common = sorted(df_metrics_current['planta'].values)

    if len(plantas_common) == 0:
        print('  No hay plantas comunes entre modelos para comparar.')
        return

    # Preparar datos
    ordered_models = [m for m in model_order if m in all_models]
    n_mod = len(ordered_models)
    x = np.arange(len(plantas_common))
    width = 0.8 / n_mod

    # Detectar si hay datos de tiempo de entrenamiento
    has_timing = any(
        'training_time_s' in all_models[m].columns
        for m in ordered_models
    )

    fig_h = 10 if has_timing else 6
    fig, axes = plt.subplots(
        2 if has_timing else 1, 1,
        figsize=(max(14, len(plantas_common) * 1.5), fig_h),
        gridspec_kw={'height_ratios': [3, 2]} if has_timing else None
    )
    ax = axes[0] if has_timing else axes

    # ── Panel 1: MAE por planta ───────────────────────────────────────────────
    for j, m_name in enumerate(ordered_models):
        df_m = all_models[m_name]
        df_m = df_m[df_m['planta'].isin(plantas_common)].set_index('planta')
        mae_vals = [df_m.loc[p, 'MAE'] for p in plantas_common]
        color = MODEL_COLORS.get(m_name, '#333333')
        offset = (j - n_mod / 2 + 0.5) * width
        bars = ax.bar(x + offset, mae_vals, width,
                      label=m_name.upper(), alpha=0.85, color=color, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(plantas_common, rotation=30, ha='right')
    ax.set_ylabel('MAE (MW)', fontsize=10)
    ax.set_title(f'Comparacion MAE entre modelos ({len(ordered_models)} modelos)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: Tiempo de entrenamiento total por modelo ────────────────────
    if has_timing:
        ax2 = axes[1]
        time_totals = {}
        for m_name in ordered_models:
            df_m = all_models[m_name]
            if 'training_time_s' in df_m.columns:
                total_s = df_m['training_time_s'].sum()
                time_totals[m_name] = total_s / 60  # convertir a minutos
            else:
                time_totals[m_name] = np.nan

        models_with_time = [m for m in ordered_models if not np.isnan(time_totals[m])]
        times_min = [time_totals[m] for m in models_with_time]
        colors_t = [MODEL_COLORS.get(m, '#333333') for m in models_with_time]

        bars_t = ax2.barh(
            [m.upper() for m in models_with_time],
            times_min,
            color=colors_t, alpha=0.85, edgecolor='white'
        )
        for bar, val in zip(bars_t, times_min):
            label = f'{val:.1f} min' if val >= 1 else f'{val * 60:.0f} s'
            ax2.text(bar.get_width() + max(times_min) * 0.01, bar.get_y() + bar.get_height() / 2,
                     label, va='center', fontsize=9)

        ax2.set_xlabel('Tiempo de entrenamiento total (minutos, suma de plantas)', fontsize=10)
        ax2.set_title('Coste computacional por modelo', fontsize=11, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim(0, max(times_min) * 1.2)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / 'comparison_all_models.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Guardado: {path}')

    plt.show()

    # Guardar CSV de comparacion
    if save_dir:
        df_cmp = pd.DataFrame({'planta': plantas_common})
        for m_name in ordered_models:
            df_m = all_models[m_name].set_index('planta')
            df_cmp[f'MAE_{m_name}'] = [df_m.loc[p, 'MAE'] for p in plantas_common]
            if 'training_time_s' in all_models[m_name].columns:
                df_cmp[f'time_s_{m_name}'] = [
                    all_models[m_name].set_index('planta').loc[p, 'training_time_s']
                    for p in plantas_common
                ]
        cmp_path = metrics_dir / 'comparison_all_models.csv'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        df_cmp.to_csv(cmp_path, index=False)
        print(f'  Guardado: {cmp_path}')


def plot_learning_curves(histories, planta, model_name, save_dir=None):
    """
    Curvas de aprendizaje train/val por fold (para modelos con k-fold).

    Args:
        histories: Lista de dicts con claves 'fold', 'train_mae_curve',
                   'val_mae_curve', 'best_iter', 'best_val_mae',
                   'n_train', 'n_val'.
        planta: Nombre de la planta.
        model_name: Nombre del modelo.
        save_dir: Directorio donde guardar. Si None, no guarda.
    """
    n_folds = len(histories)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), sharey=True)
    if n_folds == 1:
        axes = [axes]

    for ax, h in zip(axes, histories):
        tr_curve = h.get('train_mae_curve', h.get('train_losses', []))
        va_curve = h.get('val_mae_curve', h.get('val_losses', []))
        iters = range(1, len(tr_curve) + 1)

        ax.plot(iters, tr_curve, label='Train', linewidth=1.5, color='steelblue')
        ax.plot(iters, va_curve, label='Val', linewidth=1.5, color='coral')

        best_iter = h.get('best_iter', len(tr_curve))
        best_val = h.get('best_val_mae', h.get('best_val_loss', 0))
        ax.axvline(best_iter, color='coral', linestyle='--', alpha=0.6, linewidth=1)

        fold_num = h.get('fold', 1)
        n_tr = h.get('n_train', '?')
        n_va = h.get('n_val', '?')
        ax.set_title(
            f"Fold {fold_num}  (n={n_tr}/{n_va})\n"
            f"best_iter={best_iter}  val={best_val:.3f}",
            fontsize=9
        )
        ax.set_xlabel('Iteracion')
        if fold_num == 1 or fold_num == histories[0].get('fold', 1):
            ax.set_ylabel('Loss / MAE')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name.upper()} {planta} - Curvas de aprendizaje',
                 fontweight='bold', fontsize=11)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f'{planta}_learning_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Guardado: {path}')

    plt.show()
