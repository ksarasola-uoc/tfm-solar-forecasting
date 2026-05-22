"""
Métricas de evaluación para forecasting solar
"""

import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns


class ForecastEvaluator:
    """
    Evaluador de modelos de forecasting.
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         planta_nombre: str = '') -> Dict[str, float]:
        """
        Calcular métricas de error.
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones
            planta_nombre: Nombre de planta (para logging)
            
        Returns:
            Diccionario con métricas
        """
        # Filtrar valores nulos
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # MAPE (Mean Absolute Percentage Error)
        # Solo para valores > 0 (evitar división por cero)
        mask_nonzero = y_true > 0.01  # umbral pequeño
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
        else:
            mape = np.nan
        
        # R² (Coefficient of Determination)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        metrics = {
            'planta': planta_nombre,
            'n_samples': len(y_true),
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        return metrics
    
    @staticmethod
    def calculate_metrics_by_condition(df_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calcular métricas estratificadas por condiciones.
        
        Args:
            df_results: DataFrame con y_true, y_pred, y features
            
        Returns:
            DataFrame con métricas por condición
        """
        metrics_list = []
        
        # Por hora del día
        for hora in range(24):
            mask = df_results['hora'] == hora
            if mask.sum() > 0:
                df_sub = df_results[mask]
                m = ForecastEvaluator.calculate_metrics(
                    df_sub['y_true'].values,
                    df_sub['y_pred'].values,
                    planta_nombre=f"hora_{hora}"
                )
                m['condicion'] = 'hora'
                m['valor'] = hora
                metrics_list.append(m)
        
        # Por mes
        for mes in range(1, 13):
            mask = df_results['mes'] == mes
            if mask.sum() > 0:
                df_sub = df_results[mask]
                m = ForecastEvaluator.calculate_metrics(
                    df_sub['y_true'].values,
                    df_sub['y_pred'].values,
                    planta_nombre=f"mes_{mes}"
                )
                m['condicion'] = 'mes'
                m['valor'] = mes
                metrics_list.append(m)
        
        # Por nubosidad (cuartiles)
        if 'cloud_cover' in df_results.columns:
            df_results['nubosidad_cuartil'] = pd.qcut(
                df_results['cloud_cover'], 
                q=4, 
                labels=['Despejado', 'Poco nublado', 'Nublado', 'Muy nublado']
            )
            
            for cuartil in df_results['nubosidad_cuartil'].unique():
                mask = df_results['nubosidad_cuartil'] == cuartil
                if mask.sum() > 0:
                    df_sub = df_results[mask]
                    m = ForecastEvaluator.calculate_metrics(
                        df_sub['y_true'].values,
                        df_sub['y_pred'].values,
                        planta_nombre=str(cuartil)
                    )
                    m['condicion'] = 'nubosidad'
                    m['valor'] = str(cuartil)
                    metrics_list.append(m)
        
        return pd.DataFrame(metrics_list)
    
    @staticmethod
    def plot_predictions_vs_actual(df_results: pd.DataFrame, 
                                   planta_nombre: str,
                                   sample_days: int = 7,
                                   save_path: str = None):
        """
        Visualizar predicciones vs valores reales.
        
        Args:
            df_results: DataFrame con timestamp, y_true, y_pred
            planta_nombre: Nombre de la planta
            sample_days: Número de días a mostrar
            save_path: Ruta para guardar figura
        """
        # Tomar muestra de días
        df_sample = df_results.head(24 * sample_days)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(df_sample['timestamp'], df_sample['y_true'], 
               label='Real', linewidth=2, color='black', alpha=0.7)
        ax.plot(df_sample['timestamp'], df_sample['y_pred'], 
               label='Predicción', linewidth=2, color='red', alpha=0.7, linestyle='--')
        
        # Intervalos de confianza si existen
        if 'y_pred_lower' in df_sample.columns:
            ax.fill_between(
                df_sample['timestamp'],
                df_sample['y_pred_lower'],
                df_sample['y_pred_upper'],
                alpha=0.2, color='red', label='IC 95%'
            )
        
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Producción normalizada (MWh/MW)', fontsize=12)
        ax.set_title(f'Predicción vs Real - {planta_nombre}\n(Primeros {sample_days} días)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figura guardada: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_scatter(df_results: pd.DataFrame, planta_nombre: str,
                    save_path: str = None):
        """
        Scatter plot: predicho vs observado.
        
        Args:
            df_results: DataFrame con y_true, y_pred
            planta_nombre: Nombre de la planta
            save_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter
        ax.scatter(df_results['y_true'], df_results['y_pred'], 
                  alpha=0.3, s=10, color='blue')
        
        # Línea diagonal (predicción perfecta)
        max_val = max(df_results['y_true'].max(), df_results['y_pred'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Predicción perfecta')
        
        # Métricas en el gráfico
        metrics = ForecastEvaluator.calculate_metrics(
            df_results['y_true'].values,
            df_results['y_pred'].values
        )
        
        textstr = f"MAE = {metrics['MAE']:.3f}\nRMSE = {metrics['RMSE']:.3f}\nR² = {metrics['R2']:.3f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Producción real (MWh/MW)', fontsize=12)
        ax.set_ylabel('Producción predicha (MWh/MW)', fontsize=12)
        ax.set_title(f'Predicho vs Observado - {planta_nombre}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figura guardada: {save_path}")
        
        plt.show()
    
    # ── Métricas probabilísticas (Exp 3) ──────────────────────────────────────

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        """
        Pinball loss (quantile loss) para un cuantil q.

        Args:
            y_true : valores reales (N,)
            y_pred : predicciones del cuantil q (N,)
            q      : nivel del cuantil en (0, 1)

        Returns:
            Escalar float con la pinball loss media.
        """
        e = y_true - y_pred
        return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))

    @staticmethod
    def calculate_probabilistic_metrics(
            y_true: np.ndarray,
            q_preds: np.ndarray,
            quantiles: list) -> dict:
        """
        Calcula métricas de calibración probabilística.

        Args:
            y_true    : valores reales (N,)
            q_preds   : predicciones (N, n_quantiles) ordenadas por cuantil
            quantiles : lista de cuantiles, e.g. [0.05, 0.25, 0.5, 0.75, 0.95]

        Returns:
            Diccionario con:
              - pinball_{q_str}     para cada cuantil
              - CRPS                media de todas las pinball losses
              - coverage_50         fracción en [q25, q75]
              - coverage_90         fracción en [q05, q95]
              - interval_width_50_mean
              - interval_width_90_mean
        """
        metrics = {}
        pinball_values = []

        for i, q in enumerate(quantiles):
            q_str = f"{int(round(q * 100)):03d}"
            pb = ForecastEvaluator.pinball_loss(y_true, q_preds[:, i], q)
            metrics[f'pinball_{q_str}'] = pb
            pinball_values.append(pb)

        # CRPS aproximado como media de pinball losses
        metrics['CRPS'] = float(np.mean(pinball_values))

        # Coverage: fracción de observaciones dentro del intervalo
        q_map = {q: q_preds[:, i] for i, q in enumerate(quantiles)}

        if 0.25 in q_map and 0.75 in q_map:
            in_50 = (y_true >= q_map[0.25]) & (y_true <= q_map[0.75])
            metrics['coverage_50'] = float(np.mean(in_50))
            metrics['interval_width_50_mean'] = float(np.mean(q_map[0.75] - q_map[0.25]))
        else:
            metrics['coverage_50'] = np.nan
            metrics['interval_width_50_mean'] = np.nan

        if 0.05 in q_map and 0.95 in q_map:
            in_90 = (y_true >= q_map[0.05]) & (y_true <= q_map[0.95])
            metrics['coverage_90'] = float(np.mean(in_90))
            metrics['interval_width_90_mean'] = float(np.mean(q_map[0.95] - q_map[0.05]))
        else:
            metrics['coverage_90'] = np.nan
            metrics['interval_width_90_mean'] = np.nan

        return metrics

    @staticmethod
    def plot_calibration(y_true: np.ndarray,
                         q_preds: np.ndarray,
                         quantiles: list,
                         model_name: str = '',
                         save_path: str = None):
        """
        Diagrama de calibración: cuantil nominal vs fracción observada.

        Un modelo bien calibrado muestra puntos sobre la diagonal.
        La shaded area indica desviación del ideal.

        Args:
            y_true     : valores reales (N,)
            q_preds    : predicciones (N, n_quantiles)
            quantiles  : lista de cuantiles nominales
            model_name : nombre del modelo (para el título)
            save_path  : ruta para guardar la figura
        """
        observed_fractions = []
        for i, q in enumerate(quantiles):
            frac = float(np.mean(y_true <= q_preds[:, i]))
            observed_fractions.append(frac)

        fig, ax = plt.subplots(figsize=(6, 6))

        # Diagonal de calibración perfecta
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Calibración perfecta', alpha=0.6)

        # Banda de referencia ±5%
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                        alpha=0.1, color='green', label='±5% banda')

        ax.plot(quantiles, observed_fractions, 'o-',
                color='steelblue', linewidth=2, markersize=8,
                label=model_name or 'Modelo')

        for q, f in zip(quantiles, observed_fractions):
            ax.annotate(f'{f:.2f}', (q, f), textcoords='offset points',
                        xytext=(6, 4), fontsize=8)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Cuantil nominal', fontsize=12)
        ax.set_ylabel('Fracción observada', fontsize=12)
        ax.set_title(f'Diagrama de calibración\n{model_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Figura guardada: {save_path}")

        plt.show()

    @staticmethod
    def plot_error_distribution(df_results: pd.DataFrame, planta_nombre: str,
                                save_path: str = None):
        """
        Distribución de errores de predicción.
        
        Args:
            df_results: DataFrame con y_true, y_pred
            planta_nombre: Nombre de la planta
            save_path: Ruta para guardar
        """
        errors = df_results['y_pred'] - df_results['y_true']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma
        axes[0].hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
        axes[0].set_xlabel('Error de predicción (MWh/MW)', fontsize=11)
        axes[0].set_ylabel('Frecuencia', fontsize=11)
        axes[0].set_title('Distribución de errores', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (normalidad de errores)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{planta_nombre}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figura guardada: {save_path}")
        
        plt.show()