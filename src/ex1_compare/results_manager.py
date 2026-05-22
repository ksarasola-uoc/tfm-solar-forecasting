"""
results_manager.py — Gestión consistente de resultados para el Experimento 1.

Clase ResultsManager que centraliza:
- Creación de directorios de resultados
- Guardado/carga de predicciones y métricas
- Guardado/carga de checkpoints (formato por modelo)
"""

import pickle
from pathlib import Path
import pandas as pd


class ResultsManager:
    """
    Gestiona la estructura de resultados para un modelo del Experimento 1.

    Estructura creada:
        results/ex1_compare/
        ├── figures/{model_name}/
        ├── metrics/
        └── checkpoints/{model_name}/
    """

    def __init__(self, model_name, results_dir='../results/ex1_compare',
                 checkpoints_dir='../checkpoints/ex1_compare'):
        """
        Args:
            model_name: Nombre del modelo (prophet, lightgbm, lstm, nbeats, tft).
            results_dir: Directorio base de resultados.
            checkpoints_dir: Directorio base de checkpoints.
        """
        self.model_name = model_name.lower()
        self.results_dir = Path(results_dir)
        self.checkpoints_dir = Path(checkpoints_dir)

        # Crear estructura de directorios
        self.figures_dir = self.results_dir / 'figures' / self.model_name
        self.metrics_dir = self.results_dir / 'metrics'
        self.ckpt_dir = self.checkpoints_dir / self.model_name

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f'ResultsManager [{self.model_name}]')
        print(f'  Figuras:     {self.figures_dir}')
        print(f'  Metricas:    {self.metrics_dir}')
        print(f'  Checkpoints: {self.ckpt_dir}')

    def save_predictions(self, df_pred):
        """Guarda predicciones completas en CSV."""
        path = self.metrics_dir / f'{self.model_name}_predictions.csv'
        df_pred.to_csv(path, index=False)
        print(f'  Predicciones guardadas: {path}')
        return path

    def save_metrics(self, df_metrics):
        """Guarda resumen de métricas por planta en CSV."""
        path = self.metrics_dir / f'{self.model_name}_metrics_summary.csv'
        df_metrics.to_csv(path, index=False)
        print(f'  Metricas guardadas: {path}')
        return path

    def load_all_model_metrics(self):
        """
        Carga métricas de todos los modelos disponibles para comparación.

        Returns:
            dict: {model_name: DataFrame} con las métricas disponibles.
        """
        all_metrics = {}
        model_names = ['prophet', 'lightgbm', 'lstm', 'nbeats', 'tft']

        for m_name in model_names:
            path = self.metrics_dir / f'{m_name}_metrics_summary.csv'
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if 'planta' in df.columns and 'MAE' in df.columns:
                        all_metrics[m_name] = df
                except Exception:
                    pass

        print(f'  Modelos con metricas: {list(all_metrics.keys())}')
        return all_metrics

    def save_checkpoint(self, model_obj, planta, **kwargs):
        """
        Guarda checkpoint del modelo en el formato apropiado.

        - Prophet / LightGBM: pickle (.pkl)
        - LSTM / N-BEATS: torch state_dict (.pt)
        - TFT: torch checkpoint (.ckpt)

        Args:
            model_obj: Objeto del modelo o state_dict.
            planta: Nombre de la planta.
            **kwargs: Metadata adicional a incluir en el checkpoint.
        """
        if self.model_name in ('prophet', 'lightgbm'):
            path = self.ckpt_dir / f'{planta}.pkl'
            data = {'model': model_obj, 'planta': planta, **kwargs}
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        elif self.model_name in ('lstm', 'nbeats'):
            import torch
            path = self.ckpt_dir / f'{planta}.pt'
            data = {'planta': planta, **kwargs}
            if hasattr(model_obj, 'state_dict'):
                data['state_dict'] = model_obj.state_dict()
            else:
                data['state_dict'] = model_obj
            torch.save(data, path)

        elif self.model_name == 'tft':
            import torch
            path = self.ckpt_dir / f'{planta}.ckpt'
            data = {'planta': planta, **kwargs}
            if hasattr(model_obj, 'state_dict'):
                data['state_dict'] = model_obj.state_dict()
            else:
                data['state_dict'] = model_obj
            torch.save(data, path)

        else:
            path = self.ckpt_dir / f'{planta}.pkl'
            with open(path, 'wb') as f:
                pickle.dump({'model': model_obj, 'planta': planta, **kwargs}, f)

        print(f'  Checkpoint guardado: {path}')
        return path

    def load_checkpoint(self, planta):
        """
        Carga checkpoint de una planta.

        Returns:
            dict con los datos del checkpoint, o None si no existe.
        """
        extensions = {
            'prophet': '.pkl', 'lightgbm': '.pkl',
            'lstm': '.pt', 'nbeats': '.pt', 'tft': '.ckpt',
        }
        ext = extensions.get(self.model_name, '.pkl')
        path = self.ckpt_dir / f'{planta}{ext}'

        if not path.exists():
            print(f'  No existe checkpoint: {path}')
            return None

        if ext == '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            import torch
            data = torch.load(path, map_location='cpu', weights_only=False)

        print(f'  Checkpoint cargado: {path}')
        return data
