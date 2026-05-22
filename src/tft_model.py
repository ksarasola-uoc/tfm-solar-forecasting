"""
Temporal Fusion Transformer forecaster para produccion solar.
Optimizado para CPU.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')


class TFTForecaster:
    """
    Wrapper para Temporal Fusion Transformer - OPTIMIZADO PARA CPU.
    """

    def __init__(self,
                 hidden_size: int = 32,
                 attention_head_size: int = 2,
                 dropout: float = 0.1,
                 hidden_continuous_size: int = 8,
                 learning_rate: float = 0.003):
        """
        Args:
            hidden_size: Tamano de hidden layers (reducido para CPU)
            attention_head_size: Numero de attention heads (reducido para CPU)
            dropout: Dropout rate
            hidden_continuous_size: Tamano para variables continuas
            learning_rate: Learning rate
        """
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate

        self.model = None
        self.trainer = None
        self.planta_nombre = None
        self.best_model_path = None

    def create_model(self, train_dataset):
        """
        Crear modelo TFT optimizado para CPU.
        """
        self.model = TemporalFusionTransformer.from_dataset(
            train_dataset,

            # Arquitectura (reducida para CPU)
            hidden_size=self.hidden_size,
            lstm_layers=1,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,

            # Loss - Solo usar quantile 0.5 (mediana) para acelerar
            loss=QuantileLoss(quantiles=[0.5]),

            # Optimizer
            learning_rate=self.learning_rate,
            reduce_on_plateau_patience=3,

            # Logging (desactivado para velocidad)
            log_interval=-1,
            log_val_interval=-1,
        )

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Modelo TFT: {n_params:,} parametros")

        return self.model

    def fit(self,
            train_dataset,
            val_dataset,
            planta_nombre: str,
            max_epochs: int = 20,
            patience: int = 5,
            batch_size: int = 128,
            verbose: bool = True):
        """
        Entrenar modelo TFT - VERSION CPU.
        """
        self.planta_nombre = planta_nombre

        print(f"\nEntrenando TFT para {planta_nombre}...")

        if self.model is None:
            self.create_model(train_dataset)

        # DataLoaders - num_workers=0 para CPU
        train_dataloader = train_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )

        val_dataloader = val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size * 2,
            num_workers=0,
            pin_memory=False
        )

        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=patience,
            verbose=False,
            mode='min'
        )

        # Checkpoint - guardar en directorio temporal
        import tempfile
        checkpoint_dir = tempfile.mkdtemp()

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=False,
            dirpath=checkpoint_dir,
            filename=f'{planta_nombre}'
        )

        # Trainer - CONFIGURACION CPU
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='cpu',
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_progress_bar=verbose,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=True,
            num_sanity_val_steps=0
        )

        # Entrenar
        try:
            self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
        except KeyboardInterrupt:
            print("  Entrenamiento interrumpido por usuario")

        # Guardar path del mejor modelo
        self.best_model_path = checkpoint_callback.best_model_path

        # Cargar mejor checkpoint
        if self.best_model_path and Path(self.best_model_path).exists():
            try:
                self.model = TemporalFusionTransformer.load_from_checkpoint(
                    self.best_model_path
                )
            except Exception as e:
                print(f"  ⚠️  No se pudo cargar checkpoint: {e}")
                print(f"  Usando ultimo modelo entrenado")

        print(f"✓ Entrenamiento completado")
        if checkpoint_callback.best_model_score is not None:
            print(f"  Mejor val_loss: {checkpoint_callback.best_model_score:.4f}")

    def predict(self, test_dataset, batch_size: int = 128):
        """
        Predicciones en CPU.
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        print(f"\nGenerando predicciones...")

        test_dataloader = test_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )

        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                x, y = batch

                output = self.model(x)

                # Extraer tensor segun tipo
                if hasattr(output, 'prediction'):
                    predictions = output.prediction
                elif isinstance(output, dict):
                    predictions = output.get('prediction', output.get('output', output))
                elif isinstance(output, torch.Tensor):
                    predictions = output
                elif isinstance(output, tuple):
                    predictions = output[0]
                else:
                    raise TypeError(f"Tipo no soportado: {type(output)}")

                if not isinstance(predictions, torch.Tensor):
                    raise TypeError(f"predictions debe ser tensor, es {type(predictions)}")

                # Procesar shape
                if len(predictions.shape) == 3:
                    mid_idx = predictions.shape[1] // 2
                    predictions = predictions[:, mid_idx, :]

                while len(predictions.shape) > 1 and predictions.shape[-1] == 1:
                    predictions = predictions.squeeze(-1)

                if len(predictions.shape) > 1:
                    predictions = predictions[:, 0]

                # Targets
                targets = y[0]
                if len(targets.shape) > 1:
                    targets = targets.squeeze()

                # A numpy
                pred_np = predictions.cpu().numpy()
                target_np = targets.cpu().numpy()

                if pred_np.ndim == 0:
                    pred_np = np.array([pred_np])
                if target_np.ndim == 0:
                    target_np = np.array([target_np])

                pred_np = pred_np.flatten()
                target_np = target_np.flatten()

                all_predictions.append(pred_np)
                all_targets.append(target_np)

        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_targets)

        print(f"  Predicciones procesadas: {len(y_pred)}")

        df_results = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'planta_nombre': self.planta_nombre,
            'idx': np.arange(len(y_pred))
        })

        df_results['timestamp'] = pd.date_range(
            start='2025-01-01',
            periods=len(df_results),
            freq='H'
        )

        df_results = df_results[['timestamp', 'y_true', 'y_pred', 'planta_nombre', 'idx']]
        df_results['y_pred'] = df_results['y_pred'].clip(lower=0)

        print(f"✓ {len(df_results):,} predicciones generadas")

        return df_results
