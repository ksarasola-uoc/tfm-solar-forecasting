"""
Utilidades compartidas para modelos de forecasting solar.

Contiene: SolarScaler, SolarLoss, NIGHT_VALUE, train_with_kfold,
plot_kfold_curves, predict_ensemble.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

NIGHT_VALUE = -1.0   # valor escalado fijo para horas nocturnas


# ─────────────────────────────────────────────────────────────────────────────
# SolarScaler — escalador robusto para produccion solar
# ─────────────────────────────────────────────────────────────────────────────

class SolarScaler:
    """
    Escalador robusto para series de produccion solar.

    Ajusta la normalizacion SOLO sobre muestras diurnas (produccion > 0),
    evitando que los ceros nocturnos contaminen la distribucion aprendida.

    Uso:
        scaler = SolarScaler()
        scaler.fit(y_train_array)
        y_scaled = scaler.transform(y)
        y_orig   = scaler.inverse_transform(y_scaled)
    """

    def __init__(self, night_threshold: float = 0.0):
        self.night_threshold = night_threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, y: np.ndarray) -> 'SolarScaler':
        """Calcula mean y std SOLO sobre valores diurnos (produccion > threshold)."""
        y = np.asarray(y, dtype=np.float32).flatten()
        diurno_mask = y > self.night_threshold
        y_diurno = y[diurno_mask]

        if len(y_diurno) == 0:
            raise ValueError(
                "No hay muestras diurnas (produccion > 0) en los datos de entrenamiento. "
                "Verifica que df_train contenga horas de produccion real."
            )

        self.mean_ = float(np.mean(y_diurno))
        self.std_  = float(np.std(y_diurno))

        if self.std_ < 1e-8:
            raise ValueError(
                f"Desviacion estandar diurna casi cero ({self.std_:.2e}). "
                "Revisa que los datos de produccion sean validos."
            )

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Escala: nocturnos -> NIGHT_VALUE, diurnos -> (y - mean) / std."""
        if self.mean_ is None:
            raise RuntimeError("SolarScaler no ha sido ajustado. Llama a fit() primero.")

        y = np.asarray(y, dtype=np.float32).flatten().copy()
        scaled = np.empty_like(y)

        diurno_mask = y > self.night_threshold
        scaled[~diurno_mask] = NIGHT_VALUE
        scaled[diurno_mask]  = (y[diurno_mask] - self.mean_) / self.std_

        return scaled

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Deshace la escala: nocturnos -> 0.0, resto -> y_scaled * std + mean."""
        if self.mean_ is None:
            raise RuntimeError("SolarScaler no ha sido ajustado. Llama a fit() primero.")

        y_scaled = np.asarray(y_scaled, dtype=np.float32).flatten().copy()
        original = np.empty_like(y_scaled)

        nocturno_mask = y_scaled <= (NIGHT_VALUE + 0.5)
        original[nocturno_mask]  = 0.0
        original[~nocturno_mask] = y_scaled[~nocturno_mask] * self.std_ + self.mean_

        original = np.clip(original, 0.0, None)

        return original


# ─────────────────────────────────────────────────────────────────────────────
# SolarLoss
# ─────────────────────────────────────────────────────────────────────────────

class SolarLoss(nn.Module):
    """alpha * MAE + (1-alpha) * MSE."""
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return self.alpha * self.mae(y_pred, y_true) + \
               (1 - self.alpha) * self.mse(y_pred, y_true)


class QuantileLoss(nn.Module):
    """
    Pinball loss (quantile regression loss) promediada sobre N cuantiles.

    Args:
        quantiles: lista de cuantiles en (0, 1). Por defecto 5 cuantiles estándar.

    Input:
        preds:  (batch, n_quantiles) — predicciones ordenadas por cuantil
        target: (batch,) o (batch, 1) — valor real

    Returns:
        Escalar: media de las pinball losses de cada cuantil.
    """
    def __init__(self, quantiles: list = None):
        super().__init__()
        self.quantiles = quantiles or [0.05, 0.25, 0.5, 0.75, 0.95]

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.squeeze()  # (batch,)
        total = torch.zeros(1, device=preds.device)
        for i, q in enumerate(self.quantiles):
            e = target - preds[:, i]
            total = total + torch.mean(torch.where(e >= 0, q * e, (q - 1) * e))
        return total / len(self.quantiles)


# ─────────────────────────────────────────────────────────────────────────────
# train_with_kfold — TimeSeriesSplit K-Fold
# ─────────────────────────────────────────────────────────────────────────────

def train_with_kfold(
        all_sequences_dataset,
        model_factory,
        trainer_factory,
        n_splits:    int  = 3,
        epochs:      int  = 200,
        patience:    int  = 25,
        batch_size:  int  = 64,
        device:      str  = 'cpu',
        verbose:     bool = True) -> Tuple[list, List[dict]]:
    """
    Entrena K modelos con TimeSeriesSplit.

    Args:
        all_sequences_dataset : SolarSequenceDataset (train+val, sin test)
        model_factory         : callable() -> modelo nuevo
        trainer_factory       : callable(model) -> trainer nuevo
        n_splits              : numero de folds (recomendado 3 con 1 anio)

    Returns:
        (models_list, fold_histories)
    """
    from torch.utils.data import Subset

    tss          = TimeSeriesSplit(n_splits=n_splits)
    indices      = np.arange(len(all_sequences_dataset))
    models_kfold = []
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(tss.split(indices)):
        if verbose:
            print(f"\n── Fold {fold+1}/{n_splits}  "
                  f"train: {len(train_idx)} seq  val: {len(val_idx)} seq ──")

        train_sub = Subset(all_sequences_dataset, train_idx)
        val_sub   = Subset(all_sequences_dataset, val_idx)

        tl = DataLoader(train_sub, batch_size=batch_size,
                        shuffle=True, drop_last=True)
        vl = DataLoader(val_sub, batch_size=batch_size,
                        shuffle=False, drop_last=False)

        model   = model_factory()
        trainer = trainer_factory(model)
        trainer.fit(tl, vl, epochs=epochs, patience=patience, verbose=verbose)

        models_kfold.append(trainer.model)

        fold_histories.append({
            'fold':          fold + 1,
            'n_train':       len(train_idx),
            'n_val':         len(val_idx),
            'train_losses':  trainer.train_losses,
            'val_losses':    trainer.val_losses,
            'best_val_loss': min(trainer.val_losses) if trainer.val_losses else float('inf'),
        })

        if verbose:
            best = fold_histories[-1]['best_val_loss']
            print(f"  → Fold {fold+1} best val_loss: {best:.5f}")

    return models_kfold, fold_histories


def plot_kfold_curves(fold_histories: List[dict],
                      planta_nombre: str,
                      save_path: str = None):
    """Visualiza las curvas de aprendizaje de cada fold del K-Fold."""
    n_folds = len(fold_histories)
    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), sharey=True)
    if n_folds == 1:
        axes = [axes]

    for ax, h in zip(axes, fold_histories):
        ax.plot(h['train_losses'], label='Train', linewidth=1.5, color='steelblue')
        ax.plot(h['val_losses'],   label='Val',   linewidth=1.5, color='coral')
        ax.axhline(h['best_val_loss'], color='coral', linestyle='--',
                   linewidth=1, alpha=0.6)

        if h['train_losses'] and h['val_losses']:
            gap = h['val_losses'][-1] - h['train_losses'][-1]
            color_gap = 'red' if gap > 0.05 else 'green'
            ax.set_title(
                f"Fold {h['fold']}  ({h['n_train']}→{h['n_val']} seq)\n"
                f"best_val={h['best_val_loss']:.4f}  gap={gap:+.4f}",
                fontsize=9,
                color=color_gap if gap > 0.05 else 'black'
            )
        ax.set_xlabel('Época', fontsize=8)
        if h['fold'] == 1:
            ax.set_ylabel('Loss', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{planta_nombre} — curvas K-Fold', fontweight='bold', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def predict_ensemble(models_kfold:   list,
                     fold_histories: List[dict],
                     test_dataset,
                     device:         str = 'cpu') -> pd.DataFrame:
    """
    Ensemble ponderado por val_loss de cada fold.

    Pesos inversamente proporcionales al best_val_loss:
      peso_i = 1 / best_val_loss_i

    Args:
        models_kfold   : modelos devueltos por train_with_kfold
        fold_histories : historiales devueltos por train_with_kfold
        test_dataset   : SolarSequenceDataset de test
        device         : dispositivo de inferencia

    Returns:
        DataFrame con timestamp, y_true, y_pred, planta_nombre
    """
    # Pesos iguales para todos los folds.
    # El peso inverso al val_loss favorece sistemáticamente al fold 1 de TimeSeriesSplit
    # (el que tiene menos datos de entrenamiento y valida sobre la distribución más fácil),
    # degradando la contribución del fold con más datos (el que mejor generaliza al test).
    weights = np.ones(len(fold_histories)) / len(fold_histories)

    if len(models_kfold) > 1:
        print(f"\n  Pesos del ensemble (pesos iguales):")
        for h, w in zip(fold_histories, weights):
            print(f"    Fold {h['fold']}: val_loss={h['best_val_loss']:.5f}  peso={w:.3f}")

    all_preds = []

    for model in models_kfold:
        model.eval()
        preds_this = []

        with torch.no_grad():
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                if len(item) == 3:
                    X_seq, X_fut, _ = item
                    X_seq = X_seq.unsqueeze(0).to(device)
                    X_fut = X_fut.unsqueeze(0).to(device)
                    p = model(X_seq, X_fut).cpu().item()
                else:
                    X_seq, _ = item
                    X_seq = X_seq.unsqueeze(0).to(device)
                    p = model(X_seq, None).cpu().item()
                preds_this.append(p)

        all_preds.append(preds_this)

    all_preds  = np.array(all_preds)
    preds_mean = np.average(all_preds, axis=0, weights=weights)

    actuals_scaled = [test_dataset[i][-1].item() for i in range(len(test_dataset))]
    timestamps     = [test_dataset.timestamps[i + test_dataset.seq_length]
                      for i in range(len(test_dataset))]

    scaler_y = test_dataset.scaler_y
    y_pred   = scaler_y.inverse_transform(np.array(preds_mean))
    y_true   = scaler_y.inverse_transform(np.array(actuals_scaled))

    return pd.DataFrame({
        'timestamp':     timestamps,
        'y_true':        y_true,
        'y_pred':        np.clip(y_pred, 0, None),
        'planta_nombre': test_dataset.planta_nombre
    })
