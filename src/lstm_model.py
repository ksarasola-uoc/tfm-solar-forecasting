"""
LSTM forecaster para produccion solar.

Arquitectura v4: compacta con GaussianNoise, TemporalAttention y
cabeza encoder+decoder para future features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from models import SolarLoss, QuantileLoss


class GaussianNoise(nn.Module):
    """
    Anade ruido gaussiano a la entrada durante entrenamiento.

    En eval() es transparente (no anade ruido).

    Args:
        std: desviacion estandar del ruido. Recomendado: 0.01-0.05
             para datos ya normalizados.
    """
    def __init__(self, std: float = 0.02):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class TemporalAttention(nn.Module):
    """Dot-product attention sobre outputs del LSTM."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(lstm_out), dim=1)
        return (lstm_out * weights).sum(dim=1)


class LSTMForecaster(nn.Module):
    """
    LSTM compacto con attention y data augmentation por ruido.

    Args:
        n_features        : features totales de entrada (base + lags)
        n_future_features : features del timestep objetivo
        hidden_size       : recomendado 32 con <2k secuencias
        num_layers        : recomendado 2
        dropout           : recomendado 0.4 con <2k secuencias
        noise_std         : std del GaussianNoise (0 para desactivar)
    """

    def __init__(self,
                 n_features:        int,
                 n_future_features: int   = 0,
                 hidden_size:       int   = 32,
                 num_layers:        int   = 2,
                 dropout:           float = 0.4,
                 noise_std:         float = 0.02):

        super().__init__()
        self.n_features        = n_features
        self.n_future_features = n_future_features
        self.hidden_size       = hidden_size

        # Data augmentation por ruido — solo activo en train()
        self.noise = GaussianNoise(std=noise_std)

        # Encoder
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)

        # Decoder compacto
        decoder_in = hidden_size + n_future_features
        self.fc1   = nn.Linear(decoder_in, 16)
        self.fc2   = nn.Linear(16, 1)
        self.relu  = nn.ReLU()

    def forward(self,
                x_seq:    torch.Tensor,
                x_future: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ruido de entrada (data augmentation, solo en train)
        x_seq = self.noise(x_seq)

        lstm_out, _ = self.lstm(x_seq)
        context     = self.attention(lstm_out)
        context     = self.dropout(context)

        if self.n_future_features > 0 and x_future is not None:
            dec_in = torch.cat([context, x_future], dim=1)
        else:
            dec_in = context

        return self.fc2(self.relu(self.fc1(dec_in)))


class LSTMTrainer:
    """
    Trainer para LSTMForecaster con weight decay, scheduler y early stopping.
    """

    def __init__(self,
                 model:         LSTMForecaster,
                 device:        str   = 'cpu',
                 learning_rate: float = 0.001,
                 loss_alpha:    float = 0.7):

        self.model  = model.to(device)
        self.device = device

        self.criterion = SolarLoss(alpha=loss_alpha)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=15,
            min_lr=1e-6
        )
        self.train_losses = []
        self.val_losses   = []

    def _unpack(self, batch):
        if len(batch) == 3:
            X, Xf, y = batch
            return X.to(self.device), Xf.to(self.device), y.to(self.device)
        X, y = batch
        return X.to(self.device), None, y.to(self.device)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            X, Xf, y = self._unpack(batch)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X, Xf), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def evaluate(self, loader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                X, Xf, y = self._unpack(batch)
                total += self.criterion(self.model(X, Xf), y).item()
        return total / len(loader)

    def fit(self,
            train_loader,
            val_loader,
            epochs:       int  = 200,
            patience:     int  = 25,
            verbose:      bool = True):

        from tqdm import tqdm

        best_val   = float('inf')
        no_improve = 0

        it = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in it:
            tr = self.train_epoch(train_loader)
            va = self.evaluate(val_loader)
            self.train_losses.append(tr)
            self.val_losses.append(va)
            self.scheduler.step(va)

            if va < best_val:
                best_val   = va
                no_improve = 0
                self.best_state = self.model.state_dict()
            else:
                no_improve += 1

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                it.set_postfix(train=f'{tr:.4f}', val=f'{va:.4f}',
                               best=f'{best_val:.4f}', lr=f'{lr:.1e}')

            if no_improve >= patience:
                if verbose:
                    print(f"\nEarly stopping época {epoch+1}")
                break

        self.model.load_state_dict(self.best_state)
        if verbose:
            print(f"\n✓ Completado  |  best val_loss: {best_val:.6f}")

    def predict(self, loader,
                feature_cols: Optional[List[str]] = None) -> tuple:
        import numpy as np
        self.model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for batch in loader:
                X, Xf, y = self._unpack(batch)
                preds.extend(self.model(X, Xf).cpu().numpy().flatten())
                acts.extend(y.cpu().numpy().flatten())
        return np.array(acts), np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# LSTMQuantileForecaster — LSTM con cabeza multi-cuantil (Exp 3)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMQuantileForecaster(nn.Module):
    """
    LSTM con cabeza de regresión por cuantiles.

    Arquitectura idéntica a LSTMForecaster hasta la capa fc1.
    fc2 produce n_quantiles salidas con monotonicity garantizada
    via cumsum + softplus sobre los incrementos (q1, q2-q1, q3-q2, ...).

    Args:
        n_features        : features totales de entrada (base + lags)
        n_future_features : features del timestep objetivo (decoder)
        hidden_size       : recomendado 32
        num_layers        : recomendado 2
        dropout           : recomendado 0.4
        noise_std         : std del GaussianNoise (0 para desactivar)
        n_quantiles       : número de cuantiles de salida (por defecto 5)
    """

    def __init__(self,
                 n_features:        int,
                 n_future_features: int   = 0,
                 hidden_size:       int   = 32,
                 num_layers:        int   = 2,
                 dropout:           float = 0.4,
                 noise_std:         float = 0.02,
                 n_quantiles:       int   = 5):

        super().__init__()
        self.n_features        = n_features
        self.n_future_features = n_future_features
        self.hidden_size       = hidden_size
        self.n_quantiles       = n_quantiles

        self.noise = GaussianNoise(std=noise_std)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)

        decoder_in = hidden_size + n_future_features
        self.fc1   = nn.Linear(decoder_in, 16)
        self.fc2   = nn.Linear(16, n_quantiles)
        self.relu  = nn.ReLU()

    def forward(self,
                x_seq:    torch.Tensor,
                x_future: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns:
            Tensor de forma (batch, n_quantiles) con cuantiles ordenados
            de menor a mayor por construcción (monotonicity garantizada).

        Ancla en q50 (cuantil central):
            - q50 se predice libremente → no está sesgado por los extremos
            - Cuantiles superiores (q75, q95): q50 + incrementos positivos
            - Cuantiles inferiores (q05, q25): q50 - decrementos positivos
        """
        x_seq = self.noise(x_seq)

        lstm_out, _ = self.lstm(x_seq)
        context     = self.attention(lstm_out)
        context     = self.dropout(context)

        if self.n_future_features > 0 and x_future is not None:
            dec_in = torch.cat([context, x_future], dim=1)
        else:
            dec_in = context

        raw = self.fc2(self.relu(self.fc1(dec_in)))  # (batch, n_quantiles)

        mid = self.n_quantiles // 2   # índice del cuantil mediano (=2 para 5 cuantiles)

        # Cuantil mediano: predicción libre (ancla)
        q_mid = raw[:, mid:mid+1]                          # (batch, 1)

        # Cuantiles superiores: q_mid + incrementos positivos acumulados
        upper_inc = F.softplus(raw[:, mid+1:])             # (batch, n_upper)
        upper_q   = q_mid + torch.cumsum(upper_inc, dim=1) # [q75, q95]

        # Cuantiles inferiores: q_mid - decrementos positivos acumulados
        # raw[:, :mid] = [raw_q05, raw_q25] → decrementos hacia afuera desde q50
        lower_inc    = F.softplus(raw[:, :mid])                      # (batch, n_lower)
        lower_cumsum = torch.cumsum(torch.flip(lower_inc, dims=[1]), dim=1)
        lower_q      = q_mid - torch.flip(lower_cumsum, dims=[1])    # [q05, q25]

        return torch.cat([lower_q, q_mid, upper_q], dim=1)


class LSTMQuantileTrainer:
    """
    Trainer para LSTMQuantileForecaster con QuantileLoss,
    early stopping y scheduler. Paralelo a LSTMTrainer.

    predict() devuelve (y_true_scaled, y_preds_quantiles_scaled) en
    escala normalizada; la inversión se hace en el notebook con scaler_y.
    """

    def __init__(self,
                 model:         LSTMQuantileForecaster,
                 quantiles:     list  = None,
                 device:        str   = 'cpu',
                 learning_rate: float = 0.001):

        self.model     = model.to(device)
        self.device    = device
        self.quantiles = quantiles or [0.05, 0.25, 0.5, 0.75, 0.95]

        self.criterion = QuantileLoss(self.quantiles)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=15, min_lr=1e-6
        )
        self.train_losses = []
        self.val_losses   = []

    def _unpack(self, batch):
        if len(batch) == 3:
            X, Xf, y = batch
            return X.to(self.device), Xf.to(self.device), y.to(self.device)
        X, y = batch
        return X.to(self.device), None, y.to(self.device)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            X, Xf, y = self._unpack(batch)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X, Xf), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def evaluate(self, loader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                X, Xf, y = self._unpack(batch)
                total += self.criterion(self.model(X, Xf), y).item()
        return total / len(loader)

    def fit(self,
            train_loader,
            val_loader,
            epochs:   int  = 200,
            patience: int  = 25,
            verbose:  bool = True):

        from tqdm import tqdm

        best_val   = float('inf')
        no_improve = 0

        it = tqdm(range(epochs), desc="Training LSTM-Quantile") if verbose else range(epochs)

        for epoch in it:
            tr = self.train_epoch(train_loader)
            va = self.evaluate(val_loader)
            self.train_losses.append(tr)
            self.val_losses.append(va)
            self.scheduler.step(va)

            if va < best_val:
                best_val   = va
                no_improve = 0
                self.best_state = self.model.state_dict()
            else:
                no_improve += 1

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                it.set_postfix(train=f'{tr:.4f}', val=f'{va:.4f}',
                               best=f'{best_val:.4f}', lr=f'{lr:.1e}')

            if no_improve >= patience:
                if verbose:
                    print(f"\nEarly stopping época {epoch+1}")
                break

        self.model.load_state_dict(self.best_state)
        if verbose:
            print(f"\n✓ Completado  |  best val_loss: {best_val:.6f}")

    def predict(self, loader) -> tuple:
        """
        Devuelve (y_true, y_preds) en escala normalizada.

        Returns:
            y_true  : np.ndarray (N,)
            y_preds : np.ndarray (N, n_quantiles)
        """
        import numpy as np
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in loader:
                X, Xf, y = self._unpack(batch)
                q_preds = self.model(X, Xf).cpu().numpy()  # (batch, n_quantiles)
                all_preds.append(q_preds)
                all_true.extend(y.cpu().numpy().flatten())
        return np.array(all_true), np.vstack(all_preds)
