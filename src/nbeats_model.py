"""
nbeats_model.py — N-BEATS para forecasting de producción solar
==============================================================
Referencia: Oreshkin et al., ICLR 2020.
https://arxiv.org/abs/1905.10437

Por qué N-BEATS puede superar al LSTM con pocos datos
─────────────────────────────────────────────────────
El LSTM propaga gradientes a través de 48 pasos recurrentes.
Con ~1800 secuencias de entrenamiento, esa cadena introduce
ruido en los gradientes y limita la generalización.

N-BEATS recibe el vector completo de 48h de una sola vez y
lo procesa con MLPs totalmente conectados. Gradientes directos,
sin recurrencia, sin vanishing.

Más importante: sus stacks tienen inductive bias exacto para
la física de la producción solar:

  Stack TREND       — polinomios [1, t, t², t³]
                      → modela la rampa de amanecer/atardecer
                        y la campana gaussiana de la producción diurna

  Stack SEASONALITY — bases de Fourier sin(2πkt/T), cos(2πkt/T)
                      → captura el ciclo de 24h de producción,
                        la variación semanal y anual, sin que
                        tengamos que programarlo explícitamente

  Stack GENERIC     — MLP sin restricción de base
                      → aprende los residuos: variaciones por
                        nubosidad intermitente, curtailment,
                        patrones no modelados por los otros stacks

Mecanismo "double residual link":
  x_siguiente = x - backcast_i
  Cada stack solo modela lo que los anteriores no explicaron.
  → descomposición automática análoga a STL, pero aprendida
    de los datos en lugar de predefinida.

Compatibilidad con el código existente:
  NBeatsTrainer tiene exactamente la misma interfaz que LSTMTrainer:
    .fit(train_loader, val_loader, epochs, patience, verbose)
    .predict(loader)
    .train_losses  /  .val_losses
    .best_state
    .model

  Por tanto, train_with_kfold() y predict_ensemble() de models.py
  funcionan con N-BEATS sin ninguna modificación.

Diferencia en el Dataset:
  LSTM:   X_seq (batch, seq, n_features) — matriz de features
  N-BEATS: x_hist (batch, seq_length)   — solo el histórico del TARGET
           x_exog (batch, n_exog)        — features externas en t
  Esta separación es clave: las bases de trend/seasonality se ajustan
  al histórico del target, no a las features meteorológicas.
  Las features externas condicionan el MLP de cada bloque.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# SolarLoss — duplicada de models.py para que nbeats_model sea autocontenido
# ─────────────────────────────────────────────────────────────────────────────

class SolarLoss(nn.Module):
    """alpha * MAE + (1-alpha) * MSE. Idéntica a la de models.py."""
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mae(y_pred, y_true) + \
               (1.0 - self.alpha) * self.mse(y_pred, y_true)


# ─────────────────────────────────────────────────────────────────────────────
# NBeatsBlock — bloque atómico
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """
    Bloque básico de N-BEATS.

    MLP(x_hist ; x_exog) → θ → proyección sobre bases → (backcast, forecast)

    backcast : reconstrucción del pasado   (batch, seq_length)
    forecast : predicción h=1              (batch, 1)

    basis_type controla la proyección:
      'generic'     → matrices FC libres, aprendidas end-to-end
      'trend'       → polinomios Vandermonde, coeficientes θ aprendidos
      'seasonality' → Fourier sin/cos, coeficientes θ aprendidos

    En trend y seasonality las bases son buffers fijos (no entrenables).
    Solo los coeficientes θ se aprenden, lo que actúa como regularización.

    Args:
        seq_length  : longitud del lookback (48)
        n_units     : neuronas por capa del MLP
        n_layers    : capas del MLP
        basis_type  : 'generic' | 'trend' | 'seasonality'
        degree      : grado máximo del polinomio (solo trend)
        n_harmonics : número de armónicos de Fourier (solo seasonality)
        n_exog      : dimensión de features externas (0 = univariado puro)
        dropout     : dropout en el MLP
    """

    def __init__(self,
                 seq_length:  int,
                 n_units:     int   = 256,
                 n_layers:    int   = 4,
                 basis_type:  str   = 'generic',
                 degree:      int   = 3,
                 n_harmonics: int   = 8,
                 n_exog:      int   = 0,
                 dropout:     float = 0.1):

        super().__init__()
        self.seq_length = seq_length
        self.basis_type = basis_type
        self.n_exog     = n_exog

        # MLP compartido: recibe [x_hist; x_exog]
        mlp_in = seq_length + n_exog
        layers, d = [], mlp_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, n_units), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = n_units
        self.mlp = nn.Sequential(*layers)

        # Cabezas y bases
        if basis_type == 'generic':
            self.fc_back = nn.Linear(n_units, seq_length)
            self.fc_fore = nn.Linear(n_units, 1)

        elif basis_type == 'trend':
            p = degree + 1
            self.fc_back = nn.Linear(n_units, p)
            self.fc_fore = nn.Linear(n_units, p)
            # Bases polinomiales fijas (no entrenables)
            t_b = torch.linspace(-1.0, 1.0, seq_length)
            t_f = torch.tensor([1.0])
            # Matrices de Vandermonde: filas = potencias, columnas = timesteps
            B_b = torch.stack([t_b ** i for i in range(p)])  # (p, L)
            B_f = torch.stack([t_f ** i for i in range(p)])  # (p, 1)
            self.register_buffer('B_b', B_b)
            self.register_buffer('B_f', B_f)

        elif basis_type == 'seasonality':
            n_s = 2 * n_harmonics           # sin + cos por armónico
            self.fc_back = nn.Linear(n_units, n_s)
            self.fc_fore = nn.Linear(n_units, n_s)
            # Bases de Fourier fijas
            t_b = torch.linspace(0.0, 1.0, seq_length)
            t_f = torch.tensor([1.0])
            S_b = self._fourier(t_b, n_harmonics)  # (n_s, L)
            S_f = self._fourier(t_f, n_harmonics)  # (n_s, 1)
            self.register_buffer('B_b', S_b)
            self.register_buffer('B_f', S_f)

        else:
            raise ValueError(f"basis_type='{basis_type}' desconocido. "
                             "Usa 'generic', 'trend' o 'seasonality'.")

    @staticmethod
    def _fourier(t: torch.Tensor, n_harmonics: int) -> torch.Tensor:
        """Construye bases sin/cos. Shape: (2*n_harmonics, len(t))."""
        ks  = torch.arange(1, n_harmonics + 1, dtype=torch.float)
        arg = 2.0 * np.pi * ks.unsqueeze(1) * t.unsqueeze(0)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=0)

    def forward(self,
                x:    torch.Tensor,
                exog: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x    : (batch, seq_length)
            exog : (batch, n_exog) o None
        Returns:
            backcast : (batch, seq_length)
            forecast : (batch, 1)
        """
        inp = torch.cat([x, exog], dim=-1) if (exog is not None and self.n_exog > 0) else x
        h   = self.mlp(inp)

        if self.basis_type == 'generic':
            return self.fc_back(h), self.fc_fore(h)

        theta_b  = self.fc_back(h)
        theta_f  = self.fc_fore(h)
        backcast = theta_b @ self.B_b   # (batch, L)
        forecast  = theta_f @ self.B_f   # (batch, 1)
        return backcast, forecast


# ─────────────────────────────────────────────────────────────────────────────
# NBeatsStack — n bloques encadenados con double residual links
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsStack(nn.Module):
    """
    Stack de n_blocks bloques del mismo tipo.

    Double residual link entre bloques:
        x_in → bloque_i → (backcast_i, forecast_i)
        x_in  = x_in - backcast_i      ← residual al siguiente bloque
        total += forecast_i

    Al salir del stack, x_in es el residual no explicado → pasa al
    siguiente stack.
    """

    def __init__(self, n_blocks: int, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NBeatsBlock(**block_kwargs) for _ in range(n_blocks)]
        )

    def forward(self,
                x:    torch.Tensor,
                exog: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            residual       : (batch, seq_length)
            stack_forecast : (batch, 1)
        """
        stack_fc = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            bc, fc   = blk(x, exog)
            x        = x - bc
            stack_fc = stack_fc + fc
        return x, stack_fc


# ─────────────────────────────────────────────────────────────────────────────
# NBeatsForecaster — modelo completo
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsForecaster(nn.Module):
    """
    N-BEATS one-step-ahead (h=1) para producción solar.

    Tres stacks por defecto: trend → seasonality → generic
    Predicción = suma de forecasts de todos los stacks.

    Args:
        seq_length  : lookback en horas (48 recomendado)
        n_exog      : features externas en t (0 = univariado puro)
        stack_types : lista ordenada de tipos de stack
        n_blocks    : bloques por stack
        n_units     : neuronas por capa del MLP de cada bloque
        n_layers    : capas del MLP de cada bloque
        degree      : grado del polinomio de tendencia
        n_harmonics : armónicos de Fourier para estacionalidad
        dropout     : dropout en los MLPs
        noise_std   : ruido gaussiano en entrada (data augmentation, solo train)
    """

    def __init__(self,
                 seq_length:  int,
                 n_exog:      int        = 0,
                 stack_types: List[str]  = None,
                 n_blocks:    int        = 3,
                 n_units:     int        = 256,
                 n_layers:    int        = 4,
                 degree:      int        = 3,
                 n_harmonics: int        = 8,
                 dropout:     float      = 0.1,
                 noise_std:   float      = 0.02):

        super().__init__()
        self.seq_length = seq_length
        self.n_exog     = n_exog
        self.noise_std  = noise_std
        stack_types     = stack_types or ['trend', 'seasonality', 'generic']

        self.stacks = nn.ModuleList([
            NBeatsStack(
                n_blocks    = n_blocks,
                seq_length  = seq_length,
                n_units     = n_units,
                n_layers    = n_layers,
                basis_type  = st,
                degree      = degree,
                n_harmonics = n_harmonics,
                n_exog      = n_exog,
                dropout     = dropout,
            )
            for st in stack_types
        ])

    def forward(self,
                x:    torch.Tensor,
                exog: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x    : (batch, seq_length)  — histórico del target escalado
            exog : (batch, n_exog)      — features externas en t
        Returns:
            forecast : (batch, 1)
        """
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        total_fc = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for stack in self.stacks:
            x, fc    = stack(x, exog)
            total_fc = total_fc + fc
        return total_fc


# ─────────────────────────────────────────────────────────────────────────────
# NBeatsTrainer — INTERFAZ IDÉNTICA A LSTMTrainer
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsTrainer:
    """
    Trainer para NBeatsForecaster.

    Interfaz IDÉNTICA a LSTMTrainer para que train_with_kfold() y
    predict_ensemble() de models.py funcionen sin ninguna modificación.

    Atributos públicos (mismos que LSTMTrainer):
        .model         : NBeatsForecaster
        .train_losses  : List[float]
        .val_losses    : List[float]
        .best_state    : dict — state_dict del mejor checkpoint

    El DataLoader debe producir batches de NBeatsDataset:
        (x_hist, x_exog, y)   si n_exog > 0
        (x_hist, y)           si n_exog == 0
    donde x_hist tiene shape (batch, seq_length) — 1D por muestra.
    """

    def __init__(self,
                 model:         NBeatsForecaster,
                 device:        str   = 'cpu',
                 learning_rate: float = 5e-4,
                 loss_alpha:    float = 0.7):

        self.model     = model.to(device)
        self.device    = device
        self.criterion = SolarLoss(alpha=loss_alpha)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-6
        )
        self.train_losses = []
        self.val_losses   = []
        self.best_state   = None

    # ── Desempaquetado de batch: con o sin features externas ─────────────────

    def _unpack(self, batch):
        """
        Mismo patrón que LSTMTrainer._unpack.
        Acepta (xh, xe, y) con exog o (xh, y) sin exog.
        """
        if len(batch) == 3:
            xh, xe, y = batch
            return xh.to(self.device), xe.to(self.device), y.to(self.device)
        xh, y = batch
        return xh.to(self.device), None, y.to(self.device)

    # ── Epoch de entrenamiento ────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            xh, xe, y = self._unpack(batch)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(xh, xe), y)
            loss.backward()
            # Clipping conservador: N-BEATS tiene más params que LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            total += loss.item()
        return total / max(len(loader), 1)

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                xh, xe, y = self._unpack(batch)
                total += self.criterion(self.model(xh, xe), y).item()
        return total / max(len(loader), 1)

    # ── Fit con early stopping — MISMA FIRMA que LSTMTrainer.fit ─────────────

    def fit(self,
            train_loader: DataLoader,
            val_loader:   DataLoader,
            epochs:       int  = 200,
            patience:     int  = 25,
            verbose:      bool = True):

        best_val   = float('inf')
        no_improve = 0
        it = tqdm(range(epochs), desc='N-BEATS') if verbose else range(epochs)

        for epoch in it:
            tr = self.train_epoch(train_loader)
            va = self.evaluate(val_loader)
            self.train_losses.append(tr)
            self.val_losses.append(va)
            self.scheduler.step(va)

            if va < best_val:
                best_val        = va
                no_improve      = 0
                self.best_state = self.model.state_dict()
            else:
                no_improve += 1

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                it.set_postfix(train=f'{tr:.4f}', val=f'{va:.4f}',
                               best=f'{best_val:.4f}', lr=f'{lr:.1e}')

            if no_improve >= patience:
                if verbose:
                    print(f'\nEarly stopping época {epoch + 1}')
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if verbose:
            print(f'\n✓ Completado  |  best val_loss: {best_val:.6f}')

    # ── Predict — MISMA FIRMA que LSTMTrainer.predict ────────────────────────

    def predict(self, loader: DataLoader,
                feature_cols=None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (actuals_scaled, predictions_scaled)."""
        self.model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for batch in loader:
                xh, xe, y = self._unpack(batch)
                preds.extend(self.model(xh, xe).cpu().numpy().flatten())
                acts.extend(y.cpu().numpy().flatten())
        return np.array(acts), np.array(preds)
