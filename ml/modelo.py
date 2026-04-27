"""
Arquitectura CNN 1D para detección ciega de inicio de paquete en Pure ALOHA.

Entrada : tensor (N, 2, 128)  — 2 canales (I, Q), 128 muestras de tiempo.
Salida  : tensor (N, 1)       — logit escalar (sin sigmoid/softmax).

La salida es un logit crudo porque se usa BCEWithLogitsLoss durante el
entrenamiento, que incluye internamente la sigmoid por razones de estabilidad
numérica. En inferencia se aplica sigmoid para obtener un score en [0, 1]
equivalente a la probabilidad de inicio de paquete.

Arquitectura:
  Bloque 1: Conv1d(2→16,  kernel=7, pad=3) + ReLU + MaxPool1d(2) → (N, 16, 64)
  Bloque 2: Conv1d(16→32, kernel=5, pad=2) + ReLU               → (N, 32, 64)
  Flatten  : (N, 32*64) = (N, 2048)
  FC1      : Linear(2048, 64) + ReLU + Dropout(0.3)
  FC2      : Linear(64, 1)   → logit escalar

  El pooling se mantiene solo en el primer bloque para ampliar el campo
  receptivo y estabilizar el aprendizaje, pero se elimina en el segundo
  bloque para preservar resolución temporal fina en las capas finales.
"""

import torch
import torch.nn as nn


class ModeloCNN(nn.Module):
    """
    Red convolucional 1D mínima para clasificación binaria de ventanas de señal.

    Atributos
    ----------
    bloques_conv : nn.Sequential
        Dos bloques convolucionales con ReLU. Solo el primer bloque tiene
        MaxPool para ampliar campo receptivo; el segundo opera a resolución
        completa para preservar precisión temporal fina.
    cabeza_densa : nn.Sequential
        Dos capas lineales con ReLU, Dropout y salida a logit escalar.
    """

    CANALES_ENTRADA = 2       # I y Q
    LONG_VENTANA = 128        # muestras de entrada

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.bloques_conv = nn.Sequential(
            # Bloque 1: captura patrones de escala corta + pooling para campo receptivo
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # 128 → 64

            # Bloque 2: sin pooling para preservar resolución temporal fina
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Tamaño de la representación aplanada: 32 canales × 64 posiciones
        dim_conv = 32 * (self.LONG_VENTANA // 2)   # = 32 * 64 = 2048

        self.cabeza_densa = nn.Sequential(
            nn.Linear(dim_conv, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),   # logit escalar; sin sigmoid aquí
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : (N, 2, 128) — señal I/Q en formato PyTorch (canales primero).

        Retorna
        -------
        logit : (N, 1) — logit escalar por ventana.
        """
        features = self.bloques_conv(x)          # (N, 32, 64)
        features = features.flatten(start_dim=1) # (N, 2048)
        return self.cabeza_densa(features)        # (N, 1)


# ---------------------------------------------------------------------------
# Utilidad: cargar modelo desde checkpoint de Lightning
# ---------------------------------------------------------------------------

def cargar_modelo_desde_checkpoint(ruta_checkpoint: str, map_location: str = "cpu") -> ModeloCNN:
    """
    Carga los pesos del ModeloCNN desde un checkpoint de PyTorch Lightning.

    El checkpoint de Lightning guarda el estado del LightningModule; los pesos
    del modelo interno están bajo la clave 'state_dict' con prefijo 'modelo.'.

    Parámetros
    ----------
    ruta_checkpoint : ruta al archivo .ckpt
    map_location    : dispositivo de destino ('cpu', 'cuda', etc.)

    Retorna
    -------
    ModeloCNN con pesos cargados en modo evaluación.
    """
    ckpt = torch.load(ruta_checkpoint, map_location=map_location)
    state_dict_lightning = ckpt["state_dict"]

    # Eliminar el prefijo 'modelo.' añadido por LightningModule
    state_dict_modelo = {
        k.removeprefix("modelo."): v
        for k, v in state_dict_lightning.items()
        if k.startswith("modelo.")
    }

    modelo = ModeloCNN()
    modelo.load_state_dict(state_dict_modelo)
    modelo.eval()
    return modelo
