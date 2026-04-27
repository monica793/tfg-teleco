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
# Utilidades: cargar modelo desde checkpoint de Lightning
# ---------------------------------------------------------------------------

def _extraer_state_dict(ruta_checkpoint: str, map_location: str) -> dict:
    """Extrae el state_dict del modelo interno desde un checkpoint de Lightning."""
    ckpt = torch.load(ruta_checkpoint, map_location=map_location)
    return {
        k.removeprefix("modelo."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("modelo.")
    }


def cargar_modelo_desde_checkpoint(ruta_checkpoint: str, map_location: str = "cpu") -> ModeloCNN:
    """Carga ModeloCNN (IQ) desde un checkpoint de PyTorch Lightning."""
    modelo = ModeloCNN()
    modelo.load_state_dict(_extraer_state_dict(ruta_checkpoint, map_location))
    modelo.eval()
    return modelo


def cargar_checkpoint_automatico(ruta_checkpoint: str, map_location: str = "cpu") -> nn.Module:
    """
    Carga automáticamente ModeloCNN o ModeloCNNEnergia según los pesos del checkpoint.

    Detecta el tipo por la presencia de 'cabeza_densa' (IQ) o 'cabeza' (energia)
    en las claves del state_dict.

    Útil para no tener que especificar el tipo al evaluar.
    """
    from ml.modelo_energia import ModeloCNNEnergia
    state_dict = _extraer_state_dict(ruta_checkpoint, map_location)
    tiene_conv = any(k.startswith("bloques_conv") for k in state_dict)
    clase = ModeloCNN if tiene_conv else ModeloCNNEnergia
    modelo = clase()
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo
