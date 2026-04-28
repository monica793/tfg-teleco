"""
Arquitecturas CNN 1D para detección de inicio de paquete en Pure ALOHA.

Todos los modelos reciben una ventana de 128 muestras y devuelven un logit
escalar (sin sigmoid). La sigmoid se aplica en inferencia para obtener un
score en [0, 1].

Modelos disponibles
-------------------
ModeloCNN       : 2 canales I/Q, MaxPool en primer bloque (campo receptivo amplio).
ModeloCNNLegacy : 2 canales I/Q, dos MaxPool (checkpoints históricos).
ModeloCNNv3     : 3 canales (I, Q, Energía normalizada), sin MaxPool.
                  Diseño: canales reducidos (8→16) + dataset 50/50 con hard negatives.
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


class ModeloCNNLegacy(nn.Module):
    """
    Arquitectura IQ histórica (dos MaxPool), usada por checkpoints antiguos.

    Flatten: 32 * 32 = 1024.
    """
    LONG_VENTANA = 128

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bloques_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # 128 -> 64
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),      # 64 -> 32
        )
        dim_conv = 32 * (self.LONG_VENTANA // 4)   # 1024
        self.cabeza_densa = nn.Sequential(
            nn.Linear(dim_conv, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.bloques_conv(x)          # (N, 32, 32)
        features = features.flatten(start_dim=1) # (N, 1024)
        return self.cabeza_densa(features)


class ModeloCNNv3(nn.Module):
    """
    Red convolucional 1D con tres canales de entrada: I, Q y Energía normalizada.

    Motivación
    ----------
    Combinar la representación IQ (que preserva fase) con el perfil de energía
    (que codifica cuándo hay señal) da a la red más información que cualquiera
    de las dos por separado.

    Sin MaxPool en ningún bloque: la resolución temporal se preserva al 100%,
    de modo que la red debe aprender a localizar el onset con precisión de 1 muestra.
    El tamaño reducido de los canales convolucionales (8→16) compensa el mayor flatten.

    Arquitectura
    ------------
      Bloque 1: Conv1d(3→8,  kernel=7, pad=3) + ReLU  → (N, 8,  128)
      Bloque 2: Conv1d(8→16, kernel=5, pad=2) + ReLU  → (N, 16, 128)
      Flatten  : (N, 16×128) = (N, 2048)
      FC1      : Linear(2048, 64) + ReLU + Dropout(0.3)
      FC2      : Linear(64, 1)   → logit escalar

    Dataset esperado
    ----------------
    X de forma (N, 3, 128): canal 0 = I, canal 1 = Q, canal 2 = E normalizada.
    E normalizada se calcula en la generación del dataset (no en el forward),
    lo que garantiza que el DataLoader sirva exactamente lo mismo en train y eval.
    """

    CANALES_ENTRADA = 3   # I, Q, Energía normalizada
    LONG_VENTANA = 128

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.bloques_conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        dim_conv = 16 * self.LONG_VENTANA   # 16 × 128 = 2048

        self.cabeza_densa = nn.Sequential(
            nn.Linear(dim_conv, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : (N, 3, 128) — canales I, Q, E_norm.

        Retorna
        -------
        logit : (N, 1) — logit escalar por ventana.
        """
        features = self.bloques_conv(x)           # (N, 16, 128)
        features = features.flatten(start_dim=1)  # (N, 2048)
        return self.cabeza_densa(features)         # (N, 1)


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
    state_dict = _extraer_state_dict(ruta_checkpoint, map_location)
    # Compatibilidad con checkpoints IQ legacy (FC de 1024 entradas).
    in_fc = int(state_dict["cabeza_densa.0.weight"].shape[1])
    clase = ModeloCNNLegacy if in_fc == 1024 else ModeloCNN
    modelo = clase()
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo


def cargar_checkpoint_automatico(ruta_checkpoint: str, map_location: str = "cpu") -> nn.Module:
    """
    Carga automáticamente la arquitectura correcta según los pesos del checkpoint.

    Lógica de detección
    -------------------
    1. Sin 'bloques_conv' → ModeloCNNEnergia (MLP sobre perfil de energía).
    2. 'bloques_conv.0.weight' con in_channels=3 → ModeloCNNv3 (IQ + Energía, sin pool).
    3. 'bloques_conv.0.weight' con in_channels=2 + FC de 1024 → ModeloCNNLegacy.
    4. 'bloques_conv.0.weight' con in_channels=2 + FC de 2048 → ModeloCNN.
    """
    from ml.modelo_energia import ModeloCNNEnergia
    state_dict = _extraer_state_dict(ruta_checkpoint, map_location)
    tiene_conv = any(k.startswith("bloques_conv") for k in state_dict)
    if not tiene_conv:
        clase = ModeloCNNEnergia
    else:
        in_channels = int(state_dict["bloques_conv.0.weight"].shape[1])
        if in_channels == 3:
            clase = ModeloCNNv3
        else:
            in_fc = int(state_dict["cabeza_densa.0.weight"].shape[1])
            clase = ModeloCNNLegacy if in_fc == 1024 else ModeloCNN
    modelo = clase()
    modelo.load_state_dict(state_dict)
    modelo.eval()
    return modelo
