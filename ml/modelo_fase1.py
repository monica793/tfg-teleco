"""
Modelo unificado para la campaña experimental Fase 1 y siguientes.

Diseñado explícitamente para ser extensible sin reescribir código:

  Fase 1 — comparar representaciones con onset-centro binario
    ModeloFase1(in_channels=1)  → solo energía   (R1)
    ModeloFase1(in_channels=2)  → IQ              (R2)
    ModeloFase1(in_channels=3)  → IQ + energía    (R3)

  Fase 2 — cambiar definición del objetivo (ventana_llena)
    Mismo modelo, mismo in_channels; solo cambia etiquetado del dataset.

  Fase 3 — multiclase (C0/C1/C2)
    ModeloFase1(in_channels=..., num_clases=3)
    Loss: CrossEntropyLoss en entrenar_modelo.py.

Arquitectura (fija para comparabilidad entre representaciones):
  Bloque 1: Conv1d(in_channels → 8,  kernel=7, pad=3) + ReLU
  Bloque 2: Conv1d(8           → 16, kernel=5, pad=2) + ReLU
  Sin pooling — preserva resolución temporal completa.
  Flatten  : 16 × 128 = 2048
  FC1      : Linear(2048, 64) + ReLU + Dropout(p)
  Salida   : Linear(64, num_clases)   [logit(s) sin activación]

Notas de pérdida:
  num_clases=1 → BCEWithLogitsLoss   (binario)
  num_clases>1 → CrossEntropyLoss    (multiclase)
"""

import torch
import torch.nn as nn


REPRESENTACIONES_VALIDAS = ("energia", "iq", "iq_energia")
IN_CHANNELS = {"energia": 1, "iq": 2, "iq_energia": 3}
LONG_VENTANA = 128


class ModeloFase1(nn.Module):
    """
    Red CNN 1D parametrizable para detección de inicio de paquete.

    Parámetros
    ----------
    in_channels : int
        Canales de entrada: 1 (energía), 2 (IQ), 3 (IQ+energía).
    num_clases : int
        1 → salida escalar para BCEWithLogitsLoss (binario).
        N → salida N logits para CrossEntropyLoss (multiclase).
    dropout : float
        Tasa de dropout en la capa FC.
    """

    def __init__(self, in_channels: int = 2, num_clases: int = 1, dropout: float = 0.3):
        super().__init__()
        if in_channels not in (1, 2, 3):
            raise ValueError(f"in_channels debe ser 1, 2 o 3, no {in_channels}")
        self.in_channels = in_channels
        self.num_clases = num_clases

        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        dim_flat = 16 * LONG_VENTANA  # 2048

        self.cabeza = nn.Sequential(
            nn.Linear(dim_flat, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, in_channels, 128) — tensor de entrada (canales primero).

        Retorna
        -------
        (N, num_clases) — logit(s) sin activación.
        Para num_clases=1 la forma es (N, 1); aplica .squeeze(1) si necesitas (N,).
        """
        feats = self.backbone(x)          # (N, 16, 128)
        feats = feats.flatten(start_dim=1)  # (N, 2048)
        return self.cabeza(feats)           # (N, num_clases)


# ---------------------------------------------------------------------------
# Utilidades de carga
# ---------------------------------------------------------------------------

def _extraer_state_dict(ruta: str, map_location: str) -> dict:
    ckpt = torch.load(ruta, map_location=map_location, weights_only=False)
    return {
        k.removeprefix("modelo."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("modelo.")
    }


def cargar_modelo_fase1(ruta: str, map_location: str = "cpu") -> "ModeloFase1":
    """
    Carga un checkpoint de ModeloFase1 detectando in_channels y num_clases
    automáticamente desde las claves del state_dict.
    """
    sd = _extraer_state_dict(ruta, map_location)
    in_channels = int(sd["backbone.0.weight"].shape[1])
    num_clases = int(sd["cabeza.3.weight"].shape[0])
    modelo = ModeloFase1(in_channels=in_channels, num_clases=num_clases)
    modelo.load_state_dict(sd)
    modelo.eval()
    return modelo


def cargar_checkpoint(ruta: str, map_location: str = "cpu") -> nn.Module:
    """
    Punto de entrada único para cargar cualquier checkpoint (Fase 1 o histórico).

    Detecta automáticamente el tipo:
    - Claves con 'backbone.*'      → ModeloFase1
    - Claves con 'bloques_conv.*'  → modelo histórico IQ/v3
    - Sin conv                     → ModeloCNNEnergia histórico
    """
    sd = _extraer_state_dict(ruta, map_location)
    if any(k.startswith("backbone") for k in sd):
        return cargar_modelo_fase1(ruta, map_location)

    # Fallback a cargador histórico
    from ml.modelos_legacy import cargar_checkpoint_historico
    return cargar_checkpoint_historico(ruta, map_location)
