"""
Detector de inicio de paquete basado en perfil de energía.

Motivación:
  El detector IQ (ModeloCNN) produce una 'meseta' porque ventanas centradas
  en t±k contienen el mismo packet, solo desplazado dentro de la ventana.
  La red aprende 'hay packet aquí' en lugar de 'el packet empieza JUSTO aquí'.

  Solución: operar sobre la energía por segmento E[n] = I[n]² + Q[n]².
  Cuando el onset cae en el centro de la ventana (muestra 64):
    - Segmentos 0..N/2-1  → solo ruido  → energía baja
    - Segmentos N/2..N-1  → señal+ruido → energía alta

  Este patrón de salto es ÚNICO para onset en el centro. Si el onset se
  desplaza ±k muestras, el salto se desplaza proporcionalmente y el perfil
  ya no coincide con el patrón aprendido → la meseta desaparece.

Arquitectura:
  Preproceso : IQ (N,2,128) → energía media por segmento (N, N_SEGS)
  Normaliza  : divide por la media de energía de la ventana
  FC1        : Linear(N_SEGS, 32) + ReLU + Dropout
  FC2        : Linear(32, 1) → logit escalar

  Con N_SEGS=16 (segmentos de 8 muestras), la resolución temporal es 8 muestras,
  frente a las ~128 muestras de meseta del modelo IQ.
"""

import torch
import torch.nn as nn


class ModeloCNNEnergia(nn.Module):
    """
    Detector de onset mediante perfil de energía segmentada.

    Entrada : (N, 2, 128)  — misma interfaz que ModeloCNN (canales I,Q)
    Salida  : (N, 1)       — logit escalar (sin sigmoid)
    """

    CANALES_ENTRADA = 2
    LONG_VENTANA    = 128
    N_SEGS          = 16          # 16 segmentos × 8 muestras
    SEG_LEN         = LONG_VENTANA // N_SEGS   # 8

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.cabeza = nn.Sequential(
            nn.Linear(self.N_SEGS, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 2, 128) — canales I y Q

        Pipeline interno:
          1. Energía por muestra : E[n] = I[n]² + Q[n]²     → (N, 128)
          2. Segmentación        : media en bloques de SEG_LEN → (N, N_SEGS)
          3. Normalización       : divide por media global de la ventana
          4. FC                  : detecta el patrón de salto en el centro
        """
        # Energía instantánea (N, 128)
        energia = x[:, 0, :] ** 2 + x[:, 1, :] ** 2

        # Energía media por segmento (N, N_SEGS)
        segs = energia.view(-1, self.N_SEGS, self.SEG_LEN).mean(dim=2)

        # Normalización relativa: elimina dependencia de la potencia absoluta
        mu = segs.mean(dim=1, keepdim=True) + 1e-8
        segs = segs / mu

        return self.cabeza(segs)   # (N, 1)


# ---------------------------------------------------------------------------
# Utilidad: cargar modelo desde checkpoint de Lightning
# ---------------------------------------------------------------------------

def cargar_modelo_energia_desde_checkpoint(
    ruta_checkpoint: str,
    map_location: str = "cpu",
) -> ModeloCNNEnergia:
    """
    Carga los pesos de ModeloCNNEnergia desde un checkpoint de PyTorch Lightning.
    """
    ckpt = torch.load(ruta_checkpoint, map_location=map_location)
    state_dict_lightning = ckpt["state_dict"]

    state_dict_modelo = {
        k.removeprefix("modelo."): v
        for k, v in state_dict_lightning.items()
        if k.startswith("modelo.")
    }

    modelo = ModeloCNNEnergia()
    modelo.load_state_dict(state_dict_modelo)
    modelo.eval()
    return modelo
