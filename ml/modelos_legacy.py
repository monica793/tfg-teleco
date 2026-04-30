"""
Modelos históricos — versiones anteriores a la campaña Fase 1.

Estos modelos se conservan exclusivamente como referencia para la memoria del TFG
y para poder cargar checkpoints antiguos. No deben usarse para nuevos experimentos.

Para la campaña Fase 1 en adelante, usar ml/modelo_fase1.py.

Historial
---------
ModeloCNNLegacy : IQ, dos MaxPool (checkpoints epoch≈17, FC=1024)
ModeloCNN       : IQ, un MaxPool solo en primer bloque (FC=2048)
ModeloCNNv3     : IQ+Energía, sin MaxPool (checkpoints v3_5050_HN8)
ModeloCNNEnergia: MLP sobre perfil de energía segmentada (checkpoints epoch≈61)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Modelos históricos IQ
# ---------------------------------------------------------------------------

class ModeloCNNLegacy(nn.Module):
    """IQ con dos MaxPool. Checkpoints: mejor-epoch=17-val_loss=0.0210.ckpt"""
    LONG_VENTANA = 128

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bloques_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.cabeza_densa = nn.Sequential(
            nn.Linear(32 * 32, 64), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.cabeza_densa(self.bloques_conv(x).flatten(1))


class ModeloCNN(nn.Module):
    """IQ con MaxPool solo en primer bloque. Checkpoints: mejor-epoch=48-*.ckpt"""
    LONG_VENTANA = 128

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bloques_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.cabeza_densa = nn.Sequential(
            nn.Linear(32 * 64, 64), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.cabeza_densa(self.bloques_conv(x).flatten(1))


class ModeloCNNv3(nn.Module):
    """IQ+Energía, sin MaxPool. Checkpoints: v3_5050_HN8-epoch=11-*.ckpt"""
    CANALES_ENTRADA = 3
    LONG_VENTANA = 128

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bloques_conv = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.cabeza_densa = nn.Sequential(
            nn.Linear(16 * 128, 64), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.cabeza_densa(self.bloques_conv(x).flatten(1))


# ---------------------------------------------------------------------------
# Modelo histórico de energía segmentada (MLP)
# ---------------------------------------------------------------------------

class ModeloCNNEnergia(nn.Module):
    """MLP sobre perfil de energía segmentada. Checkpoints: mejor-epoch=61-*.ckpt"""
    N_SEGS = 16
    SEG_LEN = 128 // 16

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.cabeza = nn.Sequential(
            nn.Linear(self.N_SEGS, 128), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        e = x[:, 0, :] ** 2 + x[:, 1, :] ** 2
        segs = e.view(-1, self.N_SEGS, self.SEG_LEN).mean(dim=2)
        segs = segs / (segs.mean(dim=1, keepdim=True) + 1e-8)
        return self.cabeza(segs)


# ---------------------------------------------------------------------------
# Cargador automático de checkpoints históricos
# ---------------------------------------------------------------------------

def _extraer_state_dict(ruta: str, map_location: str) -> dict:
    ckpt = torch.load(ruta, map_location=map_location)
    return {k.removeprefix("modelo."): v for k, v in ckpt["state_dict"].items()
            if k.startswith("modelo.")}


def cargar_checkpoint_historico(ruta: str, map_location: str = "cpu") -> nn.Module:
    """
    Carga automáticamente el modelo histórico correcto según las claves del checkpoint.

    Lógica:
    - Sin 'bloques_conv'          → ModeloCNNEnergia
    - in_channels=3               → ModeloCNNv3
    - in_channels=2, FC=1024      → ModeloCNNLegacy
    - in_channels=2, FC=2048      → ModeloCNN
    """
    sd = _extraer_state_dict(ruta, map_location)
    tiene_conv = any(k.startswith("bloques_conv") for k in sd)
    if not tiene_conv:
        clase = ModeloCNNEnergia
    else:
        in_ch = int(sd["bloques_conv.0.weight"].shape[1])
        if in_ch == 3:
            clase = ModeloCNNv3
        else:
            in_fc = int(sd["cabeza_densa.0.weight"].shape[1])
            clase = ModeloCNNLegacy if in_fc == 1024 else ModeloCNN
    modelo = clase()
    modelo.load_state_dict(sd)
    modelo.eval()
    return modelo
