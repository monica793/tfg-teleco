"""
Entrenamiento de ModeloFase1 con PyTorch Lightning.

Uso básico:
  # Generar dataset primero:
  python -m ml.generar_dataset --representacion iq --salida data/fase1_iq_onset_centro

  # Entrenar:
  python -m ml.entrenar_modelo \\
      --datos data/fase1_iq_onset_centro \\
      --representacion iq \\
      --sin_wandb

El nombre del checkpoint se genera automáticamente con la configuración del experimento:
  <representacion>_<modo_label>-epoch=XX-val_loss=X.XXXX.ckpt
  Ejemplo: iq_onset_centro-epoch=12-val_loss=0.4231.ckpt

Extensibilidad futura:
  --num_clases 3  →  CrossEntropyLoss para multiclase (Fase 3)
  --modo_label ventana_llena  →  Fase 2 (solo cambia qué nombre tiene el dataset)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from ml.modelo_fase1 import ModeloFase1, IN_CHANNELS


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ALOHADataModule(L.LightningDataModule):
    """
    Carga X_train/Y_train/W_train y X_val/Y_val/W_val desde disco.

    Transpone (N, L, C) → (N, C, L) para Conv1d.
    Compatible con cualquier número de canales (1, 2 o 3).
    """

    def __init__(self, directorio_datos: str, batch_size: int = 512, num_workers: int = 0):
        super().__init__()
        self.directorio_datos = directorio_datos
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _cargar(self, split: str):
        X = np.load(os.path.join(self.directorio_datos, f"X_{split}.npy"))
        Y = np.load(os.path.join(self.directorio_datos, f"Y_{split}.npy"))
        ruta_w = os.path.join(self.directorio_datos, f"W_{split}.npy")
        W = np.load(ruta_w).astype(np.float32) if os.path.exists(ruta_w) \
            else np.ones(len(Y), dtype=np.float32)
        X = np.transpose(X, (0, 2, 1))  # (N, L, C) → (N, C, L)
        return (torch.from_numpy(X),
                torch.from_numpy(Y).float(),
                torch.from_numpy(W).float())

    def setup(self, stage=None):
        self.ds_train = TensorDataset(*self._cargar("train"))
        self.ds_val   = TensorDataset(*self._cargar("val"))

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class DetectorLightning(L.LightningModule):
    """
    Módulo de entrenamiento genérico para ModeloFase1.

    Pérdida:
      num_clases=1 → BCEWithLogitsLoss con pesos por muestra
      num_clases>1 → CrossEntropyLoss con pesos por muestra (Fase 3)
    """

    def __init__(self, in_channels: int = 2, num_clases: int = 1,
                 lr: float = 1e-3, dropout: float = 0.3):
        super().__init__()
        self.save_hyperparameters()
        self.modelo = ModeloFase1(in_channels=in_channels, num_clases=num_clases,
                                   dropout=dropout)
        self.binario = (num_clases == 1)
        self.criterio = (nn.BCEWithLogitsLoss(reduction="none") if self.binario
                         else nn.CrossEntropyLoss(reduction="none"))

    def forward(self, x):
        return self.modelo(x)

    def _paso_comun(self, batch, etapa: str):
        x, y, w = batch
        out = self(x)  # (N, num_clases)

        if self.binario:
            logit = out.squeeze(1)          # (N,)
            loss_vec = self.criterio(logit, y)
        else:
            # CrossEntropyLoss espera (N, C) logits y (N,) targets long
            loss_vec = self.criterio(out, y.long())

        loss = (loss_vec * w).sum() / torch.clamp(w.sum(), min=1e-8)

        with torch.no_grad():
            if self.binario:
                pred = (torch.sigmoid(out.squeeze(1)) >= 0.5).float()
            else:
                pred = out.argmax(dim=1).float()
            acc = (pred == y).float().mean()

        self.log(f"{etapa}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{etapa}_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._paso_comun(batch, "train")

    def validation_step(self, batch, _):
        self._paso_comun(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                          factor=0.5, patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def entrenar(
    directorio_datos: str,
    representacion: str = "iq",
    modo_label: str = "onset_centro",
    num_clases: int = 1,
    directorio_ckpt: str = "checkpoints",
    max_epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1e-3,
    dropout: float = 0.3,
    usar_wandb: bool = True,
    proyecto_wandb: str = "tfg-aloha-detector",
    num_workers: int = 0,
) -> str:
    """
    Entrena ModeloFase1 y guarda el mejor checkpoint.

    El nombre del checkpoint incluye la configuración del experimento para
    facilitar su identificación sin abrir el fichero:
      <representacion>_<modo_label>-epoch=XX-val_loss=X.XXXX.ckpt
    """
    in_channels = IN_CHANNELS[representacion]
    nombre_ckpt = f"{representacion}_{modo_label}"

    datamodule = ALOHADataModule(directorio_datos, batch_size=batch_size,
                                  num_workers=num_workers)
    modelo = DetectorLightning(in_channels=in_channels, num_clases=num_clases,
                                lr=lr, dropout=dropout)

    callbacks = [
        ModelCheckpoint(
            dirpath=directorio_ckpt,
            filename=f"{nombre_ckpt}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True),
    ]

    loggers = [WandbLogger(project=proyecto_wandb, log_model=False)] if usar_wandb else []

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=loggers if loggers else False,
        log_every_n_steps=200,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(modelo, datamodule=datamodule)
    mejor = trainer.checkpoint_callback.best_model_path
    print(f"\nMejor checkpoint: {mejor}")
    return mejor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena ModeloFase1.")
    parser.add_argument("--datos",           type=str,   required=True,
                        help="Directorio con los .npy del dataset.")
    parser.add_argument("--representacion",  type=str,   default="iq",
                        choices=["energia", "iq", "iq_energia"],
                        help="Representación de entrada (debe coincidir con la del dataset).")
    parser.add_argument("--modo_label",      type=str,   default="onset_centro",
                        choices=["onset_centro", "ventana_llena"],
                        help="Modo de etiquetado (informativo para el nombre del checkpoint).")
    parser.add_argument("--num_clases",      type=int,   default=1,
                        help="1=binario (Fase 1/2), 3=multiclase (Fase 3).")
    parser.add_argument("--ckpt",            type=str,   default="checkpoints")
    parser.add_argument("--epochs",          type=int,   default=80)
    parser.add_argument("--batch",           type=int,   default=512)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--sin_wandb",       action="store_true")
    parser.add_argument("--workers",         type=int,   default=0)
    args = parser.parse_args()

    entrenar(
        directorio_datos=args.datos,
        representacion=args.representacion,
        modo_label=args.modo_label,
        num_clases=args.num_clases,
        directorio_ckpt=args.ckpt,
        max_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        dropout=args.dropout,
        usar_wandb=not args.sin_wandb,
        num_workers=args.workers,
    )
