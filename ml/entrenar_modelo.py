"""
Entrenamiento del detector ciego (sin preámbulo) con PyTorch Lightning y WandB.

Uso básico (local, CPU):
  python -m ml.entrenar_modelo --datos data/dataset_aloha --epochs 50

Uso en Google Colab (GPU):
  python -m ml.entrenar_modelo --datos /content/dataset_aloha --epochs 50

El script espera que los archivos X_train.npy, Y_train.npy, X_val.npy y
Y_val.npy ya existan en el directorio `--datos`. Si no existen, se puede
generar el dataset primero con:
  python -m ml.generar_dataset --salida data/dataset_aloha

Estructura Lightning:
  ALOHADataModule  : carga y sirve los datos en DataLoader.
  DetectorALOHA    : envuelve ModeloCNN con paso de entrenamiento/validación.
  Trainer          : configura callbacks (ModelCheckpoint, EarlyStopping) y
                     el logger WandbLogger.
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

from ml.modelo import ModeloCNN


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ALOHADataModule(L.LightningDataModule):
    """
    Carga X_train.npy / Y_train.npy / X_val.npy / Y_val.npy desde disco y
    los expone como DataLoaders de PyTorch.

    Los arrays en disco tienen formato (N, 128, 2) [muestra, I/Q].
    PyTorch necesita (N, 2, 128) [canal, muestra], por lo que se transpone al cargar.
    """

    def __init__(self, directorio_datos: str, batch_size: int = 512, num_workers: int = 0):
        super().__init__()
        self.directorio_datos = directorio_datos
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        X_train = np.load(os.path.join(self.directorio_datos, "X_train.npy"))
        Y_train = np.load(os.path.join(self.directorio_datos, "Y_train.npy"))
        X_val = np.load(os.path.join(self.directorio_datos, "X_val.npy"))
        Y_val = np.load(os.path.join(self.directorio_datos, "Y_val.npy"))

        # (N, 128, 2) → (N, 2, 128) para Conv1d
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))

        self.ds_train = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(Y_train).float(),
        )
        self.ds_val = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val).float(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class DetectorALOHA(L.LightningModule):
    """
    Módulo de entrenamiento para el detector CNN 1D de inicio de paquete.

    Pérdida: BCEWithLogitsLoss (incluye sigmoid internamente, numéricamente estable).
    Optimizador: Adam con lr inicial 1e-3.
    Scheduler: ReduceLROnPlateau sobre val_loss (factor 0.5, paciencia 5 épocas).
    """

    def __init__(self, lr: float = 1e-3, dropout: float = 0.3):
        super().__init__()
        self.save_hyperparameters()
        self.modelo = ModeloCNN(dropout=dropout)
        self.criterio = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modelo(x)

    def _paso_comun(self, batch, etapa: str):
        x, y = batch
        logit = self(x).squeeze(1)       # (N,)
        loss = self.criterio(logit, y)

        with torch.no_grad():
            pred = (torch.sigmoid(logit) >= 0.5).float()
            acc = (pred == y).float().mean()

        self.log(f"{etapa}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{etapa}_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._paso_comun(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._paso_comun(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# Función principal de entrenamiento
# ---------------------------------------------------------------------------

def entrenar(
    directorio_datos: str,
    directorio_ckpt: str = "checkpoints",
    max_epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    dropout: float = 0.3,
    usar_wandb: bool = True,
    proyecto_wandb: str = "tfg-aloha-detector",
    num_workers: int = 0,
):
    """
    Lanza el entrenamiento completo con PyTorch Lightning.

    Parámetros
    ----------
    directorio_datos  : carpeta con los archivos .npy del dataset.
    directorio_ckpt   : carpeta donde se guarda el mejor checkpoint.
    max_epochs        : número máximo de épocas.
    batch_size        : tamaño del batch.
    lr                : tasa de aprendizaje inicial de Adam.
    dropout           : tasa de Dropout en la cabeza densa.
    usar_wandb        : si True, activa el logger de Weights & Biases.
    proyecto_wandb    : nombre del proyecto en WandB.
    num_workers       : workers para DataLoader (0 = sin multiprocessing).
    """
    datamodule = ALOHADataModule(
        directorio_datos=directorio_datos,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    modelo_lightning = DetectorALOHA(lr=lr, dropout=dropout)

    callbacks = [
        ModelCheckpoint(
            dirpath=directorio_ckpt,
            filename="mejor-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
    ]

    loggers = []
    if usar_wandb:
        loggers.append(WandbLogger(project=proyecto_wandb, log_model=False))

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        log_every_n_steps=10,
        accelerator="auto",   # CPU local; GPU en Colab automáticamente
        devices=1,
    )

    trainer.fit(modelo_lightning, datamodule=datamodule)

    mejor_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"\nEntrenamiento finalizado. Mejor checkpoint: {mejor_ckpt}")
    return mejor_ckpt


# ---------------------------------------------------------------------------
# Ejecución desde línea de comandos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el detector CNN 1D con PyTorch Lightning.")
    parser.add_argument("--datos", type=str, default="data/dataset_aloha",
                        help="Directorio con X_train.npy, Y_train.npy, X_val.npy, Y_val.npy")
    parser.add_argument("--ckpt", type=str, default="checkpoints",
                        help="Directorio de salida para el mejor checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--sin_wandb", action="store_true",
                        help="Desactiva WandB (útil para pruebas locales rápidas)")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    entrenar(
        directorio_datos=args.datos,
        directorio_ckpt=args.ckpt,
        max_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        dropout=args.dropout,
        usar_wandb=not args.sin_wandb,
        num_workers=args.workers,
    )
