"""
Entrenamiento del detector ciego (sin preámbulo) con PyTorch Lightning y WandB.

Uso básico (local, CPU):
  python -m ml.entrenar_modelo --datos data/dataset_aloha --epochs 50

Uso en Google Colab (GPU):
  python -m ml.entrenar_modelo --datos /content/dataset_aloha --epochs 50

El script espera que los archivos X_train.npy, Y_train.npy, X_val.npy y
Y_val.npy ya existan en el directorio `--datos`. Opcionalmente puede cargar
W_train.npy / W_val.npy para ponderar la pérdida (hard negatives cercanos).
Si no existen, se usan pesos unitarios.
Si no existen los datos, se puede
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
from ml.modelo_energia import ModeloCNNEnergia

MODELOS_DISPONIBLES = {
    "iq":      ModeloCNN,
    "energia": ModeloCNNEnergia,
}


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

        ruta_w_train = os.path.join(self.directorio_datos, "W_train.npy")
        ruta_w_val = os.path.join(self.directorio_datos, "W_val.npy")
        if os.path.exists(ruta_w_train) and os.path.exists(ruta_w_val):
            W_train = np.load(ruta_w_train).astype(np.float32)
            W_val = np.load(ruta_w_val).astype(np.float32)
            print("[DataModule] Pesos de pérdida detectados: W_train.npy / W_val.npy")
        else:
            W_train = np.ones_like(Y_train, dtype=np.float32)
            W_val = np.ones_like(Y_val, dtype=np.float32)
            print("[DataModule] Sin pesos W_*.npy: se usan pesos unitarios")

        # (N, 128, 2) → (N, 2, 128) para Conv1d
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))

        self.ds_train = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(Y_train).float(),
            torch.from_numpy(W_train).float(),
        )
        self.ds_val = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val).float(),
            torch.from_numpy(W_val).float(),
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

    Pérdida: BCEWithLogitsLoss con reducción none + promedio ponderado por muestra.
    Optimizador: Adam con lr inicial 1e-3.
    Scheduler: ReduceLROnPlateau sobre val_loss (factor 0.5, paciencia 5 épocas).
    """

    def __init__(self, lr: float = 1e-3, dropout: float = 0.3, tipo_modelo: str = "iq", pos_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        clase_modelo = MODELOS_DISPONIBLES.get(tipo_modelo, ModeloCNN)
        self.modelo = clase_modelo(dropout=dropout)
        # pos_weight registrado como buffer para moverse automáticamente a GPU/CPU
        self.register_buffer("pw", torch.tensor([float(pos_weight)]))
        self.criterio = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modelo(x)

    def _paso_comun(self, batch, etapa: str):
        x, y, w = batch
        logit = self(x).squeeze(1)       # (N,)
        # pos_weight amplifica el gradiente de los positivos
        pw_per_sample = torch.where(y > 0.5, self.pw.expand_as(y), torch.ones_like(y))
        loss_vec = self.criterio(logit, y)
        loss = (loss_vec * w * pw_per_sample).sum() / torch.clamp((w * pw_per_sample).sum(), min=1e-8)

        with torch.no_grad():
            pred = (torch.sigmoid(logit) >= 0.5).float()
            acc = (pred == y).float().mean()

        # Loguear por época (no por paso) para evitar saturar la salida en Colab.
        self.log(f"{etapa}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{etapa}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
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
    tipo_modelo: str = "iq",
    pos_weight: float = 1.0,
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

    modelo_lightning = DetectorALOHA(lr=lr, dropout=dropout, tipo_modelo=tipo_modelo, pos_weight=pos_weight)

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
        # Sin WandB: desactivar logger para reducir overhead y ruido en notebook.
        logger=loggers if loggers else False,
        # Menos frecuencia de actualización de progreso para no saturar el navegador.
        log_every_n_steps=200,
        enable_model_summary=False,
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
    parser.add_argument("--modelo", type=str, default="iq",
                        choices=list(MODELOS_DISPONIBLES.keys()),
                        help="Arquitectura: 'iq' (ModeloCNN original) o 'energia' (ModeloCNNEnergia)")
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Peso de los positivos en la pérdida (>1 fuerza más detecciones). "
                             "Recomendado: 10.0 para energia, 1.0 para iq con hard negatives.")
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
        tipo_modelo=args.modelo,
        pos_weight=args.pos_weight,
    )
