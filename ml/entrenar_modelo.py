"""
Entrenamiento de ModeloFase1 con PyTorch Lightning.

Uso básico:
  # Generar dataset primero:
  python -m ml.generar_dataset --representacion iq --modo_label multiclase_onset \\
         --salida data/v2/iq_multiclase_onset

  # Entrenar:
  python -m ml.entrenar_modelo \\
        --datos data/v2/iq_multiclase_onset \\
        --representacion iq \\
        --modo_label multiclase_onset \\
        --num_clases 3

Nombre del checkpoint (dentro de checkpoints_v2/<rep>_<modo>/):
  mejor.ckpt           ← checkpoint con menor val_loss (usado en evaluación)
  mejor_epoch_XX.ckpt  ← copia con información del epoch para la memoria
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from ml.modelo_fase1 import ModeloFase1, IN_CHANNELS


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ALOHADataModule(L.LightningDataModule):
    """
    Carga X_train/Y_train/W_train y X_val/Y_val/W_val desde disco.
    Transpone (N, L, C) → (N, C, L) para Conv1d.
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
      num_clases>1 → CrossEntropyLoss con pesos por muestra (multiclase)

    Métricas registradas en cada época:
      train_loss, val_loss       — pérdida media ponderada
      train_acc, val_acc         — accuracy (top-1 para multiclase, threshold 0.5 binario)
      val_auc                    — ROC-AUC binario C1 vs resto en validación
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
        # Buffers para acumular predicciones de val y calcular AUC por época
        self._val_scores: list = []
        self._val_labels: list = []

    def forward(self, x):
        return self.modelo(x)

    def _paso_comun(self, batch, etapa: str):
        x, y, w = batch
        out = self(x)  # (N, num_clases)

        if self.binario:
            logit = out.squeeze(1)
            loss_vec = self.criterio(logit, y)
        else:
            loss_vec = self.criterio(out, y.long())

        loss = (loss_vec * w).sum() / torch.clamp(w.sum(), min=1e-8)

        with torch.no_grad():
            if self.binario:
                pred = (torch.sigmoid(out.squeeze(1)) >= 0.5).float()
                scores = torch.sigmoid(out.squeeze(1))
            else:
                pred = out.argmax(dim=1).float()
                # Score de detección = P(C1)
                scores = torch.softmax(out, dim=1)[:, 1]
            acc = (pred == (y if self.binario else y)).float().mean()

            # Acumular scores y etiquetas binarias C1-vs-resto para AUC
            if etapa == "val":
                labels_bin = (y == 1).float() if not self.binario else y
                self._val_scores.append(scores.cpu())
                self._val_labels.append(labels_bin.cpu())

        self.log(f"{etapa}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{etapa}_acc",  acc,  on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, _):
        return self._paso_comun(batch, "train")

    def validation_step(self, batch, _):
        self._paso_comun(batch, "val")

    def on_validation_epoch_end(self):
        """Calcula y registra val_auc al final de cada época de validación."""
        if not self._val_scores:
            return
        try:
            from sklearn.metrics import roc_auc_score
            scores = torch.cat(self._val_scores).numpy()
            labels = torch.cat(self._val_labels).numpy()
            # Solo calcular si hay ambas clases presentes
            if len(np.unique(labels)) > 1:
                auc = float(roc_auc_score(labels, scores))
                self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        except Exception:
            pass
        self._val_scores.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


# ---------------------------------------------------------------------------
# Callback: guardar curvas de entrenamiento en PNG (estilo Pol Simon)
# ---------------------------------------------------------------------------

class CurvaEntrenamientoCallback(Callback):
    """
    Al final del entrenamiento guarda un PNG con:
      (a) train_loss y val_loss vs época
      (b) val_auc vs época
    Curvas estilo Pol Simon Fig. 13 — listas para la memoria.
    """

    def __init__(self, ruta_png: str, nombre_experimento: str = ""):
        self.ruta_png = ruta_png
        self.nombre = nombre_experimento
        self._hist_train_loss: list = []
        self._hist_val_loss:   list = []
        self._hist_val_auc:    list = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        tl = metrics.get("train_loss")
        if tl is not None:
            self._hist_train_loss.append(float(tl))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        vl  = metrics.get("val_loss")
        auc = metrics.get("val_auc")
        if vl  is not None: self._hist_val_loss.append(float(vl))
        if auc is not None: self._hist_val_auc.append(float(auc))

    def on_train_end(self, trainer, pl_module):
        self._guardar_figura()

    def _guardar_figura(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(os.path.dirname(self.ruta_png) or ".", exist_ok=True)

            n_subplots = 2 if self._hist_val_auc else 1
            fig, axes = plt.subplots(1, n_subplots, figsize=(5.5 * n_subplots, 4.2))
            if n_subplots == 1:
                axes = [axes]

            epochs_loss = range(1, len(self._hist_val_loss) + 1)
            epochs_auc  = range(1, len(self._hist_val_auc)  + 1)

            # (a) Loss
            ax = axes[0]
            if self._hist_train_loss:
                ax.plot(range(1, len(self._hist_train_loss) + 1),
                        self._hist_train_loss, label="Train", color="tab:blue")
            ax.plot(epochs_loss, self._hist_val_loss,
                    label="Validation", color="tab:orange")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss")
            ax.set_title(f"(a) Loss  —  {self.nombre}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # (b) val AUC
            if self._hist_val_auc:
                ax2 = axes[1]
                ax2.plot(epochs_auc, self._hist_val_auc,
                         label="Validation", color="tab:orange")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("ROC AUC")
                ax2.set_title(f"(b) ROC AUC  —  {self.nombre}")
                ax2.set_ylim(
                    max(0.4, min(self._hist_val_auc) - 0.02),
                    min(1.0, max(self._hist_val_auc) + 0.02),
                )
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(self.ruta_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [curvas] Guardada: {self.ruta_png}")
        except Exception as e:
            print(f"  [curvas] No se pudo guardar la figura: {e}")


# ---------------------------------------------------------------------------
# Función principal de entrenamiento
# ---------------------------------------------------------------------------

def entrenar(
    directorio_datos: str,
    representacion: str = "iq",
    modo_label: str = "onset_centro",
    num_clases: int = 1,
    directorio_ckpt: str = "checkpoints_v2",
    ruta_curvas: str = None,
    max_epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    dropout: float = 0.3,
    usar_wandb: bool = True,
    proyecto_wandb: str = "tfg-aloha-detector",
    nombre_run_wandb: str = None,
    num_workers: int = 0,
) -> str:
    """
    Entrena ModeloFase1 y guarda el mejor checkpoint.

    Estructura de salida:
      <directorio_ckpt>/<representacion>_<modo_label>/
          mejor.ckpt                    ← mejor según val_loss
    """
    in_channels = IN_CHANNELS[representacion]
    subcarpeta   = f"{representacion}_{modo_label}"
    dir_ckpt_exp = os.path.join(directorio_ckpt, subcarpeta)
    os.makedirs(dir_ckpt_exp, exist_ok=True)

    nombre_exp = f"{representacion} / {modo_label}"
    if ruta_curvas is None:
        ruta_curvas = os.path.join(dir_ckpt_exp, "curvas_entrenamiento.png")

    datamodule = ALOHADataModule(directorio_datos, batch_size=batch_size,
                                  num_workers=num_workers)
    modelo = DetectorLightning(in_channels=in_channels, num_clases=num_clases,
                                lr=lr, dropout=dropout)

    callbacks = [
        ModelCheckpoint(
            dirpath=dir_ckpt_exp,
            filename="mejor",          # → mejor.ckpt
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(monitor="val_loss", patience=12, mode="min", verbose=True),
        CurvaEntrenamientoCallback(
            ruta_png=ruta_curvas,
            nombre_experimento=nombre_exp,
        ),
    ]

    run_name = nombre_run_wandb or subcarpeta
    loggers = ([WandbLogger(project=proyecto_wandb, name=run_name, log_model=False)]
               if usar_wandb else [])

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=loggers if loggers else False,
        log_every_n_steps=50,
        enable_model_summary=True,
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
    parser = argparse.ArgumentParser(description="Entrena ModeloFase1 (v2).")
    parser.add_argument("--datos",          type=str,   required=True)
    parser.add_argument("--representacion", type=str,   default="iq",
                        choices=["energia", "iq", "iq_energia"])
    parser.add_argument("--modo_label",     type=str,   default="onset_centro",
                        choices=["onset_centro", "ventana_llena", "multiclase_onset"])
    parser.add_argument("--num_clases",     type=int,   default=1,
                        help="1=binario, 3=multiclase")
    parser.add_argument("--ckpt",           type=str,   default="checkpoints_v2")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch",          type=int,   default=512)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--sin_wandb",      action="store_true")
    parser.add_argument("--workers",        type=int,   default=0)
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
