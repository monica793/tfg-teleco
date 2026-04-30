"""
Punto de entrada principal — campaña experimental Fase 1.

Flujo de trabajo
----------------
1. Generar datasets (uno por representación):
   python -m ml.generar_dataset --representacion energia --salida data/fase1_energia_onset_centro
   python -m ml.generar_dataset --representacion iq      --salida data/fase1_iq_onset_centro
   python -m ml.generar_dataset --representacion iq_energia --salida data/fase1_iq_energia_onset_centro

2. Entrenar (uno por representación, en Colab con GPU):
   python -m ml.entrenar_modelo --datos data/fase1_energia_onset_centro --representacion energia --sin_wandb
   python -m ml.entrenar_modelo --datos data/fase1_iq_onset_centro      --representacion iq      --sin_wandb
   python -m ml.entrenar_modelo --datos data/fase1_iq_energia_onset_centro --representacion iq_energia --sin_wandb

3. Evaluar (en local, ejecutar este main.py):
   Rellenar CHECKPOINTS_FASE1 y ejecutar:
   python main.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.escenario_phy import generar_escenario_phy, ejecutar_receptor_neuronal
from pipeline.correlator_decoder import correlador
from pipeline.protocolo_evaluacion import (
    NUM_BITS_PRE, NUM_BITS_DATOS, SEMILLA_BASE, TOLERANCIA_MUESTRAS,
)
from pipeline.visualization import plot_zoom_respuesta_detectores
from ml.modelo_fase1 import cargar_checkpoint
from ml.evaluar import (
    tabla_metricas,
    histograma_fp_distancia,
    score_medio_vs_offset,
    curva_f1_vs_snr,
    curva_f1_vs_g,
    comparar_representaciones,
)

ROOT    = os.path.dirname(__file__)
FIGURES = os.path.join(ROOT, "results", "figures")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — rellenar antes de ejecutar
# ---------------------------------------------------------------------------

CHECKPOINTS_FASE1 = {
    "energia":    "checkpoints/energia_onset_centro-XX-X.XXXX.ckpt",  # ← rellenar
    "iq":         "checkpoints/iq_onset_centro-XX-X.XXXX.ckpt",       # ← rellenar
    "iq_energia": "checkpoints/iq_energia_onset_centro-XX-X.XXXX.ckpt",  # ← rellenar
}

# Escenarios del diagnóstico progresivo (definidos aquí, no hardcodeados en funciones)
ESCENARIOS_DIAGNOSTICO = [
    {"label": "sin_colision_sin_ruido", "G": 0.1, "SNR": 50.0},
    {"label": "sin_colision_con_ruido", "G": 0.1, "SNR":  6.0},
    {"label": "con_colision_con_ruido", "G": 0.4, "SNR":  6.0},
]

# Grids para curvas de degradación
LISTA_SNR_DEGRADACION = [-3.0, 0.0, 3.0, 6.0, 10.0, 15.0]
LISTA_G_DEGRADACION   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]


# ---------------------------------------------------------------------------
# 1. Diagnóstico progresivo por representación
# ---------------------------------------------------------------------------

def diagnostico_fase1(
    ruta_checkpoint: str,
    representacion: str,
    tau: float = 0.5,
    temperature: float = 1.0,
    semilla: int = 42,
):
    """
    Evalúa un checkpoint en los 3 escenarios del diagnóstico progresivo.
    Genera métricas numéricas y figura de zoom por escenario.
    """
    if not os.path.exists(ruta_checkpoint):
        print(f"  [SKIP] Checkpoint no encontrado: {ruta_checkpoint}")
        return {}

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint(ruta_checkpoint, map_location=dispositivo)

    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO — {representacion}")
    print(f"  Checkpoint: {ruta_checkpoint}")
    print(f"  tau={tau}  T={temperature}")
    print(f"{'='*70}")

    resultados = {}
    taus = np.linspace(0.0, 1.0, 201)

    for cfg in ESCENARIOS_DIAGNOSTICO:
        esc = generar_escenario_phy(
            carga_G=cfg["G"], ventana_frame_times=300, snr_db=cfg["SNR"],
            num_bits_pre=NUM_BITS_PRE, num_bits_datos=NUM_BITS_DATOS,
            semilla=semilla, usar_preambulo=False,
        )
        sal = ejecutar_receptor_neuronal(
            escenario=esc, modelo=modelo, umbral=tau,
            temperature=temperature, dispositivo=dispositivo,
            stride=1, long_ventana=128,
        )
        score = np.asarray(sal["score_por_muestra"], dtype=np.float32)
        met = tabla_metricas(score, esc["instantes_llegada_muestras"],
                              TOLERANCIA_MUESTRAS, taus=taus)

        label = cfg["label"]
        print(f"\n  [{label}]  G={cfg['G']}  SNR={cfg['SNR']} dB")
        print(f"    PR-AUC={met['pr_auc']:.4f}  ROC-AUC={met['roc_auc']:.4f}"
              f"  F1_best={met['f1_best']:.4f} (tau={met['tau_best']:.2f})")
        print(f"    TP={met['tp']}  FP={met['fp']}  FN={met['fn']}")

        # Zoom local
        if len(esc["instantes_llegada_muestras"]) > 0:
            centro = int(esc["instantes_llegada_muestras"][0])
            ruta_fig = os.path.join(FIGURES, f"diag_{representacion}_{label}.png")
            plot_zoom_respuesta_detectores(
                corr_norm=np.zeros_like(score),
                score_ml=score,
                instantes_reales=esc["instantes_llegada_muestras"],
                detecciones_corr=np.array([], dtype=np.int64),
                detecciones_ml=sal["instantes_detectados"],
                centro_muestra=centro,
                ancho_ventana=250,
                tau_corr=1.1,
                tau_ml=tau,
                ruta_salida=ruta_fig,
            )

        # Histograma FP y score vs offset
        ruta_hist = os.path.join(FIGURES, f"hist_fp_{representacion}_{label}.png")
        histograma_fp_distancia(score, esc["instantes_llegada_muestras"],
                                 tau=met["tau_best"], ruta_salida=ruta_hist)

        ruta_offset = os.path.join(FIGURES, f"score_offset_{representacion}_{label}.png")
        score_medio_vs_offset(score, esc["instantes_llegada_muestras"],
                               ruta_salida=ruta_offset)

        resultados[label] = met

    return resultados


# ---------------------------------------------------------------------------
# 2. Curvas de degradación
# ---------------------------------------------------------------------------

def degradacion_fase1(ruta_checkpoint: str, representacion: str):
    """Genera curvas F1 vs SNR y F1 vs G para un checkpoint."""
    if not os.path.exists(ruta_checkpoint):
        print(f"  [SKIP] Checkpoint no encontrado: {ruta_checkpoint}")
        return

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint(ruta_checkpoint, map_location=dispositivo)

    curva_f1_vs_snr(
        detector=modelo, lista_snr=LISTA_SNR_DEGRADACION,
        carga_G=0.4, n_iter=5, dispositivo=dispositivo,
        representacion=representacion,
        ruta_salida=os.path.join(FIGURES, f"f1_vs_snr_{representacion}.png"),
    )
    curva_f1_vs_g(
        detector=modelo, lista_g=LISTA_G_DEGRADACION,
        snr_db=6.0, n_iter=5, dispositivo=dispositivo,
        representacion=representacion,
        ruta_salida=os.path.join(FIGURES, f"f1_vs_g_{representacion}.png"),
    )
    print(f"  Curvas de degradación guardadas para '{representacion}'.")


# ---------------------------------------------------------------------------
# 3. Comparativa final R1/R2/R3
# ---------------------------------------------------------------------------

def comparativa_fase1(semilla: int = 42, tau: float = 0.5):
    """
    Evalúa los 3 checkpoints en el escenario objetivo (G=0.5, SNR=3 dB)
    y genera tabla comparativa.
    """
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    esc = generar_escenario_phy(
        carga_G=0.5, ventana_frame_times=400, snr_db=3.0,
        num_bits_pre=NUM_BITS_PRE, num_bits_datos=NUM_BITS_DATOS,
        semilla=semilla, usar_preambulo=False,
    )

    resultados = {}
    for rep, ruta in CHECKPOINTS_FASE1.items():
        if not os.path.exists(ruta):
            continue
        modelo = cargar_checkpoint(ruta, map_location=dispositivo)
        sal = ejecutar_receptor_neuronal(
            escenario=esc, modelo=modelo, umbral=tau,
            dispositivo=dispositivo, stride=1, long_ventana=128,
        )
        score = np.asarray(sal["score_por_muestra"], dtype=np.float32)
        resultados[rep] = tabla_metricas(score, esc["instantes_llegada_muestras"],
                                          TOLERANCIA_MUESTRAS)

    comparar_representaciones(
        resultados,
        ruta_salida=os.path.join(RESULTS, "comparativa_fase1.csv"),
    )


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Paso 1: Diagnóstico progresivo por representación ---
    for rep, ruta in CHECKPOINTS_FASE1.items():
        diagnostico_fase1(ruta_checkpoint=ruta, representacion=rep)

    # --- Paso 2: Curvas de degradación ---
    # for rep, ruta in CHECKPOINTS_FASE1.items():
    #     degradacion_fase1(ruta_checkpoint=ruta, representacion=rep)

    # --- Paso 3: Tabla comparativa en escenario objetivo ---
    # comparativa_fase1()
