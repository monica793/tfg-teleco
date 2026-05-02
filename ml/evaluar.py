"""
Funciones de evaluación y análisis para la campaña experimental.

Todas las funciones operan sobre 'score_array' (array de scores por muestra),
que puede provenir tanto de la red neuronal como del correlador clásico.
Esto garantiza comparabilidad simétrica entre detectores.

Funciones disponibles
---------------------
tabla_metricas        : PR-AUC, ROC-AUC, F1_best, FP@recall, TP/FP/FN en tau_best
histograma_fp_distancia: distribución de FP respecto al onset más cercano
score_medio_vs_offset : score promedio en función del desplazamiento al onset
curva_f1_vs_snr       : degradación de F1_best al barrer SNR (G fijo)
curva_f1_vs_g         : degradación de F1_best al barrer G (SNR fijo)
comparar_representaciones: tabla resumen para comparar R1/R2/R3
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pipeline.metricas_receptor import curva_roc_por_indice, curva_pr_por_indice
from pipeline.escenario_phy import generar_escenario_phy
from pipeline.protocolo_evaluacion import (
    NUM_BITS_PRE, NUM_BITS_DATOS, SEMILLA_BASE, TOLERANCIA_MUESTRAS
)


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _taus_estandar():
    return np.linspace(0.0, 1.0, 201, dtype=float)


def _f1_desde_pr(precision, recall):
    denom = np.maximum(1e-12, precision + recall)
    return 2.0 * precision * recall / denom


def _fp_a_recall_objetivo(recall_arr, fpr_arr, n_neg, recall_obj=0.8):
    """FP absolutos cuando recall >= recall_obj (interpolado)."""
    idx = np.where(recall_arr >= recall_obj)[0]
    if len(idx) == 0:
        return np.nan
    tpr_interp = recall_arr[idx[0]]
    fpr_interp = fpr_arr[idx[0]]
    return float(fpr_interp * n_neg)


# ---------------------------------------------------------------------------
# Tabla de métricas estándar
# ---------------------------------------------------------------------------

def tabla_metricas(
    score: np.ndarray,
    instantes_verdaderos: np.ndarray,
    tolerancia_muestras: int = TOLERANCIA_MUESTRAS,
    recall_objetivo: float = 0.8,
    taus=None,
) -> dict:
    """
    Calcula el conjunto completo de métricas para un array de scores.

    Parámetros
    ----------
    score                : array (N,) con scores por muestra [0, 1]
    instantes_verdaderos : array con los instantes reales de inicio
    tolerancia_muestras  : tolerancia para definir TP a nivel de índice
    recall_objetivo      : para calcular FP@recall_obj

    Retorna dict con:
        pr_auc, roc_auc, f1_best, tau_best,
        precision_best, recall_best,
        tp, fp, fn, n_verdaderos,
        fp_at_recall_obj
    """
    if taus is None:
        taus = _taus_estandar()

    roc = curva_roc_por_indice(score, instantes_verdaderos, tolerancia_muestras, taus)
    pr  = curva_pr_por_indice(score, instantes_verdaderos, tolerancia_muestras, taus)

    f1 = _f1_desde_pr(pr["precision"], pr["recall"])
    idx_best = int(np.nanargmax(f1)) if f1.size else 0

    n_neg = int(roc["tn"][idx_best]) + int(roc["fp"][idx_best])
    fp_obj = _fp_a_recall_objetivo(pr["recall"], roc["fpr"], n_neg, recall_objetivo)

    return {
        "pr_auc":         float(pr["pr_auc"]),
        "roc_auc":        float(roc["auc"]),
        "f1_best":        float(f1[idx_best]),
        "tau_best":       float(taus[idx_best]),
        "precision_best": float(pr["precision"][idx_best]),
        "recall_best":    float(pr["recall"][idx_best]),
        "tp":             int(roc["tp"][idx_best]),
        "fp":             int(roc["fp"][idx_best]),
        "fn":             int(roc["fn"][idx_best]),
        "tn":             int(roc["tn"][idx_best]),
        "n_verdaderos":   len(instantes_verdaderos),
        f"fp_at_recall_{recall_objetivo:.0%}": fp_obj,
    }


# ---------------------------------------------------------------------------
# Análisis espacial de FP
# ---------------------------------------------------------------------------

def histograma_fp_distancia(
    score: np.ndarray,
    instantes_verdaderos: np.ndarray,
    tau: float,
    max_dist: int = 128,
    ruta_salida: str = None,
) -> np.ndarray:
    """
    Histograma de la distancia entre cada FP y el onset más cercano.

    Permite distinguir:
    - FP concentrados cerca del onset → problema de resolución (resultado interesante)
    - FP distribuidos uniformemente  → el modelo no localiza nada útil

    Retorna el array de distancias de todos los FP.
    """
    detecciones = np.where(score >= tau)[0]
    verdaderos = np.asarray(instantes_verdaderos, dtype=np.int64)

    fp_dists = []
    for det in detecciones:
        if len(verdaderos) == 0:
            fp_dists.append(max_dist)
            continue
        dist_min = int(np.min(np.abs(verdaderos - det)))
        if dist_min > 0:   # es FP (no TP)
            fp_dists.append(min(dist_min, max_dist))

    fp_dists = np.array(fp_dists, dtype=int)

    if ruta_salida:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(fp_dists, bins=min(50, max_dist), color="tab:red", alpha=0.75, edgecolor="white")
        ax.set_xlabel("Distancia al onset más cercano (muestras)")
        ax.set_ylabel("N° de FP")
        ax.set_title(f"Distribución espacial de FP (tau={tau:.2f})\n"
                     f"Total FP: {len(fp_dists)}")
        ax.axvline(x=64, color="gray", linestyle="--", alpha=0.5, label="mitad de ventana")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ruta_salida, dpi=180)
        plt.close(fig)

    return fp_dists


# ---------------------------------------------------------------------------
# Score medio vs offset al onset
# ---------------------------------------------------------------------------

def score_medio_vs_offset(
    score: np.ndarray,
    instantes_verdaderos: np.ndarray,
    semiancho: int = 64,
    ruta_salida: str = None,
) -> tuple:
    """
    Score promedio en función del desplazamiento relativo al onset.

    Para cada onset real t, acumula score[t + delta] para delta en [-semiancho, semiancho].
    El resultado muestra si la red aprende un pico centrado en el onset
    o una función plana (meseta).

    Retorna (offsets, scores_medios).
    """
    offsets = np.arange(-semiancho, semiancho + 1)
    acum = np.zeros(len(offsets), dtype=np.float64)
    cuenta = np.zeros(len(offsets), dtype=np.int64)
    N = len(score)

    for t in instantes_verdaderos:
        t = int(t)
        for i, delta in enumerate(offsets):
            idx = t + delta
            if 0 <= idx < N:
                acum[i] += float(score[idx])
                cuenta[i] += 1

    scores_medios = np.where(cuenta > 0, acum / cuenta, np.nan)

    if ruta_salida:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(offsets, scores_medios, color="tab:blue", lw=2)
        ax.axvline(x=0, color="black", linestyle="--", lw=1, label="onset real")
        ax.axhline(y=0.5, color="gray", linestyle=":", lw=1, label="tau=0.5")
        ax.set_xlabel("Desplazamiento al onset (muestras)")
        ax.set_ylabel("Score medio")
        ax.set_title("Score medio vs offset al onset")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ruta_salida, dpi=180)
        plt.close(fig)

    return offsets, scores_medios


# ---------------------------------------------------------------------------
# Curvas de degradación
# ---------------------------------------------------------------------------

def _evaluar_escenario(modelo_o_correlador, escenario, es_correlador, dispositivo,
                        representacion, long_ventana):
    """Evalúa un detector en un escenario y devuelve el score_array."""
    import torch
    from pipeline.escenario_phy import ejecutar_receptor_neuronal
    from pipeline.correlator_decoder import correlador

    if es_correlador:
        return correlador(escenario["senal_rx"], escenario["preambulo"])
    else:
        sal = ejecutar_receptor_neuronal(
            escenario=escenario,
            modelo=modelo_o_correlador,
            umbral=0.5,
            temperature=1.0,
            dispositivo=dispositivo,
            stride=1,
            long_ventana=long_ventana,
        )
        return sal["score_por_muestra"]


def curva_f1_vs_snr(
    detector,
    lista_snr: list,
    carga_G: float = 0.4,
    ventana_frame_times: int = 300,
    semilla_base: int = SEMILLA_BASE,
    n_iter: int = 5,
    tolerancia_muestras: int = TOLERANCIA_MUESTRAS,
    es_correlador: bool = False,
    dispositivo: str = "cpu",
    representacion: str = "iq",
    long_ventana: int = 128,
    usar_preambulo: bool = False,
    ruta_salida: str = None,
) -> tuple:
    """
    Curva F1_best vs SNR para G fijo.

    Promedia sobre n_iter realizaciones para reducir varianza.
    Sirve para NN y correlador con el mismo protocolo.

    Retorna (lista_snr, f1_medias, f1_stds).
    """
    f1_medias, f1_stds = [], []
    taus = _taus_estandar()

    for snr in lista_snr:
        f1s = []
        for k in range(n_iter):
            esc = generar_escenario_phy(
                carga_G=float(carga_G),
                ventana_frame_times=ventana_frame_times,
                snr_db=float(snr),
                num_bits_pre=NUM_BITS_PRE,
                num_bits_datos=NUM_BITS_DATOS,
                semilla=semilla_base + k,
                usar_preambulo=usar_preambulo or es_correlador,
            )
            score = _evaluar_escenario(detector, esc, es_correlador,
                                        dispositivo, representacion, long_ventana)
            pr = curva_pr_por_indice(score, esc["instantes_llegada_muestras"],
                                      tolerancia_muestras, taus)
            f1 = _f1_desde_pr(pr["precision"], pr["recall"])
            f1s.append(float(np.nanmax(f1)) if f1.size else 0.0)
        f1_medias.append(float(np.mean(f1s)))
        f1_stds.append(float(np.std(f1s)))

    if ruta_salida:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(lista_snr, f1_medias, yerr=f1_stds, marker="o", capsize=4, lw=2)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("F1_best")
        ax.set_title(f"Degradación F1 vs SNR  (G={carga_G})")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ruta_salida, dpi=180)
        plt.close(fig)

    return lista_snr, f1_medias, f1_stds


def curva_f1_vs_g(
    detector,
    lista_g: list,
    snr_db: float = 6.0,
    ventana_frame_times: int = 300,
    semilla_base: int = SEMILLA_BASE,
    n_iter: int = 5,
    tolerancia_muestras: int = TOLERANCIA_MUESTRAS,
    es_correlador: bool = False,
    dispositivo: str = "cpu",
    representacion: str = "iq",
    long_ventana: int = 128,
    usar_preambulo: bool = False,
    ruta_salida: str = None,
) -> tuple:
    """
    Curva F1_best vs G para SNR fijo.

    Retorna (lista_g, f1_medias, f1_stds).
    """
    f1_medias, f1_stds = [], []
    taus = _taus_estandar()

    for g in lista_g:
        f1s = []
        for k in range(n_iter):
            esc = generar_escenario_phy(
                carga_G=float(g),
                ventana_frame_times=ventana_frame_times,
                snr_db=float(snr_db),
                num_bits_pre=NUM_BITS_PRE,
                num_bits_datos=NUM_BITS_DATOS,
                semilla=semilla_base + k,
                usar_preambulo=usar_preambulo or es_correlador,
            )
            score = _evaluar_escenario(detector, esc, es_correlador,
                                        dispositivo, representacion, long_ventana)
            pr = curva_pr_por_indice(score, esc["instantes_llegada_muestras"],
                                      tolerancia_muestras, taus)
            f1 = _f1_desde_pr(pr["precision"], pr["recall"])
            f1s.append(float(np.nanmax(f1)) if f1.size else 0.0)
        f1_medias.append(float(np.mean(f1s)))
        f1_stds.append(float(np.std(f1s)))

    if ruta_salida:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(lista_g, f1_medias, yerr=f1_stds, marker="o", capsize=4, lw=2)
        ax.set_xlabel("Carga G")
        ax.set_ylabel("F1_best")
        ax.set_title(f"Degradación F1 vs G  (SNR={snr_db} dB)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ruta_salida, dpi=180)
        plt.close(fig)

    return lista_g, f1_medias, f1_stds


# ---------------------------------------------------------------------------
# Tabla comparativa R1/R2/R3 (Fase 1)
# ---------------------------------------------------------------------------

def comparar_representaciones(
    resultados: dict,
    ruta_salida: str = None,
) -> None:
    """
    Imprime y guarda tabla comparativa de representaciones.

    Parámetros
    ----------
    resultados : dict
        Clave: nombre representación (ej. "iq", "energia", "iq_energia")
        Valor: dict devuelto por tabla_metricas()
    ruta_salida : si se indica, guarda la tabla como CSV
    """
    cabecera = ["representacion", "pr_auc", "roc_auc", "f1_best",
                "tau_best", "tp", "fp", "fn", "fp_at_recall_80%"]
    print("\n" + "=" * 90)
    print("COMPARATIVA FASE 1 — Representaciones")
    print("=" * 90)
    print(f"{'Rep.':<14} {'PR-AUC':>8} {'ROC-AUC':>9} {'F1_best':>9} "
          f"{'tau':>6} {'TP':>7} {'FP':>7} {'FN':>7} {'FP@R80':>9}")
    print("-" * 90)

    filas = []
    for rep, m in resultados.items():
        fp80 = m.get("fp_at_recall_80%", np.nan)
        print(f"{rep:<14} {m['pr_auc']:>8.4f} {m['roc_auc']:>9.4f} {m['f1_best']:>9.4f} "
              f"{m['tau_best']:>6.2f} {m['tp']:>7} {m['fp']:>7} {m['fn']:>7} "
              f"{fp80:>9.0f}")
        filas.append({
            "representacion": rep,
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
            "f1_best": m["f1_best"], "tau_best": m["tau_best"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "fp_at_recall_80%": fp80,
        })
    print("=" * 90)

    if ruta_salida and filas:
        import csv
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        with open(ruta_salida, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cabecera)
            w.writeheader()
            w.writerows(filas)
        print(f"CSV guardado: {ruta_salida}")


def etiquetas_multiclase_por_onset(
    n_muestras: int,
    instantes_verdaderos: np.ndarray,
    k_c1: int = 2,
    k_c2: int = 12,
) -> np.ndarray:
    """
    Construye etiqueta multiclase por índice de muestra:
      C1: |d| <= k_c1
      C2: k_c1 < |d| <= k_c2
      C0: resto
    donde d es distancia al onset más cercano.
    """
    y = np.zeros(n_muestras, dtype=np.int64)
    if len(instantes_verdaderos) == 0:
        return y
    dmin = np.full(n_muestras, np.inf, dtype=float)
    idx = np.arange(n_muestras, dtype=np.int64)
    for t in instantes_verdaderos:
        dmin = np.minimum(dmin, np.abs(idx - int(t)))
    y[dmin <= k_c1] = 1
    y[(dmin > k_c1) & (dmin <= k_c2)] = 2
    return y


def matriz_confusion_multiclase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_clases: int = 3,
) -> np.ndarray:
    """Matriz de confusión CxC para clases enteras [0..C-1]."""
    cm = np.zeros((num_clases, num_clases), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_clases and 0 <= p < num_clases:
            cm[t, p] += 1
    return cm
