"""
Funciones de test de experimentos anteriores a la campaña Fase 1.

Se conservan exclusivamente como referencia para la memoria del TFG.
No deben usarse para nuevos experimentos.

Para evaluaciones de Fase 1 en adelante, usar ml/evaluar.py y main.py.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── rutas ──────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(__file__)
FIGURES = os.path.join(ROOT, 'results', 'figures')
MODELS  = os.path.join(ROOT, 'models', 'trained')
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(MODELS,  exist_ok=True)

# ── imports de tus módulos ─────────────────────────────────────────────────
from aloha.pure_aloha import simular_pure_aloha, throughput_teorico
from pipeline.transmitter import generar_preambulo, generar_paquete
from pipeline.channel import canal_awgn, canal_awgn_colision
from pipeline.correlator_decoder import buscar_picos_preambulo, correlador
from pipeline.escenario_phy import (
    barrer_grid_protocolo_correlador,
    ejecutar_monte_carlo_receptor_correlador,
    ejecutar_monte_carlo_roc_correlador,
    ejecutar_monte_carlo_roc_neuronal,
    generar_escenario_phy,
    ejecutar_receptor_neuronal,
)
from pipeline.metricas_receptor import (
    curva_pr_por_indice,
    curva_roc_por_indice,
    evaluar_detecciones,
    metricas_evento_derivadas,
)
from ml.modelo import ModeloCNN, cargar_checkpoint_automatico
from pipeline.protocolo_evaluacion import (
    GRID_CARGA_G,
    GRID_SNR_DB,
    NUM_BITS_DATOS,
    NUM_BITS_PRE,
    NUM_ITERACIONES_MC,
    NUM_ITERACIONES_MC_RAPIDO,
    SEMILLA_BASE,
    TOLERANCIA_MUESTRAS,
)
from pipeline.visualization import (
    plot_colision_correlador,
    plot_respuesta_correlador_vs_ml,
    plot_zoom_respuesta_detectores,
    plot_roc_comparativa_correlador_vs_ml,
    plot_deteccion_correlador_awgn,
    plot_pure_aloha,
    plot_roc_familia_por_snr,
    plot_roc_correlador,
)


# ===========================================================================
# 1. EXPERIMENTO: CAPA MAC (Tráfico Pure ALOHA)
# ===========================================================================
def test_traffic_model_aloha():
    print("=== Pure ALOHA ===")
    G_valores = [round(g * 0.1, 1) for g in range(1, 31)]
    # Ejecutamos la simulación real para cada punto de G
    S_sim = [simular_pure_aloha(g, ventana_frame_times=100000, solo_throughput=True) for g in G_valores]
    plot_pure_aloha(G_valores, S_sim, FIGURES)

# ===========================================================================
# 2. EXPERIMENTO: CAPA PHY (Correlador y Ruido)
# ===========================================================================
def test_phy_correlador(snr_db, num_bits_pre, num_bits_data):
    # --- parámetros ---
    #NUM_BITS_PRE  = 13       # longitud del preámbulo
    #NUM_BITS_DATA = 20       # longitud de los datos
    TAU           = 0.7      # umbral de detección
 
    np.random.seed(0)        # reproducibilidad
 
    # Bloque 1 — Transmisor
    preambulo = generar_preambulo(num_bits=num_bits_pre)
    paquete   = generar_paquete(preambulo, num_bits_datos=num_bits_data)
 
    print(f"Preámbulo ({num_bits_pre} símbolos): {preambulo}")
    print(f"Paquete total: {len(paquete)} símbolos "
          f"({num_bits_pre} preámbulo + {num_bits_data} datos)")
 
    # Bloque 2 — Canal
    señal_rx, instante = canal_awgn(paquete, SNR_dB=snr_db)
 
    print(f"\nInstante de llegada real: muestra {instante}")
    print(f"Longitud señal recibida:  {len(señal_rx)} muestras")
    print(f"SNR = {snr_db} dB  →  el paquete queda oculto en el ruido")
 
    # Bloque 3 — Correlador
    corr_norm = correlador(señal_rx, preambulo)
 
    pico_val = np.max(corr_norm)
    pico_idx = np.argmax(corr_norm)
    print(f"\nPico de correlación: {pico_val:.4f} en muestra {pico_idx}")
    print(f"Instante real:       {instante}  →  "
          f"error = {abs(pico_idx - instante)} muestras")
    print(f"Detección: {'SÍ' if pico_val >= TAU else 'NO'} "
          f"(umbral τ = {TAU})")
 
    # Bloque 4 — Visualización
    plot_deteccion_correlador_awgn(señal_rx, corr_norm, instante,
                                     num_bits_pre, tau=TAU, SNR_dB=snr_db)

def test_colision_phy():
    """
    Prueba de integración MAC → PHY: colisión parcial de dos paquetes
    Pure ALOHA vista a nivel de correlador.
 
    Escenario
    ---------
    - Paquete 1 llega en muestra 50
    - Paquete 2 llega en muestra 80
    - Paquete de 33 muestras (13 preámbulo + 20 datos)
    → solapamiento de 3 muestras del preámbulo del paquete 2
      con la cola de datos del paquete 1
 
    Decisión sobre SNR=3 dB (no 10 dB como pedías originalmente):
    --------------------------------------------------------------
    Con SNR=10 dB el correlador detecta ambos paquetes perfectamente
    y la colisión no tiene efecto visible. Con SNR=3 dB el ruido
    es suficiente para ver el suelo de correlación claramente y
    apreciar cómo la interferencia en la zona solapada degrada el
    pico del segundo paquete. Esto hace la visualización más
    informativa. Puedes subir SNR_dB para ver el caso ideal.
    """
    np.random.seed(7)   # reproducibilidad
 
    # --- parámetros ---
    SNR_dB        = 3       # bajo para visualización ilustrativa
    NUM_BITS_PRE  = 13
    NUM_BITS_DATA = 20
    LONGITUD      = 200     # muestras de observación total
    TAU           = 0.7
 
    T1 = 50   # muestra de llegada paquete 1
    T2 = 80   # muestra de llegada paquete 2
    # con paquete de 33 muestras (13+20), paquete 1 ocupa [50, 83)
    # paquete 2 empieza en 80 → solapamiento en [80, 83) = 3 muestras
 
    # --- Bloque 1: Transmisor ---
    preambulo = generar_preambulo(num_bits=NUM_BITS_PRE)
    paquete1  = generar_paquete(preambulo, num_bits_datos=NUM_BITS_DATA)
    paquete2  = generar_paquete(preambulo, num_bits_datos=NUM_BITS_DATA)
 
    len_paquete = len(paquete1)   # 33 muestras
    solapamiento = (T1 + len_paquete) - T2
    print("=" * 55)
    print("TEST: Colisión PHY — Pure ALOHA")
    print("=" * 55)
    print(f"  Paquete 1 : muestras [{T1}, {T1 + len_paquete})")
    print(f"  Paquete 2 : muestras [{T2}, {T2 + len_paquete})")
    print(f"  Solapamiento: {max(0, solapamiento)} muestras")
    print(f"  SNR = {SNR_dB} dB  |  τ = {TAU}")
    print("-" * 55)
 
    # --- Bloque 2: Canal con colisión ---
    paquetes_y_tiempos = [
        (paquete1, T1),
        (paquete2, T2),
    ]
    senal_rx = canal_awgn_colision(paquetes_y_tiempos,
                                    SNR_dB=SNR_dB,
                                    longitud_total=LONGITUD)
 
    # --- Bloque 3: Correlador ---
    corr_norm = correlador(senal_rx, preambulo)
 
    # --- Bloque 4: Visualización ---
    plot_colision_correlador(
        senal_rx        = senal_rx,
        corr_norm       = corr_norm,
        instantes_reales= [T1, T2],
        len_preambulo   = NUM_BITS_PRE,
        tau             = TAU,
        SNR_dB          = SNR_dB,
    )


def test_respuesta_temporal_correlador_vs_ml(
    ruta_checkpoint: str,
    carga_G: float = 0.4,
    ventana_frame_times: int = 300,
    snr_db: float = 6.0,
    tau_corr: float = 0.65,
    tau_ml: float = 0.5,
    temperature_ml: float = 1.0,
    semilla: int = 123,
    usar_preambulo_en_escenario: bool = True,
):
    """
    Dibuja dos paneles (correlador y CNN) sobre la misma realización.
    """
    if ruta_checkpoint is None or not os.path.exists(ruta_checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint_automatico(ruta_checkpoint, map_location=dispositivo)
    modelo.eval()

    # Misma realización para ambos receptores (si hay preámbulo).
    esc = generar_escenario_phy(
        carga_G=carga_G,
        ventana_frame_times=ventana_frame_times,
        snr_db=snr_db,
        num_bits_pre=NUM_BITS_PRE,
        num_bits_datos=NUM_BITS_DATOS,
        semilla=semilla,
        usar_preambulo=usar_preambulo_en_escenario,
    )

    if usar_preambulo_en_escenario:
        corr_norm = correlador(esc["senal_rx"], esc["preambulo"])
        det_corr = buscar_picos_preambulo(
            corr_norm=corr_norm,
            tau=tau_corr,
            separacion_minima=NUM_BITS_PRE,
        )
    else:
        # Sin preámbulo el correlador no es aplicable: se deja traza vacía para
        # mantener la misma función de visualización.
        corr_norm = np.zeros_like(esc["senal_rx"], dtype=float)
        det_corr = np.array([], dtype=np.int64)

    sal_ml = ejecutar_receptor_neuronal(
        escenario=esc,
        modelo=modelo,
        umbral=tau_ml,
        separacion_minima=NUM_BITS_PRE + NUM_BITS_DATOS,
        temperature=temperature_ml,
        dispositivo=dispositivo,
        stride=1,
        long_ventana=128,
    )

    ruta = os.path.join(FIGURES, "respuesta_temporal_correlador_vs_ml.png")
    plot_respuesta_correlador_vs_ml(
        corr_norm=corr_norm,
        score_ml=sal_ml["score_por_muestra"],
        instantes_reales=esc["instantes_llegada_muestras"],
        detecciones_corr=det_corr,
        detecciones_ml=sal_ml["instantes_detectados"],
        tau_corr=tau_corr,
        tau_ml=tau_ml,
        ruta_salida=ruta,
        titulo=(
            f"Respuesta temporal de detectores (G={carga_G}, SNR={snr_db} dB, "
            f"T={temperature_ml}, preambulo={'si' if usar_preambulo_en_escenario else 'no'})"
        ),
    )

    print("\nResumen detecciones en la misma realización:")
    print(f"  Escenario con preámbulo:      {usar_preambulo_en_escenario}")
    print(f"  Inicios reales (Pure ALOHA): {list(esc['instantes_llegada_muestras'])}")
    print(f"  Detecciones correlador:      {list(det_corr)}")
    print(f"  Detecciones red neuronal:    {list(sal_ml['instantes_detectados'])}")

    # Zoom automático alrededor del primer inicio real
    if len(esc["instantes_llegada_muestras"]) > 0:
        idx_centro = min(18, len(esc["instantes_llegada_muestras"]) - 1)
        centro = int(esc["instantes_llegada_muestras"][idx_centro])
        ruta_zoom = os.path.join(FIGURES, "zoom_respuesta_correlador_vs_ml.png")
        plot_zoom_respuesta_detectores(
            corr_norm=corr_norm,
            score_ml=sal_ml["score_por_muestra"],
            instantes_reales=esc["instantes_llegada_muestras"],
            detecciones_corr=det_corr,
            detecciones_ml=sal_ml["instantes_detectados"],
            centro_muestra=centro,
            ancho_ventana=250,
            tau_corr=tau_corr,
            tau_ml=tau_ml,
            ruta_salida=ruta_zoom,
        )
# ===========================================================================
# 3. EXPERIMENTO: RED NEURONAL (Futuro)
# ===========================================================================
def test_nn_decoder():
    print("\n=== Ejecutando Simulación: Red Neuronal ===")
    # Aquí irá el código de IA en el futuro
    pass


def prueba_integracion_total(carga_G, ventana_frame_times=400, snr_db=6.0, tau=0.65):
    """
    Integración motor ALOHA → canal → correlador; métricas frente a instantes verdaderos.

    La misma API de escenario (`generar_escenario_phy` en pipeline.escenario_phy) servirá
    para el receptor por red neuronal; aquí solo se ejecuta el correlador.
    """
    num_bits_pre = NUM_BITS_PRE
    num_bits_datos = NUM_BITS_DATOS
    tolerancia_muestras = TOLERANCIA_MUESTRAS
    num_iteraciones_mc = NUM_ITERACIONES_MC_RAPIDO
    semilla_base = SEMILLA_BASE

    resumen = ejecutar_monte_carlo_receptor_correlador(
        carga_G=carga_G,
        ventana_frame_times=ventana_frame_times,
        snr_db=snr_db,
        tau=tau,
        tolerancia_muestras=tolerancia_muestras,
        separacion_minima=num_bits_pre,
        num_iteraciones=num_iteraciones_mc,
        semilla_base=semilla_base,
        num_bits_pre=num_bits_pre,
        num_bits_datos=num_bits_datos,
    )

    print("=" * 60)
    print("PRUEBA INTEGRACION: escenario PHY compartido (ALOHA + AWGN)")
    print("=" * 60)
    print(f"  Carga G = {carga_G}")
    print(f"  Ventana (frame times) = {ventana_frame_times}")
    print(f"  SNR = {snr_db} dB  |  tau = {tau}  |  MC iteraciones = {num_iteraciones_mc}")
    print("-" * 60)
    print(f"  TP medio por iteracion:       {resumen['tp_media']:.3f}")
    print(f"  FP medio por iteracion:       {resumen['fp_media']:.3f}")
    print(f"  FN medio por iteracion:       {resumen['fn_media']:.3f}")
    print(f"  Paquetes (verdad) medio:      {resumen['paquetes_medio']:.3f}")
    print(f"  Detecciones totales medio:    {resumen['detecciones_medio']:.3f}")
    print("-" * 60)
    print("  Receptor RN: pendiente; usar el mismo dict de generar_escenario_phy.")
    print("=" * 60)

    # Futuro (misma señal, mismas métricas con evaluar_detecciones):
    # from pipeline.escenario_phy import generar_escenario_phy
    # esc = generar_escenario_phy(carga_G, ventana_frame_times, snr_db, semilla=...)
    # instantes_rn = receptor_red_neuronal(esc["senal_rx"])
    # metricas_rn = evaluar_detecciones(esc["instantes_llegada_muestras"], instantes_rn, ...)

    return resumen


def test_roc_correlador(carga_G=0.4, ventana_frame_times=400, snr_db=6.0):
    """
    ROC canónica del correlador por indice de correlación, barriendo tau en [0,1].
    """
    tolerancia_muestras = TOLERANCIA_MUESTRAS
    num_iteraciones_mc = NUM_ITERACIONES_MC_RAPIDO
    semilla_base = SEMILLA_BASE
    num_bits_pre = NUM_BITS_PRE
    num_bits_datos = NUM_BITS_DATOS

    roc = ejecutar_monte_carlo_roc_correlador(
        carga_G=carga_G,
        ventana_frame_times=ventana_frame_times,
        snr_db=snr_db,
        tolerancia_muestras=tolerancia_muestras,
        num_iteraciones=num_iteraciones_mc,
        semilla_base=semilla_base,
        num_bits_pre=num_bits_pre,
        num_bits_datos=num_bits_datos,
    )

    print("=" * 60)
    print("PRUEBA ROC: correlador por índice (tau en [0,1])")
    print("=" * 60)
    print(f"  Carga G = {carga_G}")
    print(f"  Ventana (frame times) = {ventana_frame_times}")
    print(f"  SNR = {snr_db} dB")
    print(f"  MC iteraciones = {num_iteraciones_mc}")
    print(f"  AUC media = {roc['auc_media']:.4f}")
    print("=" * 60)

    ruta_roc = os.path.join(FIGURES, "roc_correlador.png")
    plot_roc_correlador(
        fpr=roc["fpr_media"],
        tpr=roc["tpr_media"],
        auc=roc["auc_media"],
        ruta_salida=ruta_roc,
    )

    return roc


def test_correlador_escenario_objetivo(
    carga_G: float = 0.5,
    snr_db: float = 3.0,
    ventana_frame_times: int = 400,
    num_bits_pre: int = 13,
    num_bits_datos: int = 20,
    tolerancia_muestras: int = TOLERANCIA_MUESTRAS,
    semilla: int = 123,
):
    """
    Evaluación completa del correlador en un escenario objetivo:
      - Barrido de umbral tau en [0,1]
      - Curvas ROC/PR y sus AUC
      - Mejor umbral (max F1 por índice)
      - Matriz de confusión en ese mejor umbral

    Esta función está pensada para fijar métricas mínimas de referencia
    que luego deberá alcanzar (o aproximar) la red neuronal.
    """
    esc = generar_escenario_phy(
        carga_G=float(carga_G),
        ventana_frame_times=int(ventana_frame_times),
        snr_db=float(snr_db),
        num_bits_pre=int(num_bits_pre),
        num_bits_datos=int(num_bits_datos),
        semilla=int(semilla),
        usar_preambulo=True,
    )
    corr_norm = correlador(esc["senal_rx"], esc["preambulo"])
    taus = np.linspace(0.0, 1.0, 101, dtype=float)

    roc = curva_roc_por_indice(
        corr_norm=corr_norm,
        instantes_verdaderos=esc["instantes_llegada_muestras"],
        tolerancia_muestras=tolerancia_muestras,
        taus=taus,
    )
    pr = curva_pr_por_indice(
        corr_norm=corr_norm,
        instantes_verdaderos=esc["instantes_llegada_muestras"],
        tolerancia_muestras=tolerancia_muestras,
        taus=taus,
    )

    f1 = 2.0 * pr["precision"] * pr["recall"] / np.maximum(1e-12, pr["precision"] + pr["recall"])
    idx_best = int(np.nanargmax(f1)) if f1.size else 0
    tau_best = float(taus[idx_best]) if taus.size else 0.5

    tp = int(roc["tp"][idx_best])
    fp = int(roc["fp"][idx_best])
    tn = int(roc["tn"][idx_best])
    fn = int(roc["fn"][idx_best])
    precision_best = float(pr["precision"][idx_best])
    recall_best = float(pr["recall"][idx_best])
    f1_best = float(f1[idx_best])

    print("=" * 78)
    print("CORRELADOR — ESCENARIO OBJETIVO")
    print("=" * 78)
    print(
        f"G={carga_G}, SNR={snr_db} dB, paquete={num_bits_pre}+{num_bits_datos}="
        f"{num_bits_pre + num_bits_datos} muestras, tol={tolerancia_muestras}"
    )
    print(f"ROC-AUC={roc['auc']:.4f}  |  PR-AUC={pr['pr_auc']:.4f}")
    print(
        f"Mejor umbral (F1): tau={tau_best:.2f}  "
        f"P={precision_best:.4f}  R={recall_best:.4f}  F1={f1_best:.4f}"
    )
    print(f"Matriz confusión (mejor tau): TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("=" * 78)

    # Figura 1: curvas PR/ROC
    ruta_curvas = os.path.join(FIGURES, "correlador_escenario_objetivo_curvas.png")
    fig1, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(pr["recall"], pr["precision"], color="tab:green", lw=2)
    ax[0].set_title(f"PR (AUC={pr['pr_auc']:.3f})")
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(roc["fpr"], roc["tpr"], color="tab:blue", lw=2)
    ax[1].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax[1].set_title(f"ROC (AUC={roc['auc']:.3f})")
    ax[1].set_xlabel("FPR")
    ax[1].set_ylabel("TPR")
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].grid(True, alpha=0.3)
    fig1.suptitle(
        f"Correlador | G={carga_G}, SNR={snr_db} dB | paquete={num_bits_pre + num_bits_datos}"
    )
    fig1.tight_layout()
    fig1.savefig(ruta_curvas, dpi=180)
    plt.close(fig1)

    # Figura 2: matriz de confusión al mejor umbral
    ruta_cm = os.path.join(FIGURES, "correlador_escenario_objetivo_confusion.png")
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig2, ax2 = plt.subplots(figsize=(4.8, 4.4))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_title(f"Matriz de confusión (tau*={tau_best:.2f})")
    ax2.set_xlabel("Predicho")
    ax2.set_ylabel("Real")
    ax2.set_xticks([0, 1], labels=["0", "1"])
    ax2.set_yticks([0, 1], labels=["0", "1"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=11)
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(ruta_cm, dpi=180)
    plt.close(fig2)

    print(f"Curvas PR/ROC guardadas en: {ruta_curvas}")
    print(f"Matriz de confusión guardada en: {ruta_cm}")

    return {
        "G": float(carga_G),
        "SNR_dB": float(snr_db),
        "num_bits_pre": int(num_bits_pre),
        "num_bits_datos": int(num_bits_datos),
        "longitud_paquete": int(num_bits_pre + num_bits_datos),
        "tau_best_f1": tau_best,
        "precision_best": precision_best,
        "recall_best": recall_best,
        "f1_best": f1_best,
        "pr_auc": float(pr["pr_auc"]),
        "roc_auc": float(roc["auc"]),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "ruta_curvas": ruta_curvas,
        "ruta_confusion": ruta_cm,
    }


def test_roc_comparativa_correlador_vs_neuronal(
    ruta_checkpoint,
    carga_G=0.4,
    ventana_frame_times=400,
    snr_db=6.0,
    usar_modo_rapido=True,
):
    """
    Genera y dibuja en una sola figura la ROC del correlador y la ROC del
    detector neuronal bajo la misma condición (G,SNR) y el mismo protocolo.
    """
    if ruta_checkpoint is None or not os.path.exists(ruta_checkpoint):
        raise FileNotFoundError(
            "Debes indicar un checkpoint válido de la red neuronal en `ruta_checkpoint`."
        )

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint_automatico(ruta_checkpoint, map_location=dispositivo)
    modelo.eval()

    num_iter = NUM_ITERACIONES_MC_RAPIDO if usar_modo_rapido else NUM_ITERACIONES_MC
    taus = np.linspace(0.0, 1.0, 101, dtype=float)

    roc_corr = ejecutar_monte_carlo_roc_correlador(
        carga_G=carga_G,
        ventana_frame_times=ventana_frame_times,
        snr_db=snr_db,
        tolerancia_muestras=TOLERANCIA_MUESTRAS,
        num_iteraciones=num_iter,
        semilla_base=SEMILLA_BASE,
        num_bits_pre=NUM_BITS_PRE,
        num_bits_datos=NUM_BITS_DATOS,
        taus=taus,
    )

    roc_ml = ejecutar_monte_carlo_roc_neuronal(
        carga_G=carga_G,
        ventana_frame_times=ventana_frame_times,
        snr_db=snr_db,
        tolerancia_muestras=TOLERANCIA_MUESTRAS,
        num_iteraciones=num_iter,
        modelo=modelo,
        semilla_base=SEMILLA_BASE,
        num_bits_pre=NUM_BITS_PRE,
        num_bits_datos=NUM_BITS_DATOS,
        dispositivo=dispositivo,
        stride=1,
        long_ventana=128,
        taus=taus,
    )

    print("=" * 70)
    print("ROC COMPARATIVA — Correlador vs Red Neuronal")
    print("=" * 70)
    print(f"G={carga_G}, SNR={snr_db} dB, MC={num_iter}")
    print(f"AUC correlador:    {roc_corr['auc_media']:.4f}")
    print(f"AUC red neuronal:  {roc_ml['auc_media']:.4f}")
    print("=" * 70)

    ruta = os.path.join(FIGURES, "roc_comparativa_correlador_vs_ml.png")
    plot_roc_comparativa_correlador_vs_ml(
        fpr_corr=roc_corr["fpr_media"],
        tpr_corr=roc_corr["tpr_media"],
        auc_corr=roc_corr["auc_media"],
        fpr_ml=roc_ml["fpr_media"],
        tpr_ml=roc_ml["tpr_media"],
        auc_ml=roc_ml["auc_media"],
        ruta_salida=ruta,
        carga_G=carga_G,
        snr_db=snr_db,
    )

    return {"roc_correlador": roc_corr, "roc_ml": roc_ml}


def test_protocolo_comun_correlador(
    ventana_frame_times=400,
    tau_evento=0.65,
    usar_modo_rapido=False,
):
    """
    Ejecuta el protocolo común congelado en grid (G,SNR) y reporta tabla + ROC.
    """
    num_iter = NUM_ITERACIONES_MC_RAPIDO if usar_modo_rapido else NUM_ITERACIONES_MC
    filas = barrer_grid_protocolo_correlador(
        cargas_G=GRID_CARGA_G,
        snrs_db=GRID_SNR_DB,
        ventana_frame_times=ventana_frame_times,
        tau_evento=tau_evento,
        tolerancia_muestras=TOLERANCIA_MUESTRAS,
        separacion_minima=NUM_BITS_PRE,
        num_iteraciones=num_iter,
        semilla_base=SEMILLA_BASE,
        num_bits_pre=NUM_BITS_PRE,
        num_bits_datos=NUM_BITS_DATOS,
    )

    print("=" * 108)
    print("PROTOCOLO COMÚN (CONGELADO) — Tabla por (G,SNR)")
    print("=" * 108)
    print(
        "  G   SNR  TP_media±std   FP_media±std   FN_media±std   Recall   Precision   F1     AUC"
    )
    print("-" * 108)
    for f in filas:
        print(
            f"{f['G']:>3.1f} {f['SNR_dB']:>5.1f} "
            f"{f['tp_media']:>6.2f}±{f['tp_std']:<5.2f} "
            f"{f['fp_media']:>6.2f}±{f['fp_std']:<5.2f} "
            f"{f['fn_media']:>6.2f}±{f['fn_std']:<5.2f} "
            f"{f['recall']:>7.3f}   {f['precision']:>8.3f}  {f['f1']:>6.3f}  {f['auc']:>6.3f}"
        )
    print("=" * 108)

    # Figura obligatoria: familia ROC por SNR a G fijo (primer G del grid).
    g_ref = float(GRID_CARGA_G[0])
    curvas_por_snr = {}
    for snr in GRID_SNR_DB:
        roc = ejecutar_monte_carlo_roc_correlador(
            carga_G=g_ref,
            ventana_frame_times=ventana_frame_times,
            snr_db=float(snr),
            tolerancia_muestras=TOLERANCIA_MUESTRAS,
            num_iteraciones=num_iter,
            semilla_base=SEMILLA_BASE,
            num_bits_pre=NUM_BITS_PRE,
            num_bits_datos=NUM_BITS_DATOS,
        )
        curvas_por_snr[float(snr)] = {
            "fpr": roc["fpr_media"],
            "tpr": roc["tpr_media"],
            "auc": roc["auc_media"],
        }
    ruta_familia = os.path.join(FIGURES, "roc_familia_por_snr.png")
    plot_roc_familia_por_snr(
        curvas_por_snr=curvas_por_snr,
        ruta_salida=ruta_familia,
        carga_G=g_ref,
    )

    return filas


def test_protocolo_comun_neuronal(
    ruta_checkpoint: str | None = None,
    ventana_frame_times: int = 400,
    umbral: float = 0.5,
    usar_modo_rapido: bool = True,
):
    """
    Ejecuta el protocolo común congelado sobre el detector ML (CNN 1D) en el
    mismo grid (G, SNR) que el correlador, permitiendo comparativa directa.

    Si se proporciona `ruta_checkpoint`, carga los pesos entrenados.
    Si no, usa el modelo sin entrenar (aleatorio) como prueba del pipeline.

    Parámetros
    ----------
    ruta_checkpoint   : ruta al archivo .ckpt generado por entrenar_modelo.py,
                        o None para ejecutar con pesos aleatorios (demo).
    ventana_frame_times: horizonte temporal de cada escenario.
    umbral            : umbral de probabilidad [0,1] para NMS del detector ML.
    usar_modo_rapido  : si True, usa NUM_ITERACIONES_MC_RAPIDO.
    """
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    separacion_minima = NUM_BITS_PRE   # misma que el correlador

    if ruta_checkpoint is not None and os.path.exists(ruta_checkpoint):
        modelo = cargar_checkpoint_automatico(ruta_checkpoint, map_location=dispositivo)
        print(f"[ML] Modelo cargado desde: {ruta_checkpoint}")
    else:
        modelo = ModeloCNN()
        modelo.eval()
        if ruta_checkpoint is not None:
            print(f"[ML] ADVERTENCIA: checkpoint no encontrado en '{ruta_checkpoint}'.")
        print("[ML] Usando modelo con pesos aleatorios (sin entrenar). "
              "Ejecuta ml/entrenar_modelo.py para obtener un modelo entrenado.")

    num_iter = NUM_ITERACIONES_MC_RAPIDO if usar_modo_rapido else NUM_ITERACIONES_MC

    filas = []
    for g in GRID_CARGA_G:
        for snr in GRID_SNR_DB:
            tp_hist, fp_hist, fn_hist = [], [], []
            for k in range(num_iter):
                esc = generar_escenario_phy(
                    carga_G=float(g),
                    ventana_frame_times=ventana_frame_times,
                    snr_db=float(snr),
                    num_bits_pre=NUM_BITS_PRE,
                    num_bits_datos=NUM_BITS_DATOS,
                    semilla=SEMILLA_BASE + k,
                )
                sal_ml = ejecutar_receptor_neuronal(
                    escenario=esc,
                    modelo=modelo,
                    umbral=umbral,
                    separacion_minima=separacion_minima,
                    dispositivo=dispositivo,
                )
                metricas = evaluar_detecciones(
                    esc["instantes_llegada_muestras"],
                    sal_ml["instantes_detectados"],
                    TOLERANCIA_MUESTRAS,
                )
                tp_hist.append(metricas["tp"])
                fp_hist.append(metricas["fp"])
                fn_hist.append(metricas["fn"])

            tp_m = float(np.mean(tp_hist))
            fp_m = float(np.mean(fp_hist))
            fn_m = float(np.mean(fn_hist))
            tp_std = float(np.std(tp_hist, ddof=0))
            fp_std = float(np.std(fp_hist, ddof=0))
            fn_std = float(np.std(fn_hist, ddof=0))
            derivadas = metricas_evento_derivadas(tp_m, fp_m, fn_m)

            filas.append({
                "G": float(g),
                "SNR_dB": float(snr),
                "tp_media": tp_m, "tp_std": tp_std,
                "fp_media": fp_m, "fp_std": fp_std,
                "fn_media": fn_m, "fn_std": fn_std,
                "recall": derivadas["recall"],
                "precision": derivadas["precision"],
                "f1": derivadas["f1"],
            })

    print("=" * 108)
    print("PROTOCOLO COMÚN (CONGELADO) — Detector ML (CNN 1D)")
    print("=" * 108)
    print(
        "  G   SNR  TP_media±std   FP_media±std   FN_media±std   Recall   Precision   F1"
    )
    print("-" * 108)
    for f in filas:
        print(
            f"{f['G']:>3.1f} {f['SNR_dB']:>5.1f} "
            f"{f['tp_media']:>6.2f}±{f['tp_std']:<5.2f} "
            f"{f['fp_media']:>6.2f}±{f['fp_std']:<5.2f} "
            f"{f['fn_media']:>6.2f}±{f['fn_std']:<5.2f} "
            f"{f['recall']:>7.3f}   {f['precision']:>8.3f}  {f['f1']:>6.3f}"
        )
    print("=" * 108)

    return filas


def test_ablacion_compacta(
    checkpoints: dict,
    carga_G: float = 0.5,
    snr_db: float = 3.0,
    ventana_frame_times: int = 400,
    semillas: tuple = (10, 20, 30, 40, 50),
):
    """
    Ablación compacta: evalúa múltiples checkpoints en el escenario objetivo
    con varias semillas y reporta PR-AUC, F1_best y tau_best (media ± std).

    Parámetros
    ----------
    checkpoints : dict
        Clave: etiqueta legible (p.ej. "iq_legacy / sin_HN / sin_PW")
        Valor: ruta al .ckpt
    carga_G, snr_db : escenario objetivo (por defecto el realista G=0.5, SNR=3)
    semillas : tupla de semillas para evaluar (5 por defecto)

    Genera:
      - Tabla en consola con media ± std
      - Figura de barras comparativa (PR-AUC y F1_best por combinación)
      - CSV en results/ablacion_compacta.csv
    """
    import csv

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    taus = np.linspace(0.0, 1.0, 101, dtype=float)
    filas = []

    print("=" * 90)
    print("ABLACIÓN COMPACTA")
    print(f"Escenario: G={carga_G}, SNR={snr_db} dB | {len(semillas)} semillas")
    print("=" * 90)
    print(f"{'Combinación':<40} {'PR-AUC':>12} {'F1_best':>12} {'tau_best':>10}")
    print("-" * 90)

    for label, ruta in checkpoints.items():
        if not os.path.exists(ruta):
            print(f"  ⚠ Checkpoint no encontrado: {ruta}")
            continue

        modelo = cargar_checkpoint_automatico(ruta, map_location=dispositivo)
        pr_aucs, f1s, tau_bests = [], [], []

        for semilla in semillas:
            esc = generar_escenario_phy(
                carga_G=float(carga_G),
                ventana_frame_times=int(ventana_frame_times),
                snr_db=float(snr_db),
                num_bits_pre=NUM_BITS_PRE,
                num_bits_datos=NUM_BITS_DATOS,
                semilla=int(semilla),
                usar_preambulo=False,
            )
            sal = ejecutar_receptor_neuronal(
                escenario=esc,
                modelo=modelo,
                umbral=0.5,
                separacion_minima=NUM_BITS_PRE + NUM_BITS_DATOS,
                dispositivo=dispositivo,
                stride=1,
                long_ventana=128,
            )
            score = np.asarray(sal["score_por_muestra"], dtype=float)

            pr = curva_pr_por_indice(
                corr_norm=score,
                instantes_verdaderos=esc["instantes_llegada_muestras"],
                tolerancia_muestras=TOLERANCIA_MUESTRAS,
                taus=taus,
            )
            f1_arr = (
                2.0 * pr["precision"] * pr["recall"]
                / np.maximum(1e-12, pr["precision"] + pr["recall"])
            )
            idx_best = int(np.nanargmax(f1_arr)) if f1_arr.size else 0

            pr_aucs.append(float(pr["pr_auc"]))
            f1s.append(float(f1_arr[idx_best]))
            tau_bests.append(float(taus[idx_best]))

        pr_auc_m, pr_auc_s = float(np.mean(pr_aucs)), float(np.std(pr_aucs))
        f1_m, f1_s = float(np.mean(f1s)), float(np.std(f1s))
        tau_m, tau_s = float(np.mean(tau_bests)), float(np.std(tau_bests))

        print(
            f"  {label:<38} "
            f"{pr_auc_m:.3f}±{pr_auc_s:.3f}  "
            f"{f1_m:.3f}±{f1_s:.3f}  "
            f"{tau_m:.2f}±{tau_s:.2f}"
        )
        filas.append({
            "combinacion": label,
            "pr_auc_media": pr_auc_m, "pr_auc_std": pr_auc_s,
            "f1_media": f1_m, "f1_std": f1_s,
            "tau_media": tau_m, "tau_std": tau_s,
        })

    print("=" * 90)

    # CSV
    ruta_csv = os.path.join(ROOT, "results", "ablacion_compacta.csv")
    if filas:
        with open(ruta_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            writer.writeheader()
            writer.writerows(filas)
        print(f"CSV guardado: {ruta_csv}")

    # Figura de barras
    if filas:
        labels = [f["combinacion"] for f in filas]
        pr_medias = [f["pr_auc_media"] for f in filas]
        pr_stds = [f["pr_auc_std"] for f in filas]
        f1_medias = [f["f1_media"] for f in filas]
        f1_stds = [f["f1_std"] for f in filas]
        x = np.arange(len(labels))
        w = 0.38

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 5.5))
        ax.bar(x - w / 2, pr_medias, w, yerr=pr_stds, label="PR-AUC",
               color="tab:green", alpha=0.85, capsize=4)
        ax.bar(x + w / 2, f1_medias, w, yerr=f1_stds, label="F1_best",
               color="tab:blue", alpha=0.85, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Métrica")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Ablación compacta — G={carga_G}, SNR={snr_db} dB")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        ruta_fig = os.path.join(FIGURES, "ablacion_compacta.png")
        fig.savefig(ruta_fig, dpi=180)
        plt.close(fig)
        print(f"Figura guardada: {ruta_fig}")

    return filas


def test_diagnostico_profesor(
    ruta_checkpoint: str,
    tau: float = 0.5,
    temperature: float = 1.0,
    semilla: int = 42,
):
    """
    Diagnóstico progresivo sugerido por el tutor:
      1. Sin colisiones, sin ruido  (G bajo, SNR muy alto)
      2. Sin colisiones, con ruido  (G bajo, SNR=6 dB)
      3. Con colisiones normales    (G=0.4,  SNR=6 dB)

    Evalúa el modelo ya entrenado sin necesidad de reentrenar,
    y genera por escenario:
      - Métricas a umbral fijo
      - Curva PR + PR-AUC (barrido de umbral)
      - Curva ROC + ROC-AUC (barrido de umbral)
      - Figura zoom local de la respuesta temporal
      - Figura PR/ROC

    La fase aleatoria por paquete ya está activa en generar_escenario_phy
    por defecto (aplicar_fase_aleatoria_por_paquete=True).
    """
    if not os.path.exists(ruta_checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {ruta_checkpoint}")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint_automatico(ruta_checkpoint, map_location=dispositivo)

    escenarios = [
        {"label": "1_sin_colision_sin_ruido",  "G": 0.1, "SNR": 50.0},
        {"label": "2_sin_colision_con_ruido",  "G": 0.1, "SNR":  6.0},
        {"label": "3_con_colision_con_ruido",  "G": 0.4, "SNR":  6.0},
    ]

    print("=" * 70)
    print("DIAGNÓSTICO TUTOR — escenarios progresivos")
    print(f"  Checkpoint : {ruta_checkpoint}")
    print(f"  Umbral     : {tau}  |  Temperatura: {temperature}")
    print(f"  NOTA: fase aleatoria por paquete activa por defecto")
    print("=" * 70)

    resultados = []
    taus = np.linspace(0.0, 1.0, 101, dtype=float)

    for cfg in escenarios:
        esc = generar_escenario_phy(
            carga_G=cfg["G"],
            ventana_frame_times=300,
            snr_db=cfg["SNR"],
            num_bits_pre=NUM_BITS_PRE,
            num_bits_datos=NUM_BITS_DATOS,
            semilla=semilla,
            usar_preambulo=False,
        )
        sal_ml = ejecutar_receptor_neuronal(
            escenario=esc,
            modelo=modelo,
            umbral=tau,
            separacion_minima=NUM_BITS_PRE + NUM_BITS_DATOS,
            temperature=temperature,
            dispositivo=dispositivo,
            stride=1,
            long_ventana=128,
        )
        met = evaluar_detecciones(
            esc["instantes_llegada_muestras"],
            sal_ml["instantes_detectados"],
            TOLERANCIA_MUESTRAS,
        )
        score_max = float(sal_ml["score_por_muestra"].max())
        score = np.asarray(sal_ml["score_por_muestra"], dtype=float)

        roc = curva_roc_por_indice(
            corr_norm=score,
            instantes_verdaderos=esc["instantes_llegada_muestras"],
            tolerancia_muestras=TOLERANCIA_MUESTRAS,
            taus=taus,
        )
        pr = curva_pr_por_indice(
            corr_norm=score,
            instantes_verdaderos=esc["instantes_llegada_muestras"],
            tolerancia_muestras=TOLERANCIA_MUESTRAS,
            taus=taus,
        )
        f1 = 2.0 * pr["precision"] * pr["recall"] / np.maximum(1e-12, pr["precision"] + pr["recall"])
        idx_best = int(np.nanargmax(f1)) if f1.size else 0
        tau_best = float(taus[idx_best]) if taus.size else float(tau)
        f1_best = float(f1[idx_best]) if f1.size else 0.0

        print(f"\n  [{cfg['label']}]  G={cfg['G']}  SNR={cfg['SNR']} dB")
        print(f"    Paquetes reales: {met['num_verdaderos']}  |  Detecciones: {met['num_detectados']}")
        print(f"    TP={met['tp']}  FP={met['fp']}  FN={met['fn']}  |  Score máx: {score_max:.4f}")
        print(f"    PR-AUC={pr['pr_auc']:.4f}  |  ROC-AUC={roc['auc']:.4f}")
        print(
            f"    Mejor F1 por barrido: F1={f1_best:.4f}  "
            f"(tau={tau_best:.2f}, P={pr['precision'][idx_best]:.4f}, R={pr['recall'][idx_best]:.4f})"
        )

        if len(esc["instantes_llegada_muestras"]) > 0:
            centro = int(esc["instantes_llegada_muestras"][0])
            ruta_fig = os.path.join(FIGURES, f"diagnostico_{cfg['label']}.png")
            plot_zoom_respuesta_detectores(
                corr_norm=np.zeros_like(sal_ml["score_por_muestra"]),
                score_ml=sal_ml["score_por_muestra"],
                instantes_reales=esc["instantes_llegada_muestras"],
                detecciones_corr=np.array([], dtype=np.int64),
                detecciones_ml=sal_ml["instantes_detectados"],
                centro_muestra=centro,
                ancho_ventana=250,
                tau_corr=1.1,   # fuera de rango: oculta la línea del correlador
                tau_ml=tau,
                ruta_salida=ruta_fig,
            )
            print(f"    Figura: {ruta_fig}")

        ruta_curvas = os.path.join(FIGURES, f"diagnostico_curvas_{cfg['label']}.png")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(pr["recall"], pr["precision"], color="tab:green", lw=2)
        ax[0].set_title(f"PR (AUC={pr['pr_auc']:.3f})")
        ax[0].set_xlabel("Recall")
        ax[0].set_ylabel("Precision")
        ax[0].grid(True, alpha=0.3)
        ax[0].set_xlim(0.0, 1.0)
        ax[0].set_ylim(0.0, 1.0)

        ax[1].plot(roc["fpr"], roc["tpr"], color="tab:blue", lw=2)
        ax[1].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax[1].set_title(f"ROC (AUC={roc['auc']:.3f})")
        ax[1].set_xlabel("FPR")
        ax[1].set_ylabel("TPR")
        ax[1].grid(True, alpha=0.3)
        ax[1].set_xlim(0.0, 1.0)
        ax[1].set_ylim(0.0, 1.0)
        fig.suptitle(f"Diagnóstico {cfg['label']} | G={cfg['G']}, SNR={cfg['SNR']} dB")
        fig.tight_layout()
        fig.savefig(ruta_curvas, dpi=180)
        plt.close(fig)
        print(f"    Curvas PR/ROC: {ruta_curvas}")

        precision_fijo = float(met["tp"]) / max(1.0, float(met["tp"] + met["fp"]))
        recall_fijo = float(met["tp"]) / max(1.0, float(met["tp"] + met["fn"]))
        f1_fijo = (
            2.0 * precision_fijo * recall_fijo / max(1e-12, precision_fijo + recall_fijo)
        )

        resultados.append(
            {
                "label": cfg["label"],
                "G": float(cfg["G"]),
                "SNR_dB": float(cfg["SNR"]),
                "tp": int(met["tp"]),
                "fp": int(met["fp"]),
                "fn": int(met["fn"]),
                "precision_tau_fijo": precision_fijo,
                "recall_tau_fijo": recall_fijo,
                "f1_tau_fijo": f1_fijo,
                "pr_auc": float(pr["pr_auc"]),
                "roc_auc": float(roc["auc"]),
                "tau_best_f1": tau_best,
                "f1_best": f1_best,
            }
        )

    print("\n" + "=" * 70)
    return resultados


# ── ejecución ──────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # 1. Simulación de red
    #test_traffic_model_aloha()

    # 2. Simulación física: Fase 1 (Ideal, sin ruido)
    #test_phy_correlador(snr_db=100, num_bits_pre = 13, num_bits_data = 20)
    #print("=== Test colisión PHY ===")
    #test_colision_phy()
    # 3. Simulación física: Fase 2 (Con ruido)
    # test_phy_correlador(snr_db=2, num_bits_pre = 13, num_bits_data = 20)

    # 3b. Integración ALOHA → PHY → correlador (Monte Carlo)
    #prueba_integracion_total(carga_G=0.4)
    # ROC canónica (por índice) para todos los umbrales tau en [0,1]
    # test_roc_correlador(carga_G=0.4, ventana_frame_times=400, snr_db=6.0)
    #  test_roc_comparativa_correlador_vs_neuronal(
    #     ruta_checkpoint="checkpoints/mejor-epoch=17-val_loss=0.0210.ckpt",
    #     carga_G=0.4,
    #     ventana_frame_times=400,
    #     snr_db=6.0,
    #     usar_modo_rapido=True,
    # ) 
    # Protocolo común congelado (tabla + familia ROC); usar_modo_rapido=True para iterar rápido
    # test_protocolo_comun_correlador(ventana_frame_times=400, tau_evento=0.65, usar_modo_rapido=True)
    # test_phy_correlador(snr_db=-5) # Prueba a poner SNR negativo para ver cómo falla

    # 4. Fase de IA: detector ML (CNN 1D)
    # Para entrenar el modelo:  python -m ml.generar_dataset --salida data/dataset_aloha
    #                           python -m ml.entrenar_modelo --datos data/dataset_aloha --sin_wandb
    # Para evaluar (sin checkpoint, pesos aleatorios):
    # test_protocolo_comun_neuronal(usar_modo_rapido=True)
    # Para evaluar con el modelo entrenado:
    # test_protocolo_comun_neuronal(ruta_checkpoint='checkpoints/mejor-epoch=17-val_loss=0.0210.ckpt', usar_modo_rapido=True)
    
    # test_respuesta_temporal_correlador_vs_ml(
    #     ruta_checkpoint="checkpoints/v3_5050_HN8-epoch=11-val_loss=0.4538.ckpt",
    #     carga_G=0.1,
    #     ventana_frame_times=300,
    #     snr_db=6.0,
    #     tau_corr=0.7,
    #     tau_ml=0.7,
    #     temperature_ml=1.0,
    #     semilla=123,
    # )

     test_diagnostico_profesor(
        ruta_checkpoint="checkpoints/mejor-epoch=61-val_loss=0.5041.ckpt",
        tau=0.5,
        temperature=1.0,
        semilla=123,
    )
    # test_correlador_escenario_objetivo(
    #     carga_G=0.5,
    #     snr_db=3.0,
    #     ventana_frame_times=400,
    #     num_bits_pre=23,
    #     num_bits_datos=105,   # ajusta aquí tu longitud de paquete
    #     tolerancia_muestras=0,
    #     semilla=123,
    # )

