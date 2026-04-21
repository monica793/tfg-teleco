
import os
import numpy as np

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
from pipeline.correlator_decoder import correlador
from pipeline.escenario_phy import (
    barrer_grid_protocolo_correlador,
    ejecutar_monte_carlo_receptor_correlador,
    ejecutar_monte_carlo_roc_correlador,
)
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
    test_roc_correlador(carga_G=0.4, ventana_frame_times=400, snr_db=6.0)
    # Protocolo común congelado (tabla + familia ROC); usar_modo_rapido=True para iterar rápido
    # test_protocolo_comun_correlador(ventana_frame_times=400, tau_evento=0.65, usar_modo_rapido=True)
    # test_phy_correlador(snr_db=-5) # Prueba a poner SNR negativo para ver cómo falla

    # 4. Fase de IA (Pendiente)
    # test_nn_decoder()