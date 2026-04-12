
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
from pipeline.visualization import plot_pure_aloha,  plot_deteccion_correlador_awgn, plot_colision_correlador


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

# ── ejecución ──────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # 1. Simulación de red
    #test_traffic_model_aloha()

    # 2. Simulación física: Fase 1 (Ideal, sin ruido)
    #test_phy_correlador(snr_db=100, num_bits_pre = 13, num_bits_data = 20)
    #print("=== Test colisión PHY ===")
    #test_colision_phy()
    # 3. Simulación física: Fase 2 (Con ruido)
    test_phy_correlador(snr_db=2, num_bits_pre = 13, num_bits_data = 20)
    # test_phy_correlador(snr_db=-5) # Prueba a poner SNR negativo para ver cómo falla

    # 4. Fase de IA (Pendiente)
    # test_nn_decoder()