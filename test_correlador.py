"""
Pipeline del Correlador (Matched Filter) — Capa Física (PHY)
=============================================================
Bloque 1: Transmisor (Tx) — genera preámbulo BPSK
Bloque 2: Canal AWGN      — retraso aleatorio + ruido gaussiano
Bloque 3: Receptor (Rx)   — correlador de ventana deslizante normalizado
Bloque 4: Visualización   — señal recibida + salida del correlador

Diseñado para integrarse con la simulación ALOHA (MAC layer):
  - canal_awgn() acepta instantes de llegada y soporta colisiones (suma de señales)
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# BLOQUE 1 — TRANSMISOR: genera preámbulo BPSK
# =============================================================================

def generar_preambulo(num_bits=13, semilla=42):
    """
    Genera un preámbulo conocido modulado en BPSK.

    Parámetros
    ----------
    num_bits : longitud del preámbulo en símbolos (recomendado: primo, ej. 13)
    semilla  : semilla fija para que el preámbulo sea siempre el mismo

    Retorna
    -------
    preambulo : array de +1/-1 de longitud num_bits

    Nota: transmisor y receptor usan la misma semilla para conocer la secuencia.
    """
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=num_bits)   # bits aleatorios 0/1
    preambulo = 2 * bits - 1                    # BPSK: 0 → -1, 1 → +1
    return preambulo


def generar_paquete(preambulo, num_bits_datos=20):
    """
    Construye un paquete completo: [preámbulo | datos aleatorios].

    Parámetros
    ----------
    preambulo      : array BPSK del preámbulo (conocido)
    num_bits_datos : longitud de la parte de datos (desconocida para el receptor)

    Retorna
    -------
    paquete : array [preámbulo | datos] en BPSK
    """
    datos = np.random.choice([-1, 1], size=num_bits_datos)
    paquete = np.concatenate([preambulo, datos])
    return paquete


# =============================================================================
# BLOQUE 2 — CANAL: retraso aleatorio + ruido AWGN
# =============================================================================

def canal_awgn(paquete, SNR_dB, longitud_total=None, instante_llegada=None):
    """
    Simula el canal físico: coloca el paquete en el tiempo y añade ruido AWGN.

    Parámetros
    ----------
    paquete          : señal BPSK a transmitir (preámbulo + datos)
    SNR_dB           : relación señal-ruido en dB
    longitud_total   : duración total de la señal observada (en muestras).
                       Si es None, se calcula automáticamente con margen.
    instante_llegada : muestra en la que empieza el paquete.
                       Si es None, se elige aleatoriamente.

    Retorna
    -------
    señal_rx         : señal recibida (ruido + paquete en la posición correcta)
    instante_llegada : instante real de llegada (útil para evaluar detección)

    Nota de integración con ALOHA:
    --------------------------------
    Para simular colisiones, llama a esta función dos veces con distintos
    instantes_llegada y suma las señales_rx resultantes antes de correlacionar:

        rx1 = canal_awgn(paquete1, SNR_dB, longitud_total, t1)[0]
        rx2 = canal_awgn(paquete2, SNR_dB, longitud_total, t2)[0]
        señal_con_colision = rx1 + rx2   # colisión: ambas en el canal
    """
    L = len(paquete)

    if longitud_total is None:
        longitud_total = 5 * L          # margen generoso a ambos lados

    if instante_llegada is None:
        # el paquete cabe entero dentro del intervalo
        instante_llegada = np.random.randint(L, longitud_total - L)

    # --- señal limpia: ceros salvo donde está el paquete ---
    señal_limpia = np.zeros(longitud_total)
    fin = instante_llegada + L
    señal_limpia[instante_llegada:fin] = paquete

    # --- potencia de la señal (solo en las muestras del paquete) ---
    potencia_señal = np.mean(paquete ** 2)          # = 1.0 para BPSK puro

    # --- varianza del ruido a partir de la SNR ---
    # SNR_lineal = potencia_señal / varianza_ruido
    # varianza_ruido = potencia_señal / SNR_lineal
    SNR_lineal = 10 ** (SNR_dB / 10)
    varianza_ruido = potencia_señal / SNR_lineal
    sigma = np.sqrt(varianza_ruido)

    # --- ruido gaussiano sobre toda la señal ---
    ruido = np.random.normal(0, sigma, longitud_total)
    señal_rx = señal_limpia + ruido

    return señal_rx, instante_llegada


# =============================================================================
# BLOQUE 3 — RECEPTOR: correlador de ventana deslizante normalizado
# =============================================================================

def correlador(señal_rx, preambulo):
    """
    Correlación cruzada normalizada entre la señal recibida y el preámbulo.

    En un caso ideal (sin ruido, paquete alineado), el pico máximo vale 1.0.
    En presencia de ruido el pico baja, y el suelo de correlación sube.

    Parámetros
    ----------
    señal_rx  : señal recibida del canal (ruidosa)
    preambulo : plantilla del preámbulo conocida por el receptor

    Retorna
    -------
    corr_norm : array con la correlación normalizada (valores entre ~0 y ~1)

    Uso del umbral:
        detecciones = np.where(corr_norm >= tau)[0]
    """
    L = len(preambulo)

    # np.correlate en modo 'full' devuelve longitud len(señal) + len(preámbulo) - 1
    # usamos 'valid' para obtener solo las posiciones donde el preámbulo cabe entero
    corr = np.correlate(señal_rx, preambulo, mode='valid')

    # Normalización: divide por L para que el pico ideal sea exactamente 1
    corr_norm = np.abs(corr) / L

    return corr_norm


# =============================================================================
# BLOQUE 4 — VISUALIZACIÓN Y PRUEBA
# =============================================================================

def visualizar_resultado(señal_rx, corr_norm, instante_llegada,
                         len_preambulo, tau=0.7, SNR_dB=None):
    """
    Genera la figura de dos subplots:
      - Arriba : señal recibida en el tiempo (solo se ve ruido)
      - Abajo  : salida del correlador con pico de detección y umbral τ
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    titulo = f"Pipeline del Correlador — SNR = {SNR_dB} dB" if SNR_dB else \
             "Pipeline del Correlador"
    fig.suptitle(titulo, fontsize=13)

    # --- subplot 1: señal recibida ---
    ax1.plot(señal_rx, color='steelblue', linewidth=0.7, alpha=0.85)
    ax1.axvline(x=instante_llegada, color='green', linestyle='--',
                linewidth=1, label=f'Inicio paquete (muestra {instante_llegada})')
    ax1.axvline(x=instante_llegada + len_preambulo, color='orange',
                linestyle='--', linewidth=1, label='Fin preámbulo')
    ax1.set_ylabel('Amplitud')
    ax1.set_title('Señal recibida — el paquete queda oculto en el ruido')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- subplot 2: salida del correlador ---
    muestras_corr = np.arange(len(corr_norm))
    ax2.plot(muestras_corr, corr_norm, color='coral', linewidth=0.9,
             alpha=0.9, label='Correlación normalizada')

    # umbral de detección
    ax2.axhline(y=tau, color='red', linestyle=':', linewidth=1.5,
                label=f'Umbral τ = {tau}')

    # pico esperado (posición teórica)
    ax2.axvline(x=instante_llegada, color='green', linestyle='--',
                linewidth=1, label=f'Pico esperado (muestra {instante_llegada})')

    # pico real detectado
    pico_idx = np.argmax(corr_norm)
    pico_val = corr_norm[pico_idx]
    ax2.plot(pico_idx, pico_val, 'r*', markersize=12,
             label=f'Pico detectado: {pico_val:.3f} (muestra {pico_idx})')

    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('Magnitud normalizada')
    ax2.set_xlabel('Muestras')
    ax2.set_title('Salida del correlador — el pico sobresale del suelo de ruido')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/correlador_resultado.png', dpi=150)
    plt.show()
    print("Gráfica guardada en results/figures/correlador_resultado.png")


# =============================================================================
# MAIN — ejecución de prueba con un solo paquete
# =============================================================================

def main():
    # --- parámetros ---
    SNR_dB        = 2        # SNR baja: el paquete queda enterrado en ruido
    NUM_BITS_PRE  = 13       # longitud del preámbulo
    NUM_BITS_DATA = 20       # longitud de los datos
    TAU           = 0.7      # umbral de detección

    np.random.seed(0)        # reproducibilidad

    # Bloque 1 — Transmisor
    preambulo = generar_preambulo(num_bits=NUM_BITS_PRE)
    paquete   = generar_paquete(preambulo, num_bits_datos=NUM_BITS_DATA)

    print(f"Preámbulo ({NUM_BITS_PRE} símbolos): {preambulo}")
    print(f"Paquete total: {len(paquete)} símbolos "
          f"({NUM_BITS_PRE} preámbulo + {NUM_BITS_DATA} datos)")

    # Bloque 2 — Canal
    señal_rx, instante = canal_awgn(paquete, SNR_dB=SNR_dB)

    print(f"\nInstante de llegada real: muestra {instante}")
    print(f"Longitud señal recibida:  {len(señal_rx)} muestras")
    print(f"SNR = {SNR_dB} dB  →  el paquete queda oculto en el ruido")

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
    visualizar_resultado(señal_rx, corr_norm, instante,
                         NUM_BITS_PRE, tau=TAU, SNR_dB=SNR_dB)


if __name__ == '__main__':
    main()
