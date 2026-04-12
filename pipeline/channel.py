import numpy as np

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
 
 # ============================================================
# AÑADIR a pipeline/channel.py
# ============================================================

def canal_awgn_colision(paquetes_y_tiempos, SNR_dB, longitud_total):
    """
    Canal AWGN con solapamiento de múltiples paquetes (colisión PHY).

    Coloca cada paquete en su instante de llegada dentro de un array de
    longitud_total, suma todas las señales (la interferencia aparece
    naturalmente donde se solapan) y añade ruido AWGN una única vez
    sobre la señal compuesta.

    Parámetros
    ----------
    paquetes_y_tiempos : list of (array, int)
        Lista de tuplas (paquete_bpsk, instante_llegada).
        instante_llegada es la muestra donde empieza ese paquete.
    SNR_dB             : float
        Relación señal-ruido en dB. Se calcula respecto a la potencia
        media de UN paquete individual (no de la suma).
    longitud_total     : int
        Duración total de la ventana de observación en muestras.

    Retorna
    -------
    senal_rx : array de longitud_total con la señal compuesta + ruido.

    Nota de diseño:
    ---------------
    El ruido se añade UNA SOLA VEZ sobre la suma, no por separado a cada
    paquete. Esto es correcto físicamente: el canal tiene un único ruido
    térmico, independientemente de cuántos transmisores haya activos.
    La interferencia entre paquetes solapados es determinista (suma de
    amplitudes), mientras que el ruido es estocástico.

    Integración con ALOHA (MAC → PHY):
    ------------------------------------
    Los instantes_llegada pueden venir directamente de la simulación ALOHA.
    Convierte el tiempo continuo de ALOHA a muestras multiplicando por la
    tasa de símbolo (muestras_por_simbolo). Con 1 muestra/símbolo, el
    instante en frame times coincide directamente con el índice de muestra.
    """
    senal_compuesta = np.zeros(longitud_total)

    for paquete, t_inicio in paquetes_y_tiempos:
        t_fin = t_inicio + len(paquete)

        if t_fin > longitud_total:
            # el paquete se sale de la ventana: recorta (caso borde)
            paquete = paquete[:longitud_total - t_inicio]
            t_fin = longitud_total

        senal_compuesta[t_inicio:t_fin] += paquete   # += crea interferencia

    # potencia de referencia: un paquete BPSK puro tiene potencia = 1.0
    # usamos eso como referencia para la SNR, no la potencia de la suma
    potencia_ref  = 1.0
    SNR_lineal    = 10 ** (SNR_dB / 10)
    varianza_ruido = potencia_ref / SNR_lineal
    sigma          = np.sqrt(varianza_ruido)

    ruido    = np.random.normal(0, sigma, longitud_total)
    senal_rx = senal_compuesta + ruido

    return senal_rx