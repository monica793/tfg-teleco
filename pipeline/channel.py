import numpy as np


def _ruido_awgn(longitud, sigma, complejo):
    """
    Genera ruido AWGN real o complejo circularmente simétrico.

    Si complejo=True:
        n = nI + j*nQ, con nI,nQ ~ N(0, sigma^2/2), de forma que E[|n|^2] = sigma^2.
    """
    if not complejo:
        return np.random.normal(0.0, sigma, longitud)

    sigma_eje = sigma / np.sqrt(2.0)
    ruido_i = np.random.normal(0.0, sigma_eje, longitud)
    ruido_q = np.random.normal(0.0, sigma_eje, longitud)
    return ruido_i + 1j * ruido_q


def canal_awgn(paquete, SNR_dB, longitud_total=None, instante_llegada=None):
    """
    Simula el canal físico para señales reales o complejas.

    Coloca un paquete en el tiempo y añade ruido AWGN. Si el paquete es complejo,
    el ruido es complejo circularmente simétrico.
    """
    paquete = np.asarray(paquete)
    L = len(paquete)
    es_compleja = np.iscomplexobj(paquete)
    dtype_rx = np.complex128 if es_compleja else np.float64

    if longitud_total is None:
        longitud_total = 5 * L

    if instante_llegada is None:
        instante_llegada = np.random.randint(L, longitud_total - L)

    señal_limpia = np.zeros(longitud_total, dtype=dtype_rx)
    fin = instante_llegada + L
    señal_limpia[instante_llegada:fin] = paquete.astype(dtype_rx)

    potencia_señal = np.mean(np.abs(paquete) ** 2)
    SNR_lineal = 10 ** (SNR_dB / 10)
    varianza_ruido = potencia_señal / SNR_lineal
    sigma = np.sqrt(varianza_ruido)

    ruido = _ruido_awgn(longitud_total, sigma=sigma, complejo=es_compleja)
    señal_rx = señal_limpia + ruido

    return señal_rx, instante_llegada


def canal_awgn_colision(paquetes_y_tiempos, SNR_dB, longitud_total):
    """
    Canal AWGN con colisión para múltiples paquetes (real o complejo).

    El ruido se añade una sola vez a la señal compuesta.
    """
    contiene_compleja = any(np.iscomplexobj(np.asarray(p)) for p, _ in paquetes_y_tiempos)
    dtype_rx = np.complex128 if contiene_compleja else np.float64
    senal_compuesta = np.zeros(longitud_total, dtype=dtype_rx)

    for paquete, t_inicio in paquetes_y_tiempos:
        paquete = np.asarray(paquete, dtype=dtype_rx)
        t_inicio = int(t_inicio)
        if t_inicio < 0:
            t_inicio = 0
        t_fin = t_inicio + len(paquete)

        if t_inicio >= longitud_total:
            continue

        if t_fin > longitud_total:
            paquete = paquete[:longitud_total - t_inicio]
            t_fin = longitud_total

        senal_compuesta[t_inicio:t_fin] += paquete

    potencia_ref = 1.0
    SNR_lineal = 10 ** (SNR_dB / 10)
    varianza_ruido = potencia_ref / SNR_lineal
    sigma = np.sqrt(varianza_ruido)

    ruido = _ruido_awgn(longitud_total, sigma=sigma, complejo=contiene_compleja)
    senal_rx = senal_compuesta + ruido

    return senal_rx
