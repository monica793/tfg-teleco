import numpy as np


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

    Nota
    ----
    Modo 'valid' de np.correlate: el índice k del array de salida corresponde a
    alinear el preámbulo con señal_rx[k : k + L], es decir, el inicio del preámbulo
    en la señal recibida está en la muestra k (ver mapear_indice_correlador_a_muestra_rx).
    """
    L = len(preambulo)

    corr = np.correlate(señal_rx, preambulo, mode="valid")

    corr_norm = np.abs(corr) / L

    return corr_norm


def mapear_indice_correlador_a_muestra_rx(indice_correlador):
    """
    Con correlación en modo 'valid', el índice del correlador coincide con la muestra
    de inicio del preámbulo en la señal recibida (origen 0 en el vector RX).
    """
    return int(indice_correlador)


def buscar_picos_preambulo(corr_norm, tau, separacion_minima):
    """
    Selecciona picos de correlación por encima de tau con supresión no máxima (NMS).

    Se ordenan los candidatos por valor de correlación descendente y se acepta un
    pico si no hay otro ya aceptado a menos de separacion_minima muestras (típico:
    longitud del preámbulo o del paquete).

    Parámetros
    ----------
    corr_norm         : salida de correlador()
    tau               : umbral mínimo de correlación normalizada
    separacion_minima : distancia mínima entre picos (en índices del correlador = muestras RX)

    Retorna
    -------
    indices_picos : np.ndarray (int64) — índices en el dominio del correlador (= inicio preámbulo en RX)
    """
    c = np.asarray(corr_norm, dtype=float)
    candidatos = np.where(c >= tau)[0]
    if candidatos.size == 0:
        return np.array([], dtype=np.int64)

    valores = c[candidatos]
    orden = np.argsort(-valores)
    candidatos = candidatos[orden]

    elegidos = []
    for idx in candidatos:
        idx = int(idx)
        if all(abs(idx - e) >= separacion_minima for e in elegidos):
            elegidos.append(idx)

    return np.array(sorted(elegidos), dtype=np.int64)
