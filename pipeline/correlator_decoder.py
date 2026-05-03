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


def buscar_picos_preambulo(corr_norm, tau):
    """
    Selecciona detecciones por umbral simple (todas las muestras con c >= tau).

    Parámetros
    ----------
    corr_norm : salida de correlador() o score por muestra
    tau       : umbral mínimo

    Retorna
    -------
    indices_picos : np.ndarray (int64)
    """
    c = np.asarray(corr_norm, dtype=float)
    candidatos = np.where(c >= tau)[0]
    if candidatos.size == 0:
        return np.array([], dtype=np.int64)
    return candidatos.astype(np.int64)
