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


def buscar_picos_nms(score, tau, ventana_nms):
    """
    Detección con umbral + supresión de no-máximos (NMS).

    Entre todas las muestras que superan `tau`, elimina las que estén a menos de
    `ventana_nms` muestras de otra con score mayor. Así, respuestas anchas (CNN)
    o rachas de correlación (correlador) producen como máximo una detección por
    zona de activación, igual que el número real de paquetes.

    Parámetros
    ----------
    score       : array de scores por muestra (float)
    tau         : umbral mínimo de activación
    ventana_nms : distancia mínima en muestras entre dos detecciones consecutivas

    Retorna
    -------
    picos : np.ndarray (int64) — instantes detectados tras NMS
    """
    c = np.asarray(score, dtype=float)
    candidatos = np.where(c >= tau)[0]
    if candidatos.size == 0:
        return np.array([], dtype=np.int64)

    scores_c = c[candidatos]
    orden = np.argsort(-scores_c)  # descendente por score

    kept = []
    for i in orden:
        d = int(candidatos[i])
        if all(abs(d - k) > ventana_nms for k in kept):
            kept.append(d)

    return np.array(sorted(kept), dtype=np.int64)


def buscar_picos_centro_activacion(score, tau):
    """
    Detección por centro de zona de activación.

    Para cada grupo de muestras contiguas con score >= tau, devuelve la muestra
    central de ese grupo como instante detectado. Esto es apropiado cuando el
    detector neuronal genera activaciones uniformes en una zona de anchura fija
    (e.g., 2·k_c1 + 1 muestras), ya que el centro geométrico de esa zona es el
    mejor estimador del instante real de inicio del paquete.

    Parámetros
    ----------
    score : array de scores por muestra (float)
    tau   : umbral mínimo de activación

    Retorna
    -------
    centros : np.ndarray (int64) — un instante detectado por zona de activación
    """
    c = np.asarray(score, dtype=float)
    activado = (c >= tau).astype(np.int8)
    if not np.any(activado):
        return np.array([], dtype=np.int64)

    cambios = np.diff(activado, prepend=0, append=0)
    inicios = np.where(cambios == 1)[0]
    fines   = np.where(cambios == -1)[0] - 1  # último índice incluido

    centros = ((inicios + fines) // 2).astype(np.int64)
    return centros
