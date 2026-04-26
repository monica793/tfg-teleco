"""
Métricas de detección frente a instantes de llegada verdaderos del simulador.

Reutilizable por el correlador y por un receptor basado en red neuronal.
"""
import numpy as np


def evaluar_detecciones(instantes_verdaderos, instantes_detectados, tolerancia_muestras):
    """
    Empareja detecciones con instantes verdaderos dentro de ±tolerancia_muestras.

    Cada detección cuenta como un TP como máximo una vez (la más cercana por verdadero
    en caso de varias en ventana). Las detecciones no emparejadas son FP; los verdaderos
    sin detección en ventana son FN.

    Parámetros
    ----------
    instantes_verdaderos  : secuencia de int — inicios reales de paquete (muestras)
    instantes_detectados  : secuencia de int — salida del receptor
    tolerancia_muestras   : int — mitad del intervalo de aceptación

    Retorna
    -------
    dict con claves: tp, fp, fn, num_verdaderos, num_detectados
    """
    verdaderos = np.asarray(instantes_verdaderos, dtype=np.int64).ravel()
    detectados = np.asarray(instantes_detectados, dtype=np.int64).ravel()
    verdaderos = np.sort(verdaderos)
    detectados = np.sort(detectados)

    tol = int(tolerancia_muestras)
    usados = set()
    tp = 0

    for t in verdaderos:
        en_ventana = detectados[(detectados >= t - tol) & (detectados <= t + tol)]
        mejor = None
        mejor_dist = tol + 1
        for d in en_ventana:
            if d in usados:
                continue
            dist = abs(int(d) - int(t))
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = int(d)
        if mejor is not None:
            tp += 1
            usados.add(mejor)

    fp = int(len(detectados) - len(usados))
    fn = int(len(verdaderos) - tp)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_verdaderos": int(len(verdaderos)),
        "num_detectados": int(len(detectados)),
    }


def metricas_evento_derivadas(tp, fp, fn):
    """
    Métricas derivadas por evento para reporte común.
    """
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

    recall = tp / max(1.0, tp + fn)
    precision = tp / max(1.0, tp + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def construir_mascara_positivos_correlacion(
    longitud_corr,
    instantes_verdaderos,
    tolerancia_muestras,
):
    """
    Construye etiquetas binarias por indice del correlador para ROC canónica.

    Un indice mu es positivo si cae dentro de ±tolerancia_muestras de algun
    instante verdadero de llegada. El resto de indices son negativos.
    """
    longitud_corr = int(longitud_corr)
    if longitud_corr <= 0:
        raise ValueError("longitud_corr debe ser > 0")

    tol = int(tolerancia_muestras)
    if tol < 0:
        raise ValueError("tolerancia_muestras debe ser >= 0")

    verdaderos = np.asarray(instantes_verdaderos, dtype=np.int64).ravel()
    mascara = np.zeros(longitud_corr, dtype=bool)

    for t in verdaderos:
        inicio = max(0, int(t) - tol)
        fin = min(longitud_corr - 1, int(t) + tol)
        if inicio <= fin:
            mascara[inicio : fin + 1] = True

    return mascara


def curva_roc_por_indice(
    corr_norm,
    instantes_verdaderos,
    tolerancia_muestras,
    taus=None,
):
    """
    ROC a nivel de indice del correlador.

    Para cada umbral tau:
      pred[mu] = 1{corr_norm[mu] >= tau}
      true[mu] = 1 si mu está en ±tol de algun instante verdadero

    Retorna TPR/FPR por umbral y AUC estimada.
    """
    c = np.asarray(corr_norm, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("corr_norm no puede ser vacío")

    if taus is None:
        taus = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        taus = np.asarray(taus, dtype=float).ravel()
        if taus.size == 0:
            raise ValueError("taus no puede ser vacío")

    y_true = construir_mascara_positivos_correlacion(
        longitud_corr=c.size,
        instantes_verdaderos=instantes_verdaderos,
        tolerancia_muestras=tolerancia_muestras,
    )

    positivos = int(np.sum(y_true))
    negativos = int(c.size - positivos)

    tpr = np.zeros_like(taus, dtype=float)
    fpr = np.zeros_like(taus, dtype=float)
    tp = np.zeros_like(taus, dtype=np.int64)
    fp = np.zeros_like(taus, dtype=np.int64)
    tn = np.zeros_like(taus, dtype=np.int64)
    fn = np.zeros_like(taus, dtype=np.int64)

    for i, tau in enumerate(taus):
        y_pred = c >= float(tau)
        tp_i = int(np.sum(y_pred & y_true))
        fp_i = int(np.sum(y_pred & ~y_true))
        tn_i = int(np.sum(~y_pred & ~y_true))
        fn_i = int(np.sum(~y_pred & y_true))

        tp[i] = tp_i
        fp[i] = fp_i
        tn[i] = tn_i
        fn[i] = fn_i

        tpr[i] = tp_i / max(1, positivos)
        fpr[i] = fp_i / max(1, negativos)

    # AUC por regla trapezoidal en FPR ascendente.
    orden = np.argsort(fpr)
    auc = float(np.trapz(tpr[orden], fpr[orden]))

    return {
        "taus": taus,
        "tpr": tpr,
        "fpr": fpr,
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "num_positivos": positivos,
        "num_negativos": negativos,
    }


def curva_pr_por_indice(
    corr_norm,
    instantes_verdaderos,
    tolerancia_muestras,
    taus=None,
):
    """
    Curva Precision-Recall a nivel de indice del estadístico temporal.

    Para cada umbral tau:
      pred[mu] = 1{corr_norm[mu] >= tau}
      true[mu] = 1 si mu está en ±tol de algun instante verdadero
    """
    c = np.asarray(corr_norm, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("corr_norm no puede ser vacío")

    if taus is None:
        taus = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        taus = np.asarray(taus, dtype=float).ravel()
        if taus.size == 0:
            raise ValueError("taus no puede ser vacío")

    y_true = construir_mascara_positivos_correlacion(
        longitud_corr=c.size,
        instantes_verdaderos=instantes_verdaderos,
        tolerancia_muestras=tolerancia_muestras,
    )

    precision = np.zeros_like(taus, dtype=float)
    recall = np.zeros_like(taus, dtype=float)
    tp = np.zeros_like(taus, dtype=np.int64)
    fp = np.zeros_like(taus, dtype=np.int64)
    fn = np.zeros_like(taus, dtype=np.int64)

    positivos = int(np.sum(y_true))
    for i, tau in enumerate(taus):
        y_pred = c >= float(tau)
        tp_i = int(np.sum(y_pred & y_true))
        fp_i = int(np.sum(y_pred & ~y_true))
        fn_i = int(np.sum(~y_pred & y_true))

        tp[i] = tp_i
        fp[i] = fp_i
        fn[i] = fn_i

        precision[i] = tp_i / max(1, tp_i + fp_i)
        recall[i] = tp_i / max(1, positivos)

    # AUC(PR): integrar precision en función de recall creciente.
    orden = np.argsort(recall)
    recall_ord = recall[orden]
    precision_ord = precision[orden]
    pr_auc = float(np.trapz(precision_ord, recall_ord))

    return {
        "taus": taus,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_positivos": positivos,
    }
