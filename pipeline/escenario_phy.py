"""
Generación de escenarios PHY compartidos: tráfico Pure ALOHA + canal con colisiones.

La misma salida (`senal_rx`, instantes verdaderos, metadatos) sirve para comparar
receptores (correlador hoy, red neuronal en el futuro).
"""
import random

import numpy as np

from aloha.pure_aloha import simular_pure_aloha
from pipeline.channel import canal_awgn_colision
from pipeline.correlator_decoder import buscar_picos_preambulo, correlador
from pipeline.metricas_receptor import (
    curva_roc_por_indice,
    evaluar_detecciones,
    metricas_evento_derivadas,
)
from pipeline.transmitter import generar_preambulo, generar_paquete


def generar_escenario_phy(
    carga_G,
    ventana_frame_times,
    snr_db,
    num_bits_pre=13,
    num_bits_datos=20,
    semilla=None,
    margen_muestras=None,
    incluir_mascara_colision_mac=True,
):
    """
    Construye una realización: motor MAC (Pure ALOHA) + paquetes BPSK + canal AWGN con colisiones.

    Parámetros
    ----------
    carga_G               : float — carga ofrecida G
    ventana_frame_times   : int — horizonte temporal MAC (en duraciones de paquete)
    snr_db                : float — SNR de referencia por transmisor (ver canal_awgn_colision)
    num_bits_pre          : int — longitud del preámbulo
    num_bits_datos        : int — bits de datos por paquete
    semilla               : int opcional — fija random y numpy para reproducibilidad
    margen_muestras       : int opcional — cola de silencio al final del buffer RX
    incluir_mascara_colision_mac : si True, incluye colisión según MAC clásico (referencia TFG)

    Retorna
    -------
    dict con claves en español (senal_rx, instantes_llegada_muestras, preambulo, ...)
    """
    if semilla is not None:
        np.random.seed(semilla)
        random.seed(semilla)

    preambulo = generar_preambulo(num_bits=num_bits_pre)
    paquete_ref = generar_paquete(preambulo, num_bits_datos=num_bits_datos)
    longitud_paquete_muestras = len(paquete_ref)

    if margen_muestras is None:
        margen_muestras = longitud_paquete_muestras

    if incluir_mascara_colision_mac:
        salida_mac = simular_pure_aloha(
            carga_G,
            ventana_frame_times,
            muestras_por_paquete=longitud_paquete_muestras,
            solo_throughput=False,
            devolver_mascara_colision_mac=True,
        )
        instantes_muestras, num_paquetes_con_colision_mac, colision_mac_clasica = salida_mac
    else:
        instantes_muestras, num_paquetes_con_colision_mac = simular_pure_aloha(
            carga_G,
            ventana_frame_times,
            muestras_por_paquete=longitud_paquete_muestras,
            solo_throughput=False,
            devolver_mascara_colision_mac=False,
        )
        colision_mac_clasica = None

    paquetes_y_tiempos = []
    for _ in instantes_muestras:
        paq = generar_paquete(preambulo, num_bits_datos=num_bits_datos)
        paquetes_y_tiempos.append((paq, _))

    if len(instantes_muestras) == 0:
        longitud_total = max(1, int(ventana_frame_times * longitud_paquete_muestras) + margen_muestras)
    else:
        ultimo_fin = max(t + len(p) for p, t in paquetes_y_tiempos)
        longitud_total = int(ultimo_fin + margen_muestras)

    senal_rx = canal_awgn_colision(
        paquetes_y_tiempos,
        SNR_dB=snr_db,
        longitud_total=longitud_total,
    )

    escenario = {
        "senal_rx": senal_rx,
        "instantes_llegada_muestras": np.asarray(instantes_muestras, dtype=np.int64),
        "num_paquetes_con_colision_mac": int(num_paquetes_con_colision_mac),
        "longitud_total": int(longitud_total),
        "carga_G": float(carga_G),
        "snr_db": float(snr_db),
        "preambulo": preambulo,
        "longitud_paquete_muestras": int(longitud_paquete_muestras),
        "ventana_frame_times": int(ventana_frame_times),
        "num_bits_pre": int(num_bits_pre),
        "num_bits_datos": int(num_bits_datos),
    }
    if colision_mac_clasica is not None:
        escenario["colision_mac_clasica"] = np.asarray(colision_mac_clasica, dtype=bool)
    return escenario


def ejecutar_receptor_correlador(escenario, tau, separacion_minima):
    """
    Receptor PHY clásico: correlación + picos con NMS.

    Retorna
    -------
    dict: instantes_detectados (np.ndarray), corr_norm
    """
    senal_rx = escenario["senal_rx"]
    preambulo = escenario["preambulo"]
    corr_norm = correlador(senal_rx, preambulo)
    picos = buscar_picos_preambulo(corr_norm, tau=tau, separacion_minima=separacion_minima)
    return {"instantes_detectados": picos, "corr_norm": corr_norm}


def ejecutar_monte_carlo_receptor_correlador(
    carga_G,
    ventana_frame_times,
    snr_db,
    tau,
    tolerancia_muestras,
    separacion_minima,
    num_iteraciones,
    semilla_base=0,
    num_bits_pre=13,
    num_bits_datos=20,
):
    """
    Repite generar_escenario_phy + correlador + evaluar_detecciones para estadística simple.

    Cada iteración usa semilla distinta (semilla_base + k) para independencia aproximada.
    """
    tp_hist = []
    fp_hist = []
    fn_hist = []
    paquetes_hist = []
    detecciones_hist = []

    for k in range(num_iteraciones):
        esc = generar_escenario_phy(
            carga_G,
            ventana_frame_times,
            snr_db,
            num_bits_pre=num_bits_pre,
            num_bits_datos=num_bits_datos,
            semilla=semilla_base + k,
        )
        sal_corr = ejecutar_receptor_correlador(
            esc,
            tau=tau,
            separacion_minima=separacion_minima,
        )
        metricas = evaluar_detecciones(
            esc["instantes_llegada_muestras"],
            sal_corr["instantes_detectados"],
            tolerancia_muestras,
        )
        tp_hist.append(metricas["tp"])
        fp_hist.append(metricas["fp"])
        fn_hist.append(metricas["fn"])
        paquetes_hist.append(metricas["num_verdaderos"])
        detecciones_hist.append(metricas["num_detectados"])
    tp_arr = np.asarray(tp_hist, dtype=float)
    fp_arr = np.asarray(fp_hist, dtype=float)
    fn_arr = np.asarray(fn_hist, dtype=float)
    paquetes_arr = np.asarray(paquetes_hist, dtype=float)
    detecciones_arr = np.asarray(detecciones_hist, dtype=float)

    tp_media = float(np.mean(tp_arr))
    fp_media = float(np.mean(fp_arr))
    fn_media = float(np.mean(fn_arr))
    paquetes_media = float(np.mean(paquetes_arr))
    detecciones_media = float(np.mean(detecciones_arr))
    derivadas = metricas_evento_derivadas(tp_media, fp_media, fn_media)
    return {
        "num_iteraciones": int(num_iteraciones),
        "tp_media": tp_media,
        "fp_media": fp_media,
        "fn_media": fn_media,
        "paquetes_medio": paquetes_media,
        "detecciones_medio": detecciones_media,
        "tp_std": float(np.std(tp_arr, ddof=0)),
        "fp_std": float(np.std(fp_arr, ddof=0)),
        "fn_std": float(np.std(fn_arr, ddof=0)),
        "recall": derivadas["recall"],
        "precision": derivadas["precision"],
        "f1": derivadas["f1"],
    }


def ejecutar_monte_carlo_roc_correlador(
    carga_G,
    ventana_frame_times,
    snr_db,
    tolerancia_muestras,
    num_iteraciones,
    semilla_base=0,
    num_bits_pre=13,
    num_bits_datos=20,
    taus=None,
):
    """
    ROC canónica por indice del correlador, promediada en Monte Carlo.

    Retorna FPR/TPR medios por umbral y AUC media.
    """
    if taus is None:
        taus = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        taus = np.asarray(taus, dtype=float).ravel()
        if taus.size == 0:
            raise ValueError("taus no puede ser vacío")

    tpr_sum = np.zeros_like(taus, dtype=float)
    fpr_sum = np.zeros_like(taus, dtype=float)
    auc_sum = 0.0

    for k in range(num_iteraciones):
        esc = generar_escenario_phy(
            carga_G,
            ventana_frame_times,
            snr_db,
            num_bits_pre=num_bits_pre,
            num_bits_datos=num_bits_datos,
            semilla=semilla_base + k,
        )
        corr_norm = correlador(esc["senal_rx"], esc["preambulo"])
        roc = curva_roc_por_indice(
            corr_norm=corr_norm,
            instantes_verdaderos=esc["instantes_llegada_muestras"],
            tolerancia_muestras=tolerancia_muestras,
            taus=taus,
        )
        tpr_sum += roc["tpr"]
        fpr_sum += roc["fpr"]
        auc_sum += roc["auc"]

    n = max(1, int(num_iteraciones))
    return {
        "num_iteraciones": int(num_iteraciones),
        "taus": taus,
        "tpr_media": tpr_sum / n,
        "fpr_media": fpr_sum / n,
        "auc_media": float(auc_sum / n),
    }


def barrer_grid_protocolo_correlador(
    cargas_G,
    snrs_db,
    ventana_frame_times,
    tau_evento,
    tolerancia_muestras,
    separacion_minima,
    num_iteraciones,
    semilla_base,
    num_bits_pre=13,
    num_bits_datos=20,
):
    """
    Ejecuta el protocolo común en un grid (G, SNR) y devuelve filas de reporte.
    """
    filas = []
    for g in cargas_G:
        for snr in snrs_db:
            resumen_evento = ejecutar_monte_carlo_receptor_correlador(
                carga_G=g,
                ventana_frame_times=ventana_frame_times,
                snr_db=snr,
                tau=tau_evento,
                tolerancia_muestras=tolerancia_muestras,
                separacion_minima=separacion_minima,
                num_iteraciones=num_iteraciones,
                semilla_base=semilla_base,
                num_bits_pre=num_bits_pre,
                num_bits_datos=num_bits_datos,
            )
            resumen_roc = ejecutar_monte_carlo_roc_correlador(
                carga_G=g,
                ventana_frame_times=ventana_frame_times,
                snr_db=snr,
                tolerancia_muestras=tolerancia_muestras,
                num_iteraciones=num_iteraciones,
                semilla_base=semilla_base,
                num_bits_pre=num_bits_pre,
                num_bits_datos=num_bits_datos,
            )
            filas.append(
                {
                    "G": float(g),
                    "SNR_dB": float(snr),
                    "tp_media": resumen_evento["tp_media"],
                    "fp_media": resumen_evento["fp_media"],
                    "fn_media": resumen_evento["fn_media"],
                    "tp_std": resumen_evento["tp_std"],
                    "fp_std": resumen_evento["fp_std"],
                    "fn_std": resumen_evento["fn_std"],
                    "recall": resumen_evento["recall"],
                    "precision": resumen_evento["precision"],
                    "f1": resumen_evento["f1"],
                    "auc": resumen_roc["auc_media"],
                }
            )
    return filas
