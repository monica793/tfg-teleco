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
    curva_pr_por_indice,
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
    aplicar_fase_aleatoria_por_paquete=True,
    usar_preambulo=True,
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
    aplicar_fase_aleatoria_por_paquete : si True, aplica una fase global aleatoria
        constante dentro de cada paquete (modelo de offset de fase por usuario/paquete)
    usar_preambulo        : si True, paquete = [preámbulo | datos].
                            Si False, paquete = [solo datos] (modo ciego para NN).

    Retorna
    -------
    dict con claves en español (senal_rx, instantes_llegada_muestras, preambulo, ...)
    """
    if semilla is not None:
        np.random.seed(semilla)
        random.seed(semilla)

    if usar_preambulo:
        preambulo = generar_preambulo(num_bits=num_bits_pre)
        bits_datos_paquete = num_bits_datos
    else:
        preambulo = np.array([], dtype=np.complex128)
        # Mantener longitud total constante (= num_bits_pre + num_bits_datos) para
        # comparar detectores con la misma escala temporal de paquete.
        bits_datos_paquete = num_bits_pre + num_bits_datos

    paquete_ref = generar_paquete(preambulo, num_bits_datos=bits_datos_paquete)
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
    fases_paquetes_rad = []
    for _ in instantes_muestras:
        paq = generar_paquete(preambulo, num_bits_datos=bits_datos_paquete)
        if aplicar_fase_aleatoria_por_paquete:
            fase = float(np.random.uniform(0.0, 2.0 * np.pi))
            paq = paq.astype(np.complex128) * np.exp(1j * fase)
        else:
            fase = 0.0
        paquetes_y_tiempos.append((paq, _))
        fases_paquetes_rad.append(fase)

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
        "usar_preambulo": bool(usar_preambulo),
        "num_bits_datos_paquete": int(bits_datos_paquete),
        "fases_paquetes_rad": np.asarray(fases_paquetes_rad, dtype=np.float64),
        "aplicar_fase_aleatoria_por_paquete": bool(aplicar_fase_aleatoria_por_paquete),
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
    if len(preambulo) == 0:
        raise ValueError(
            "El correlador requiere preámbulo, pero el escenario fue generado con usar_preambulo=False."
        )
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

    Retorna ROC y PR medios por umbral, junto con AUC media.
    """
    if taus is None:
        taus = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        taus = np.asarray(taus, dtype=float).ravel()
        if taus.size == 0:
            raise ValueError("taus no puede ser vacío")

    tpr_sum = np.zeros_like(taus, dtype=float)
    fpr_sum = np.zeros_like(taus, dtype=float)
    precision_sum = np.zeros_like(taus, dtype=float)
    recall_sum = np.zeros_like(taus, dtype=float)
    roc_auc_sum = 0.0
    pr_auc_sum = 0.0

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
        pr = curva_pr_por_indice(
            corr_norm=corr_norm,
            instantes_verdaderos=esc["instantes_llegada_muestras"],
            tolerancia_muestras=tolerancia_muestras,
            taus=taus,
        )
        tpr_sum += roc["tpr"]
        fpr_sum += roc["fpr"]
        precision_sum += pr["precision"]
        recall_sum += pr["recall"]
        roc_auc_sum += roc["auc"]
        pr_auc_sum += pr["pr_auc"]

    n = max(1, int(num_iteraciones))
    return {
        "num_iteraciones": int(num_iteraciones),
        "taus": taus,
        "tpr_media": tpr_sum / n,
        "fpr_media": fpr_sum / n,
        "precision_media": precision_sum / n,
        "recall_media": recall_sum / n,
        "auc_media": float(roc_auc_sum / n),
        "pr_auc_media": float(pr_auc_sum / n),
    }


def ejecutar_receptor_neuronal(
    escenario: dict,
    modelo,
    umbral: float,
    separacion_minima: int,
    dispositivo: str = "cpu",
    stride: int = 1,
    batch_size: int = 1024,
    long_ventana: int = 128,
):
    """
    Receptor PHY basado en red neuronal CNN 1D.

    Desliza una ventana de `long_ventana` muestras sobre `senal_rx` con paso
    `stride` e infiere la probabilidad de que un paquete nuevo nazca en la
    muestra central de cada ventana. El score resultante es equivalente a
    `corr_norm` del correlador y permite usar el mismo árbitro de métricas.

    Parámetros
    ----------
    escenario        : dict generado por generar_escenario_phy()
    modelo           : ModeloCNN ya entrenado (nn.Module en modo eval)
    umbral           : umbral de probabilidad en [0, 1] para declarar detección
    separacion_minima: distancia mínima entre picos en muestras (NMS)
    dispositivo      : 'cpu' o 'cuda'
    stride           : paso de la ventana deslizante (1 = resolución máxima)
    batch_size       : número de ventanas por batch de inferencia
    long_ventana     : longitud de cada ventana (debe coincidir con la del modelo)

    Retorna
    -------
    dict con:
        score_por_muestra  : np.ndarray float32 (len senal_rx,) — score en [0,1]
                             (0 fuera del alcance de la ventana)
        instantes_detectados: np.ndarray int64 — instantes con score >= umbral tras NMS
    """
    import torch

    senal_rx = np.asarray(escenario["senal_rx"])
    N = len(senal_rx)
    mitad = long_ventana // 2   # posición de la diana dentro de cada ventana

    # Generación de ventanas deslizantes
    indices_inicio = np.arange(0, N - long_ventana + 1, stride, dtype=np.int64)
    if indices_inicio.size == 0:
        score_por_muestra = np.zeros(N, dtype=np.float32)
        return {
            "score_por_muestra": score_por_muestra,
            "instantes_detectados": np.array([], dtype=np.int64),
        }

    n_ventanas = len(indices_inicio)
    ventanas_iq = np.empty((n_ventanas, long_ventana, 2), dtype=np.float32)
    for i, inicio in enumerate(indices_inicio):
        seg = senal_rx[inicio: inicio + long_ventana]
        ventanas_iq[i, :, 0] = seg.real.astype(np.float32)
        ventanas_iq[i, :, 1] = seg.imag.astype(np.float32)

    # (N_ventanas, 128, 2) → (N_ventanas, 2, 128) para Conv1d
    ventanas_tensor = torch.from_numpy(
        np.transpose(ventanas_iq, (0, 2, 1))
    ).to(dispositivo)

    modelo = modelo.to(dispositivo)
    modelo.eval()

    scores_ventanas = np.empty(n_ventanas, dtype=np.float32)
    with torch.no_grad():
        for inicio_batch in range(0, n_ventanas, batch_size):
            fin_batch = min(inicio_batch + batch_size, n_ventanas)
            logits = modelo(ventanas_tensor[inicio_batch:fin_batch])  # (B, 1)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            scores_ventanas[inicio_batch:fin_batch] = probs

    # Asignar cada score a la muestra "diana" de su ventana
    score_por_muestra = np.zeros(N, dtype=np.float32)
    diana_indices = indices_inicio + mitad
    np.maximum.at(score_por_muestra, diana_indices, scores_ventanas)

    # NMS sobre el score usando la misma función que el correlador
    instantes_detectados = buscar_picos_preambulo(
        corr_norm=score_por_muestra,
        tau=umbral,
        separacion_minima=separacion_minima,
    )

    return {
        "score_por_muestra": score_por_muestra,
        "instantes_detectados": instantes_detectados,
    }


def ejecutar_monte_carlo_roc_neuronal(
    carga_G,
    ventana_frame_times,
    snr_db,
    tolerancia_muestras,
    num_iteraciones,
    modelo,
    semilla_base=0,
    num_bits_pre=13,
    num_bits_datos=20,
    dispositivo="cpu",
    stride=1,
    long_ventana=128,
    taus=None,
    usar_preambulo=False,
):
    """
    ROC/PR por indice para el detector neuronal, promediadas en Monte Carlo.

    Usa `score_por_muestra` como estadístico de decisión (equivalente a corr_norm):
    para cada tau, pred[mu] = 1{score[mu] >= tau}.
    """
    if taus is None:
        taus = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        taus = np.asarray(taus, dtype=float).ravel()
        if taus.size == 0:
            raise ValueError("taus no puede ser vacío")

    tpr_sum = np.zeros_like(taus, dtype=float)
    fpr_sum = np.zeros_like(taus, dtype=float)
    precision_sum = np.zeros_like(taus, dtype=float)
    recall_sum = np.zeros_like(taus, dtype=float)
    roc_auc_sum = 0.0
    pr_auc_sum = 0.0

    for k in range(num_iteraciones):
        esc = generar_escenario_phy(
            carga_G,
            ventana_frame_times,
            snr_db,
            num_bits_pre=num_bits_pre,
            num_bits_datos=num_bits_datos,
            semilla=semilla_base + k,
            usar_preambulo=usar_preambulo,
        )
        sal_ml = ejecutar_receptor_neuronal(
            escenario=esc,
            modelo=modelo,
            umbral=0.5,  # no afecta a ROC; aquí solo se usa score_por_muestra
            separacion_minima=num_bits_pre + num_bits_datos,
            dispositivo=dispositivo,
            stride=stride,
            long_ventana=long_ventana,
        )
        roc = curva_roc_por_indice(
            corr_norm=sal_ml["score_por_muestra"],
            instantes_verdaderos=esc["instantes_llegada_muestras"],
            tolerancia_muestras=tolerancia_muestras,
            taus=taus,
        )
        pr = curva_pr_por_indice(
            corr_norm=sal_ml["score_por_muestra"],
            instantes_verdaderos=esc["instantes_llegada_muestras"],
            tolerancia_muestras=tolerancia_muestras,
            taus=taus,
        )
        tpr_sum += roc["tpr"]
        fpr_sum += roc["fpr"]
        precision_sum += pr["precision"]
        recall_sum += pr["recall"]
        roc_auc_sum += roc["auc"]
        pr_auc_sum += pr["pr_auc"]

    n = max(1, int(num_iteraciones))
    return {
        "num_iteraciones": int(num_iteraciones),
        "taus": taus,
        "tpr_media": tpr_sum / n,
        "fpr_media": fpr_sum / n,
        "precision_media": precision_sum / n,
        "recall_media": recall_sum / n,
        "auc_media": float(roc_auc_sum / n),
        "pr_auc_media": float(pr_auc_sum / n),
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
                    "pr_auc": resumen_roc["pr_auc_media"],
                }
            )
    return filas
