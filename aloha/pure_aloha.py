"""
Pure ALOHA — funciones de simulación y cálculo teórico
"""
import random
import math
import numpy as np

def throughput_teorico(G):
    """S = G * e^(-2G)"""
    return G * math.exp(-2 * G)

def simular_pure_aloha(
    G,
    ventana_frame_times,
    muestras_por_paquete=None,
    solo_throughput=False,
    devolver_mascara_colision_mac=False,
):
    """
    Simulador de tráfico Pure ALOHA optimizado para detección con correlador.

    ¿Qué hace esta función?
    1. Calcula cuántos paquetes se intentan enviar en un tiempo determinado (Proceso de Poisson).
    2. Asigna a cada paquete un instante de inicio aleatorio.
    3. Detecta qué paquetes chocan según el modelo MAC clásico (solape en el tiempo).
    4. Convierte esos tiempos en índices de muestra PHY para el receptor.

    Parámetros:
    -----------
    G : float
        Carga del canal (intentos medios por unidad de tiempo de paquete).
    ventana_frame_times : int
        Duración total de la simulación en unidades de duración de paquete.
    muestras_por_paquete : int, opcional
        Muestras por paquete (1 muestra/símbolo → longitud del vector BPSK del paquete).
        Obligatorio si solo_throughput es False.
    solo_throughput : bool
        Si True: solo devuelve S simulado.
        Si False: devuelve instantes en muestras y conteo de paquetes en colisión MAC clásica.
    devolver_mascara_colision_mac : bool
        Si True (y solo_throughput False): la tupla incluye una máscara por paquete
        (True = ese intento participa en al menos una colisión según MAC clásico).
        Referencia auxiliar para el TFG (captura vs modelo ideal); no sustituye a la
        verdad de llegada para métricas PHY.

    Retorna:
    --------
    solo_throughput=True:
        float — S simulado.
    solo_throughput=False, devolver_mascara_colision_mac=False:
        (instantes_muestras, num_paquetes_con_colision_mac)
    solo_throughput=False, devolver_mascara_colision_mac=True:
        (instantes_muestras, num_paquetes_con_colision_mac, colision_mac_clasica)
        colision_mac_clasica es list[bool] de longitud N (misma orden que instantes_muestras).
    """
    if not solo_throughput and muestras_por_paquete is None:
        raise ValueError(
            "muestras_por_paquete es obligatorio cuando solo_throughput=False"
        )

    num_paquetes = int(np.random.poisson(G * ventana_frame_times))

    if num_paquetes == 0:
        if solo_throughput:
            return 0.0
        if devolver_mascara_colision_mac:
            return [], 0, []
        return [], 0

    tiempos = sorted([random.uniform(0, ventana_frame_times) for _ in range(num_paquetes)])
    colisionados = [False] * num_paquetes

    for i, t in enumerate(tiempos):
        vulnerable_inicio = t - 1.0
        vulnerable_fin = t + 1.0
        j = i - 1
        while j >= 0 and tiempos[j] > vulnerable_inicio:
            colisionados[i] = True
            colisionados[j] = True
            j -= 1

        j = i + 1
        while j < num_paquetes and tiempos[j] < vulnerable_fin:
            colisionados[i] = True
            colisionados[j] = True
            j += 1

    num_paquetes_con_colision_mac = int(sum(colisionados))
    exitos = num_paquetes - num_paquetes_con_colision_mac
    S_simulado = exitos / ventana_frame_times

    if solo_throughput:
        return S_simulado

    instantes_muestras = [int(t * muestras_por_paquete) for t in tiempos]

    if devolver_mascara_colision_mac:
        return instantes_muestras, num_paquetes_con_colision_mac, colisionados
    return instantes_muestras, num_paquetes_con_colision_mac


def barrer_G(G_valores, ventana_frame_times=50_000):
    """
    Devuelve listas (S_teoricos, S_simulados) para una lista de valores G.
    Usa solo_throughput=True en la simulación (no requiere muestras_por_paquete).
    """
    S_teoricos = []
    S_simulados = []
    for G in G_valores:
        S_teoricos.append(throughput_teorico(G))
        S_simulados.append(
            simular_pure_aloha(
                G, ventana_frame_times, solo_throughput=True
            )
        )
    return S_teoricos, S_simulados
