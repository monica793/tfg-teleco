"""
Pure ALOHA — funciones de simulación y cálculo teórico
"""
import random
import math
import numpy as np

def throughput_teorico(G):
    """S = G * e^(-2G)"""
    return G * math.exp(-2 * G)

def simular_pure_aloha(G, ventana_frame_times, muestras_por_paquete = None, solo_throughput=False):
    """
    Simulador de tráfico Pure ALOHA optimizado para detección con correlador.
    
    ¿Qué hace esta función?
    1. Calcula cuántos paquetes se intentan enviar en un tiempo determinado (Proceso de Poisson).
    2. Asigna a cada paquete un instante de inicio aleatorio.
    3. Detecta qué paquetes chocan físicamente (si se solapan en el tiempo).
    4. Convierte esos tiempos en índices reales de memoria para que el correlador sepa dónde buscar.

    Parámetros:
    -----------
    G : float
        Carga del canal. Representa el número medio de intentos de envío por cada 
        "hueco" de tiempo (frame time). 
        Ejemplo: G=0.5 significa que el canal está al 50% de su capacidad teórica.
        
    ventana_frame_times : int
        La duración total de la simulación medida en "unidades de paquete".
        Ejemplo: Si vale 100, simulamos el tiempo que tardarían en pasar 100 paquetes seguidos.
        
    muestras_por_paquete : int, opcional
        Cuántos datos (muestras) mide tu señal real (tus 1, 1, -1, -1...).
        Es el "largo" de tu vector de símbolos. Obligatorio si no es solo_throughput.
        
    solo_throughput : bool
        Si es True: Devuelve solo el valor S (éxitos/tiempo) para tus gráficas.
        Si es False: Devuelve los índices exactos para el correlador.

    Retorna:
    --------
    Si solo_throughput=True:
        S_simulado (float): El rendimiento del canal (0.0 a 0.18).
    Si solo_throughput=False:
        instantes_muestras (list[int]): Lista de posiciones donde empiezan tus señales en el buffer.
        num_colisiones_mac (int): Total de paquetes que se han pisado entre sí.
    """
    # 1. El número de paquetes sigue una distribución Poisson según la carga y ventana
    num_frames = np.random.poisson(G * ventana_frame_times)

    if num_frames == 0:
        return 0 if solo_throughput else ([], 0)
    
    tiempos = sorted([random.uniform(0, ventana_frame_times) for _ in range(num_frames)])
    colisionados = [False] * num_frames

    for i, t in enumerate(tiempos):
        vulnerable_inicio = t - 1.0
        vulnerable_fin    = t + 1.0
        # Mirar hacia atrás.
        j = i - 1 
        while j >= 0 and tiempos[j] > vulnerable_inicio:
            colisionados[i] = True
            colisionados[j] = True # Si i choca con j, j también choca con i
            j -= 1

        # Mirar hacia adelante. 
        j = i + 1
        while j < num_frames and tiempos[j] < vulnerable_fin:
            colisionados[i] = True
            colisionados[j] = True # Si i choca con j, j también choca con i
            j += 1
                
    # Cálculo de resultados 
    num_colisiones_mac = sum(colisionados)
    exitos = num_frames - num_colisiones_mac
    S_simulado = exitos / ventana_frame_times # S = Éxitos / Tiempo total

    if solo_throughput:
        return S_simulado
    
    # Mapeo a muestras PHY para el correlador
    instantes_muestras = [int(t * muestras_por_paquete) for t in tiempos]

    return instantes_muestras, num_colisiones_mac

def barrer_G(G_valores, num_frames=50_000):
    """Devuelve listas (S_teoricos, S_simulados) para una lista de valores G."""
    S_teoricos  = []
    S_simulados = []
    for G in G_valores:
        S_teoricos.append(throughput_teorico(G))
        S_simulados.append(simular_pure_aloha(G, num_frames))
    return S_teoricos, S_simulados