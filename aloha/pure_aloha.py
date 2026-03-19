"""
Pure ALOHA — funciones de simulación y cálculo teórico
"""
import random
import math

def throughput_teorico(G):
    """S = G * e^(-2G)"""
    return G * math.exp(-2 * G)

def simular_pure_aloha(G, num_frames=50_000):
    """Simula Pure ALOHA para una carga G. Devuelve S simulado.
    Al usar random.uniform sobre un tiempo total $T$, estamos creando un Proceso de Poisson. 
    """
    T = num_frames / G

    tiempos = sorted([random.uniform(0, T) for _ in range(num_frames)])
    exitos = 0
    for i, t in enumerate(tiempos):
        vulnerable_inicio = t - 1.0
        vulnerable_fin    = t + 1.0
        colision = False
        j = i - 1
        while j >= 0 and tiempos[j] > vulnerable_inicio:
            colision = True
            break
            j -= 1
        if not colision:
            j = i + 1
            while j < num_frames and tiempos[j] < vulnerable_fin:
                colision = True
                break
                j += 1
        if not colision:
            exitos += 1
    return exitos / T

def barrer_G(G_valores, num_frames=50_000):
    """Devuelve listas (S_teoricos, S_simulados) para una lista de valores G."""
    S_teoricos  = []
    S_simulados = []
    for G in G_valores:
        S_teoricos.append(throughput_teorico(G))
        S_simulados.append(simular_pure_aloha(G, num_frames))
    return S_teoricos, S_simulados