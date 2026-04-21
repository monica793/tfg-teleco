"""
Configuración congelada del protocolo común de evaluación.

Este módulo centraliza parámetros que deben ser idénticos para comparar
receptores (correlador y futuros modelos ML) en condiciones justas.
"""

# Tolerancia común para matching por evento y definición de positivos ROC.
TOLERANCIA_MUESTRAS = 4

# Grid oficial inicial de evaluación.
GRID_CARGA_G = (0.2, 0.4, 0.6, 0.8)
GRID_SNR_DB = (0.0, 3.0, 6.0, 10.0)

# Monte Carlo y semillas.
NUM_ITERACIONES_MC = 100
NUM_ITERACIONES_MC_RAPIDO = 25
SEMILLA_BASE = 7

# Parámetros base de paquete.
NUM_BITS_PRE = 13
NUM_BITS_DATOS = 20

# Barrido estándar de umbrales para ROC por índice.
NUM_PUNTOS_ROC = 101
