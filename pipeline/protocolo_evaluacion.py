"""
Configuración congelada del protocolo común de evaluación.

Este módulo centraliza parámetros que deben ser idénticos para comparar
receptores (correlador y CNN) en condiciones justas.

Estandarización física (Bloque 1):
  - Preámbulo Zadoff-Chu : 23 muestras (número primo)
  - Payload BPSK         : 105 muestras
  - Paquete total        : 128 = 2^7 (hardware-friendly, alineado con ventana CNN)
"""

# ---------------------------------------------------------------------------
# Parámetros físicos del paquete (congelados para todo el proyecto)
# ---------------------------------------------------------------------------
NUM_BITS_PRE   = 23    # longitud preámbulo Zadoff-Chu (primo → propiedades CAZAC)
NUM_BITS_DATOS = 105   # payload BPSK
LONG_PAQUETE   = NUM_BITS_PRE + NUM_BITS_DATOS   # = 128 = 2^7

# Ventana de observación CNN y correlador (debe coincidir con LONG_PAQUETE)
LONG_VENTANA_CNN = 128

# ---------------------------------------------------------------------------
# Protocolo de evaluación
# ---------------------------------------------------------------------------

# Tolerancia para matching por evento y definición de positivos en ROC.
TOLERANCIA_MUESTRAS = 4

# Grid oficial de evaluación (G × SNR).
GRID_CARGA_G = (0.2, 0.4, 0.6, 0.8)
GRID_SNR_DB  = (0.0, 3.0, 6.0, 10.0)

# Monte Carlo: iteraciones para resultados finales y modo rápido (demo/debug).
NUM_ITERACIONES_MC       = 100
NUM_ITERACIONES_MC_RAPIDO = 25

# Semilla base reproducible.
SEMILLA_BASE = 7

# Barrido de umbrales para curva ROC.
NUM_PUNTOS_ROC = 101
