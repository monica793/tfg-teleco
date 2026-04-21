"""
Generación offline del dataset para el detector ciego (sin preámbulo) de Pure ALOHA.

Diseño de clases (Diana Central):
  La red debe predecir 1 si y solo si un paquete NUEVO nace en la muestra central
  de la ventana (muestra 64 de un total de 128). Esto permite, en inferencia,
  deslizar la ventana de 1 en 1 y obtener resolución temporal de 1 muestra.

4 clases equiprobables (25% cada una):
  - Clase 0a (Ruido puro)     : 128 muestras AWGN. Etiqueta = 0.
  - Clase 0b (Paquete de paso): paquete BPSK completo (fase aleatoria) en las 128
                                 muestras + AWGN. El paquete ya estaba en curso,
                                 no hay nacimiento en la muestra 64. Etiqueta = 0.
  - Clase 1a (Inicio limpio)  : AWGN en 0-63, paquete BPSK (fase aleatoria)
                                 en 64-127 + AWGN. Etiqueta = 1.
  - Clase 1b (Colisión)       : paquete A (fase aleatoria) en 0-127 + paquete B
                                 (fase aleatoria distinta) que nace en 64-127 + AWGN.
                                 Etiqueta = 1.

Parámetros clave:
  - Ventana        : LONG_VENTANA = 128 muestras
  - Longitud paquete: LONG_PAQUETE = 64 muestras de BPSK puro (sin preámbulo)
  - Inicio diana   : DIANA = 64  (centro de la ventana)
  - SNR            : muestreado uniformemente en [SNR_MIN_DB, SNR_MAX_DB] por ejemplo
  - Fase           : muestreada uniformemente en [0, 2*pi) por paquete

Formato de salida:
  - NumPy en disco : (N, 128, 2)  float32  [I, Q como últimas dos dimensiones]
  - PyTorch entrada: (N, 2, 128)  float32  [transposición al cargar en DataModule]
"""

import argparse
import os

import numpy as np

# ---------------------------------------------------------------------------
# Constantes del dataset (congeladas para reproducibilidad)
# ---------------------------------------------------------------------------
LONG_VENTANA = 128
LONG_PAQUETE = 64
DIANA = LONG_VENTANA // 2          # muestra 64: índice de nacimiento del paquete positivo
SNR_MIN_DB = 0.0
SNR_MAX_DB = 12.0
N_TOTAL_DEFAULT = 200_000          # 150k train + 50k val
SPLIT_VAL = 0.25
SEMILLA_DEFAULT = 42


# ---------------------------------------------------------------------------
# Funciones auxiliares de síntesis de señal
# ---------------------------------------------------------------------------

def _sigma_desde_snr(snr_db: np.ndarray) -> np.ndarray:
    """
    Calcula la amplitud del ruido (sigma) a partir de la SNR en dB.

    Se asume potencia de señal unitaria (BPSK: |±1|² = 1), por lo que:
        SNR = P_s / P_n = 1 / sigma²  =>  sigma = 10^(-SNR_dB / 20)
    """
    return 10.0 ** (-snr_db / 20.0)


def _ruido_complejo(n_ejemplos: int, n_muestras: int, sigma: np.ndarray, rng) -> np.ndarray:
    """
    Genera ruido AWGN complejo circularmente simétrico.

    Cada eje (I, Q) tiene varianza sigma²/2 para que la potencia total sea sigma².

    Retorna array (n_ejemplos, n_muestras) complejo.
    """
    sigma_eje = sigma[:, None] / np.sqrt(2.0)   # (N, 1) broadcast a (N, n_muestras)
    return (
        rng.standard_normal((n_ejemplos, n_muestras)) * sigma_eje
        + 1j * rng.standard_normal((n_ejemplos, n_muestras)) * sigma_eje
    )


def _paquete_bpsk_rotado(n_ejemplos: int, n_muestras: int, rng) -> np.ndarray:
    """
    Genera n_ejemplos paquetes BPSK con fase aleatoria independiente por paquete.

    Retorna array (n_ejemplos, n_muestras) complejo.
    """
    bits = rng.integers(0, 2, size=(n_ejemplos, n_muestras)) * 2 - 1  # {-1, +1}
    fase = rng.uniform(0.0, 2.0 * np.pi, size=(n_ejemplos, 1))        # (N, 1) broadcast
    return bits.astype(np.float64) * np.exp(1j * fase)


def _complejo_a_iq_float32(x: np.ndarray) -> np.ndarray:
    """
    Convierte array complejo (N, L) a float32 (N, L, 2) con canales [I, Q].
    """
    out = np.empty((*x.shape, 2), dtype=np.float32)
    out[..., 0] = x.real.astype(np.float32)
    out[..., 1] = x.imag.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Generadores por clase
# ---------------------------------------------------------------------------

def _clase_0a_ruido_puro(n: int, rng) -> np.ndarray:
    """
    Clase 0a: ventana de ruido puro. No hay ninguna señal.
    """
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB, size=n)
    sigma = _sigma_desde_snr(snr_db)
    ventanas = _ruido_complejo(n, LONG_VENTANA, sigma, rng)
    return ventanas


def _clase_0b_paquete_de_paso(n: int, rng) -> np.ndarray:
    """
    Clase 0b: un paquete BPSK ya en curso ocupa las 128 muestras.
    No hay nacimiento en la muestra 64: el paquete empezó antes de la ventana.
    """
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB, size=n)
    sigma = _sigma_desde_snr(snr_db)
    senal = _paquete_bpsk_rotado(n, LONG_VENTANA, rng)
    ruido = _ruido_complejo(n, LONG_VENTANA, sigma, rng)
    return senal + ruido


def _clase_1a_inicio_limpio(n: int, rng) -> np.ndarray:
    """
    Clase 1a: muestras 0-63 solo tienen ruido; en la 64 nace un paquete BPSK nuevo.
    El paquete ocupa exactamente 64-127 (LONG_PAQUETE = 64 muestras).
    """
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB, size=n)
    sigma = _sigma_desde_snr(snr_db)
    ruido = _ruido_complejo(n, LONG_VENTANA, sigma, rng)

    paquete_b = _paquete_bpsk_rotado(n, LONG_PAQUETE, rng)
    senal = np.zeros((n, LONG_VENTANA), dtype=np.complex128)
    senal[:, DIANA:] = paquete_b   # inserta desde la muestra 64

    return senal + ruido


def _clase_1b_colision(n: int, rng) -> np.ndarray:
    """
    Clase 1b: paquete A ocupa toda la ventana (0-127); en la muestra 64 nace
    el paquete B, que se suma aditivamente a A. El nacimiento de B es el evento
    de interés (etiqueta = 1).
    """
    snr_db = rng.uniform(SNR_MIN_DB, SNR_MAX_DB, size=n)
    sigma = _sigma_desde_snr(snr_db)
    ruido = _ruido_complejo(n, LONG_VENTANA, sigma, rng)

    paquete_a = _paquete_bpsk_rotado(n, LONG_VENTANA, rng)

    paquete_b = _paquete_bpsk_rotado(n, LONG_PAQUETE, rng)
    contribucion_b = np.zeros((n, LONG_VENTANA), dtype=np.complex128)
    contribucion_b[:, DIANA:] = paquete_b

    return paquete_a + contribucion_b + ruido


# ---------------------------------------------------------------------------
# Generador principal del dataset
# ---------------------------------------------------------------------------

def generar_dataset(
    n_total: int = N_TOTAL_DEFAULT,
    split_val: float = SPLIT_VAL,
    semilla: int = SEMILLA_DEFAULT,
) -> tuple:
    """
    Genera el dataset completo de entrenamiento y validación.

    Parámetros
    ----------
    n_total   : número total de ejemplos (se divide en 4 clases iguales).
    split_val : fracción para validación (ej. 0.25 -> 75% train, 25% val).
    semilla   : semilla del generador aleatorio para reproducibilidad.

    Retorna
    -------
    X_train, Y_train, X_val, Y_val  (arrays NumPy float32/int64)
    X tiene forma (N, 128, 2) y Y tiene forma (N,).
    """
    rng = np.random.default_rng(semilla)

    n_por_clase = n_total // 4
    n_total_real = n_por_clase * 4   # ajuste si n_total no es divisible por 4

    print(f"Generando dataset: {n_total_real} ejemplos ({n_por_clase} por clase)")
    print(f"  SNR muestreado en [{SNR_MIN_DB}, {SNR_MAX_DB}] dB por ejemplo")
    print(f"  Longitud ventana: {LONG_VENTANA} | Longitud paquete: {LONG_PAQUETE} | Diana: {DIANA}")

    # Síntesis de cada clase
    x_0a = _clase_0a_ruido_puro(n_por_clase, rng)
    x_0b = _clase_0b_paquete_de_paso(n_por_clase, rng)
    x_1a = _clase_1a_inicio_limpio(n_por_clase, rng)
    x_1b = _clase_1b_colision(n_por_clase, rng)

    etiquetas_negativas = np.zeros(n_por_clase * 2, dtype=np.int64)
    etiquetas_positivas = np.ones(n_por_clase * 2, dtype=np.int64)

    X_complejo = np.concatenate([x_0a, x_0b, x_1a, x_1b], axis=0)  # (N, 128) complejo
    Y = np.concatenate([etiquetas_negativas, etiquetas_positivas], axis=0)

    X_iq = _complejo_a_iq_float32(X_complejo)   # (N, 128, 2) float32

    # Mezcla aleatoria antes del split
    indices = rng.permutation(n_total_real)
    X_iq = X_iq[indices]
    Y = Y[indices]

    n_val = int(n_total_real * split_val)
    n_train = n_total_real - n_val

    X_train, Y_train = X_iq[:n_train], Y[:n_train]
    X_val, Y_val = X_iq[n_train:], Y[n_train:]

    print(f"  Train: {n_train} | Val: {n_val}")
    print(f"  Positivos train: {Y_train.sum()} ({100*Y_train.mean():.1f}%)")
    print(f"  Positivos val  : {Y_val.sum()} ({100*Y_val.mean():.1f}%)")

    return X_train, Y_train, X_val, Y_val


def guardar_dataset(
    directorio_salida: str,
    n_total: int = N_TOTAL_DEFAULT,
    split_val: float = SPLIT_VAL,
    semilla: int = SEMILLA_DEFAULT,
) -> None:
    """
    Genera y guarda el dataset en formato .npy en el directorio especificado.
    """
    os.makedirs(directorio_salida, exist_ok=True)

    X_train, Y_train, X_val, Y_val = generar_dataset(
        n_total=n_total,
        split_val=split_val,
        semilla=semilla,
    )

    np.save(os.path.join(directorio_salida, "X_train.npy"), X_train)
    np.save(os.path.join(directorio_salida, "Y_train.npy"), Y_train)
    np.save(os.path.join(directorio_salida, "X_val.npy"), X_val)
    np.save(os.path.join(directorio_salida, "Y_val.npy"), Y_val)

    print(f"\nDataset guardado en: {directorio_salida}")
    print(f"  X_train.npy : {X_train.shape}  {X_train.dtype}")
    print(f"  Y_train.npy : {Y_train.shape}  {Y_train.dtype}")
    print(f"  X_val.npy   : {X_val.shape}    {X_val.dtype}")
    print(f"  Y_val.npy   : {Y_val.shape}    {Y_val.dtype}")
    tamanio_mb = (X_train.nbytes + X_val.nbytes) / 1e6
    print(f"  Tamaño total en disco: ~{tamanio_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Ejecución desde línea de comandos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera el dataset offline para el detector ML de Pure ALOHA.")
    parser.add_argument("--salida", type=str, default="data/dataset_aloha",
                        help="Directorio de salida para los archivos .npy")
    parser.add_argument("--n_total", type=int, default=N_TOTAL_DEFAULT,
                        help="Número total de ejemplos a generar")
    parser.add_argument("--split_val", type=float, default=SPLIT_VAL,
                        help="Fracción para validación (ej. 0.25)")
    parser.add_argument("--semilla", type=int, default=SEMILLA_DEFAULT,
                        help="Semilla para reproducibilidad")
    args = parser.parse_args()

    guardar_dataset(
        directorio_salida=args.salida,
        n_total=args.n_total,
        split_val=args.split_val,
        semilla=args.semilla,
    )
