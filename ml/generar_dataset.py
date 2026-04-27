"""
Generación del dataset de entrenamiento por sampleo de escenarios Pure ALOHA reales.

Filosofía (sustituye la generación por clases aisladas):
  En lugar de fabricar ventanas sintéticas con clases diseñadas a mano,
  se generan escenarios ALOHA completos con `generar_escenario_phy` (que ya
  incluye colisiones, ruido AWGN y tráfico Poisson real) y se extrae el
  dataset deslizando una ventana de 128 muestras sobre la señal larga.

Etiquetado (Diana Central):
  Una ventana que empieza en el índice i tiene su centro en (i + 64).
  Esa ventana recibe etiqueta Y=1 si y solo si su centro coincide exactamente
  (tolerancia ±0 en entrenamiento) con un instante real de llegada de paquete.
  Cualquier otro caso → Y=0.

Prevención de data leakage:
  El split train/val se realiza dividiendo ESCENARIOS COMPLETOS, nunca
  partiendo ventanas del mismo escenario en ambos conjuntos.

Balanceo de clases:
  El tráfico ALOHA genera un desbalance brutal (~1 positivo cada L muestras).
  Se aplica undersampling de la clase negativa: por cada positivo se guardan
  como máximo RATIO_UNDERSAMPLING negativos.

Hard negatives:
  Las ventanas negativas cercanas al inicio real (|delta| <= HARD_NEG_RADIUS,
  excluyendo delta=0) se conservan de forma preferente y reciben un peso de
  pérdida mayor para forzar precisión temporal fina.
"""

import argparse
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pipeline.escenario_phy import generar_escenario_phy
from pipeline.protocolo_evaluacion import (
    NUM_BITS_PRE,
    NUM_BITS_DATOS,
    SEMILLA_BASE,
    LONG_VENTANA_CNN,
)

# ---------------------------------------------------------------------------
# Constantes del dataset (congeladas)
# ---------------------------------------------------------------------------
LONG_VENTANA       = LONG_VENTANA_CNN   # 128 muestras = 2^7
DIANA              = LONG_VENTANA // 2  # muestra 64: centro de la ventana
TOLERANCIA_LABEL   = 0                  # ±0 en entrenamiento (detección exacta)
RATIO_UNDERSAMPLING = 10                # negativos guardados por cada positivo
HARD_NEG_RADIUS = 5                     # vecindad temporal "difícil" (muestras)
HARD_NEG_WEIGHT = 1.5                   # penalización extra en pérdida


# ---------------------------------------------------------------------------
# Núcleo: extracción vectorizada de ventanas desde un escenario
# ---------------------------------------------------------------------------

def _extraer_ventanas_y_etiquetas(
    senal_rx: np.ndarray,
    instantes_llegada: np.ndarray,
    long_ventana: int = LONG_VENTANA,
    diana: int = DIANA,
    tolerancia: int = TOLERANCIA_LABEL,
    ratio_undersampling: int = RATIO_UNDERSAMPLING,
    hard_neg_radius: int = HARD_NEG_RADIUS,
    hard_neg_weight: float = HARD_NEG_WEIGHT,
    rng=None,
) -> tuple:
    """
    Desliza ventana de `long_ventana` muestras (stride=1) sobre un escenario.

    Para cada ventana i:
      - centro = i + diana
      - Y[i] = 1 si |centro - t_real| <= tolerancia para algún t_real
      - Y[i] = 0 en otro caso

    Después aplica undersampling de negativos.

    Retorna
    -------
    X : (n_sel, long_ventana, 2)  float32  [canales I, Q]
    Y : (n_sel,)                  int64
    W : (n_sel,)                  float32  (peso de pérdida por muestra)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(senal_rx)
    n_ventanas = N - long_ventana + 1
    if n_ventanas <= 0:
        return (
            np.empty((0, long_ventana, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    # Centros de cada ventana: centro[i] = i + diana
    centros = np.arange(n_ventanas, dtype=np.int64) + diana

    # Máscara de positivos vectorizada: O(n_paquetes) bucles, O(n_ventanas) ops cada uno
    y_true = np.zeros(n_ventanas, dtype=bool)
    hard_neg_mask = np.zeros(n_ventanas, dtype=bool)
    for t in instantes_llegada:
        dist = np.abs(centros - int(t))
        y_true |= dist <= tolerancia
        if hard_neg_radius > 0:
            hard_neg_mask |= (dist <= hard_neg_radius)

    # hard negatives: cercanos pero estrictamente negativos
    hard_neg_mask &= ~y_true

    # Índices de positivos y negativos
    idx_pos = np.where(y_true)[0]
    idx_neg = np.where(~y_true)[0]
    idx_neg_hard = np.where(hard_neg_mask)[0]
    idx_neg_rest = np.setdiff1d(idx_neg, idx_neg_hard, assume_unique=False)

    # Undersampling: máximo ratio × n_positivos negativos
    n_neg_max = max(1, len(idx_pos)) * ratio_undersampling
    if len(idx_neg) > n_neg_max:
        if len(idx_neg_hard) >= n_neg_max:
            idx_neg = rng.choice(idx_neg_hard, size=n_neg_max, replace=False)
        else:
            n_faltan = n_neg_max - len(idx_neg_hard)
            if len(idx_neg_rest) > n_faltan:
                idx_neg_rest = rng.choice(idx_neg_rest, size=n_faltan, replace=False)
            idx_neg = np.concatenate([idx_neg_hard, idx_neg_rest])
    else:
        idx_neg = np.concatenate([idx_neg_hard, idx_neg_rest])

    # Extracción vectorizada con sliding_window_view (sin copia de datos)
    # todas_ventanas: (n_ventanas, long_ventana) complejo
    todas_ventanas = sliding_window_view(senal_rx, window_shape=long_ventana)

    # Seleccionar solo los índices necesarios y convertir a I/Q float32
    idx_sel = np.concatenate([idx_pos, idx_neg])
    ventanas_sel = todas_ventanas[idx_sel]                 # (n_sel, L) complejo

    # (n_sel, L, 2): canal 0 = I, canal 1 = Q
    X = np.stack(
        [ventanas_sel.real.astype(np.float32),
         ventanas_sel.imag.astype(np.float32)],
        axis=-1,
    )

    Y = np.concatenate([
        np.ones(len(idx_pos), dtype=np.int64),
        np.zeros(len(idx_neg), dtype=np.int64),
    ])

    w_pos = np.ones(len(idx_pos), dtype=np.float32)
    if len(idx_neg) > 0:
        w_neg = np.where(np.isin(idx_neg, idx_neg_hard), float(hard_neg_weight), 1.0).astype(np.float32)
    else:
        w_neg = np.empty(0, dtype=np.float32)
    W = np.concatenate([w_pos, w_neg]).astype(np.float32)

    return X, Y, W


# ---------------------------------------------------------------------------
# Generador principal: sampleo de escenarios ALOHA
# ---------------------------------------------------------------------------

def generar_dataset_desde_escenarios(
    n_escenarios_train: int = 200,
    n_escenarios_val: int = 50,
    lista_G: tuple = (0.2, 0.4, 0.6, 0.8),
    lista_SNR_dB: tuple = (0.0, 3.0, 6.0, 10.0),
    ventana_frame_times: int = 50,
    ratio_undersampling: int = RATIO_UNDERSAMPLING,
    semilla_base: int = SEMILLA_BASE,
    long_ventana: int = LONG_VENTANA,
    diana: int = DIANA,
    tolerancia_label: int = TOLERANCIA_LABEL,
    hard_neg_radius: int = HARD_NEG_RADIUS,
    hard_neg_weight: float = HARD_NEG_WEIGHT,
) -> tuple:
    """
    Genera dataset sampleando escenarios ALOHA reales para cubrir todo el
    espacio operativo (G × SNR).

    Prevención de data leakage:
      Los primeros n_escenarios_train escenarios van a train;
      los siguientes n_escenarios_val van a val.
      Nunca se mezclan ventanas del mismo escenario en ambos conjuntos.

    Retorna
    -------
    X_train, Y_train, W_train, X_val, Y_val, W_val  (arrays NumPy)
    X tiene forma (N, long_ventana, 2) float32; Y tiene forma (N,) int64.
    """
    rng = np.random.default_rng(semilla_base)
    n_total = n_escenarios_train + n_escenarios_val
    n_cond  = len(lista_G) * len(lista_SNR_dB)

    print(f"Generando dataset desde escenarios ALOHA reales:")
    print(f"  Condiciones (G × SNR) : {n_cond}")
    print(f"  Escenarios por condición: {n_total} ({n_escenarios_train} train + {n_escenarios_val} val)")
    print(f"  Paquete: {NUM_BITS_PRE} (ZC) + {NUM_BITS_DATOS} (BPSK) = {NUM_BITS_PRE + NUM_BITS_DATOS} muestras")
    print(f"  Ventana CNN: {long_ventana}  |  Diana: {diana}  |  Undersampling ratio: {ratio_undersampling}")

    X_train_list, Y_train_list, W_train_list = [], [], []
    X_val_list,   Y_val_list,   W_val_list   = [], [], []

    for g in lista_G:
        for snr in lista_SNR_dB:
            for k in range(n_total):
                # Semilla única por (G, SNR, k) para reproducibilidad total
                semilla_k = semilla_base + int(g * 1000) + int(snr * 100) + k

                esc = generar_escenario_phy(
                    carga_G=float(g),
                    ventana_frame_times=ventana_frame_times,
                    snr_db=float(snr),
                    num_bits_pre=NUM_BITS_PRE,
                    num_bits_datos=NUM_BITS_DATOS,
                    semilla=semilla_k,
                    usar_preambulo=False,
                )

                X_k, Y_k, W_k = _extraer_ventanas_y_etiquetas(
                    senal_rx=esc["senal_rx"],
                    instantes_llegada=esc["instantes_llegada_muestras"],
                    long_ventana=long_ventana,
                    diana=diana,
                    tolerancia=tolerancia_label,
                    ratio_undersampling=ratio_undersampling,
                    hard_neg_radius=hard_neg_radius,
                    hard_neg_weight=hard_neg_weight,
                    rng=rng,
                )

                if len(X_k) == 0:
                    continue

                # Split por escenario completo (anti-leakage)
                if k < n_escenarios_train:
                    X_train_list.append(X_k)
                    Y_train_list.append(Y_k)
                    W_train_list.append(W_k)
                else:
                    X_val_list.append(X_k)
                    Y_val_list.append(Y_k)
                    W_val_list.append(W_k)

    # Concatenar todos los escenarios
    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    W_train = np.concatenate(W_train_list, axis=0)
    X_val   = np.concatenate(X_val_list,   axis=0)
    Y_val   = np.concatenate(Y_val_list,   axis=0)
    W_val   = np.concatenate(W_val_list,   axis=0)

    # Mezcla aleatoria dentro de cada conjunto (sin cruzar conjuntos)
    idx_tr = rng.permutation(len(X_train))
    idx_vl = rng.permutation(len(X_val))
    X_train, Y_train, W_train = X_train[idx_tr], Y_train[idx_tr], W_train[idx_tr]
    X_val,   Y_val,   W_val   = X_val[idx_vl],   Y_val[idx_vl],   W_val[idx_vl]

    print(f"\nResultado:")
    print(f"  X_train: {X_train.shape}  |  positivos: {Y_train.sum()} ({100*Y_train.mean():.1f}%)")
    print(f"  X_val  : {X_val.shape}    |  positivos: {Y_val.sum()} ({100*Y_val.mean():.1f}%)")
    print(f"  Peso medio (train/val): {W_train.mean():.3f} / {W_val.mean():.3f}")

    return X_train, Y_train, W_train, X_val, Y_val, W_val


def guardar_dataset(
    directorio_salida: str,
    n_escenarios_train: int = 200,
    n_escenarios_val: int = 50,
    lista_G: tuple = (0.2, 0.4, 0.6, 0.8),
    lista_SNR_dB: tuple = (0.0, 3.0, 6.0, 10.0),
    ventana_frame_times: int = 50,
    ratio_undersampling: int = RATIO_UNDERSAMPLING,
    semilla_base: int = SEMILLA_BASE,
) -> None:
    """
    Genera y guarda el dataset en formato .npy en el directorio especificado.
    """
    os.makedirs(directorio_salida, exist_ok=True)

    X_train, Y_train, W_train, X_val, Y_val, W_val = generar_dataset_desde_escenarios(
        n_escenarios_train=n_escenarios_train,
        n_escenarios_val=n_escenarios_val,
        lista_G=lista_G,
        lista_SNR_dB=lista_SNR_dB,
        ventana_frame_times=ventana_frame_times,
        ratio_undersampling=ratio_undersampling,
        semilla_base=semilla_base,
    )

    np.save(os.path.join(directorio_salida, "X_train.npy"), X_train)
    np.save(os.path.join(directorio_salida, "Y_train.npy"), Y_train)
    np.save(os.path.join(directorio_salida, "W_train.npy"), W_train)
    np.save(os.path.join(directorio_salida, "X_val.npy"),   X_val)
    np.save(os.path.join(directorio_salida, "Y_val.npy"),   Y_val)
    np.save(os.path.join(directorio_salida, "W_val.npy"),   W_val)

    tam_mb = (X_train.nbytes + X_val.nbytes) / 1e6
    print(f"\nDataset guardado en: {directorio_salida}")
    print(f"  X_train.npy : {X_train.shape}  {X_train.dtype}")
    print(f"  Y_train.npy : {Y_train.shape}  {Y_train.dtype}")
    print(f"  W_train.npy : {W_train.shape}  {W_train.dtype}")
    print(f"  X_val.npy   : {X_val.shape}    {X_val.dtype}")
    print(f"  Y_val.npy   : {Y_val.shape}    {Y_val.dtype}")
    print(f"  W_val.npy   : {W_val.shape}    {W_val.dtype}")
    print(f"  Tamaño en disco: ~{tam_mb:.0f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset de detección CNN desde escenarios ALOHA reales."
    )
    parser.add_argument("--salida",            type=str,   default="data/dataset_aloha")
    parser.add_argument("--n_train",           type=int,   default=200)
    parser.add_argument("--n_val",             type=int,   default=50)
    parser.add_argument("--ventana_ft",        type=int,   default=50,
                        help="Duración de cada escenario en frame times")
    parser.add_argument("--ratio_undersample", type=int,   default=RATIO_UNDERSAMPLING)
    parser.add_argument("--semilla",           type=int,   default=SEMILLA_BASE)
    args = parser.parse_args()

    guardar_dataset(
        directorio_salida=args.salida,
        n_escenarios_train=args.n_train,
        n_escenarios_val=args.n_val,
        ventana_frame_times=args.ventana_ft,
        ratio_undersampling=args.ratio_undersample,
        semilla_base=args.semilla,
    )
