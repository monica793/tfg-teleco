"""
Generación del dataset para la campaña experimental Fase 1 y siguientes.

Parámetros clave
----------------
representacion : "energia" | "iq" | "iq_energia"
    Determina los canales de entrada que verá la red.
    - "energia"    → 1 canal:  E = (I²+Q²) normalizado por media de ventana
    - "iq"         → 2 canales: I, Q
    - "iq_energia" → 3 canales: I, Q, E

modo_label : "onset_centro" | "ventana_llena"
    Define qué ventanas reciben etiqueta positiva.
    - "onset_centro" : y=1 si el inicio real cae en el centro (pos. 64)
    - "ventana_llena": y=1 si el inicio real cae al inicio de la ventana
                       (paquete ocupa toda la ventana exactamente)
    El segundo modo se añade para Fase 2 y siguientes.

Protocolo de generación (idéntico entre representaciones para comparabilidad)
---------------------------------------------------------------------------
- Escenarios Pure ALOHA reales (senal_rx completa).
- Ventana deslizante stride=1, longitud 128.
- Split por escenario completo (sin data leakage).
- Balance 50/50 positivo/negativo en train y val.
- Hard negatives (ventanas cercanas al onset) con prioridad en el muestreo
  y peso de pérdida ligeramente aumentado.
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
from ml.modelo_fase1 import IN_CHANNELS, REPRESENTACIONES_VALIDAS

# ---------------------------------------------------------------------------
# Constantes de protocolo (congeladas para Fase 1)
# ---------------------------------------------------------------------------
LONG_VENTANA        = LONG_VENTANA_CNN  # 128
DIANA               = LONG_VENTANA // 2  # 64 — centro de ventana
TOLERANCIA_LABEL    = 0                  # onset estricto
RATIO_UNDERSAMPLING = 1                  # 50/50
HARD_NEG_RADIUS     = 4                  # muestras alrededor del onset
HARD_NEG_WEIGHT     = 1.25              # peso extra en pérdida para hard negatives

LISTA_G_DEFAULT   = (0.2, 0.4, 0.6, 0.8)
LISTA_SNR_DEFAULT = (0.0, 3.0, 6.0, 10.0)


# ---------------------------------------------------------------------------
# Construcción de canales según representación
# ---------------------------------------------------------------------------

def _construir_canales(ventanas_sel: np.ndarray, representacion: str) -> np.ndarray:
    """
    Transforma ventanas complejas (n_sel, L) al array de canales (n_sel, L, C).

    representacion : "energia" → C=1, "iq" → C=2, "iq_energia" → C=3
    """
    I = ventanas_sel.real.astype(np.float32)   # (n_sel, L)
    Q = ventanas_sel.imag.astype(np.float32)   # (n_sel, L)

    if representacion == "iq":
        return np.stack([I, Q], axis=-1)        # (n_sel, L, 2)

    # Energía normalizada: E[n] = I[n]² + Q[n]², dividida por media de ventana
    E = I ** 2 + Q ** 2                         # (n_sel, L)
    mu = E.mean(axis=1, keepdims=True) + 1e-8
    E_norm = (E / mu).astype(np.float32)        # (n_sel, L)

    if representacion == "energia":
        return E_norm[:, :, None]               # (n_sel, L, 1)

    # "iq_energia"
    return np.stack([I, Q, E_norm], axis=-1)    # (n_sel, L, 3)


# ---------------------------------------------------------------------------
# Etiquetado según modo_label
# ---------------------------------------------------------------------------

def _calcular_etiquetas(
    centros: np.ndarray,
    indices_inicio: np.ndarray,
    instantes_llegada: np.ndarray,
    modo_label: str,
    tolerancia: int,
    hard_neg_radius: int,
) -> tuple:
    """
    Calcula y_true (bool) y hard_neg_mask (bool) para las n_ventanas ventanas.

    Retorna (y_true, hard_neg_mask) — ambos (n_ventanas,) bool.
    """
    n = len(centros)
    y_true = np.zeros(n, dtype=bool)
    hard_neg_mask = np.zeros(n, dtype=bool)

    if modo_label == "onset_centro":
        ref = centros   # distancia desde centro de ventana al onset
    elif modo_label == "ventana_llena":
        ref = indices_inicio  # distancia desde inicio de ventana al onset
    else:
        raise ValueError(f"modo_label desconocido: {modo_label!r}")

    for t in instantes_llegada:
        dist = np.abs(ref - int(t))
        y_true |= dist <= tolerancia
        if hard_neg_radius > 0:
            hard_neg_mask |= dist <= hard_neg_radius

    hard_neg_mask &= ~y_true
    return y_true, hard_neg_mask


# ---------------------------------------------------------------------------
# Extracción de ventanas + etiquetas + pesos para un escenario
# ---------------------------------------------------------------------------

def _extraer_ventanas_y_etiquetas(
    senal_rx: np.ndarray,
    instantes_llegada: np.ndarray,
    representacion: str = "iq",
    modo_label: str = "onset_centro",
    long_ventana: int = LONG_VENTANA,
    diana: int = DIANA,
    tolerancia: int = TOLERANCIA_LABEL,
    ratio_undersampling: int = RATIO_UNDERSAMPLING,
    hard_neg_radius: int = HARD_NEG_RADIUS,
    hard_neg_weight: float = HARD_NEG_WEIGHT,
    rng=None,
) -> tuple:
    """
    Desliza ventana stride=1 sobre el escenario y extrae X, Y, W.

    Retorna
    -------
    X : (n_sel, L, C)  float32
    Y : (n_sel,)        int64
    W : (n_sel,)        float32  peso de pérdida por muestra
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(senal_rx)
    n_ventanas = N - long_ventana + 1
    if n_ventanas <= 0:
        C = IN_CHANNELS[representacion]
        return (
            np.empty((0, long_ventana, C), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    indices_inicio = np.arange(n_ventanas, dtype=np.int64)
    centros = indices_inicio + diana

    y_true, hard_neg_mask = _calcular_etiquetas(
        centros, indices_inicio, instantes_llegada,
        modo_label, tolerancia, hard_neg_radius,
    )

    idx_pos = np.where(y_true)[0]
    idx_neg_hard = np.where(hard_neg_mask)[0]
    idx_neg_rest = np.where(~y_true & ~hard_neg_mask)[0]

    # Undersampling 50/50: máximo ratio × n_pos negativos, priorizando hard negatives
    n_neg_max = max(1, len(idx_pos)) * ratio_undersampling
    if len(idx_neg_hard) >= n_neg_max:
        idx_neg = rng.choice(idx_neg_hard, size=n_neg_max, replace=False)
    else:
        n_faltan = n_neg_max - len(idx_neg_hard)
        rest_sel = rng.choice(idx_neg_rest, size=min(n_faltan, len(idx_neg_rest)), replace=False)
        idx_neg = np.concatenate([idx_neg_hard, rest_sel])

    idx_sel = np.concatenate([idx_pos, idx_neg])
    todas_ventanas = sliding_window_view(senal_rx, window_shape=long_ventana)
    ventanas_sel = todas_ventanas[idx_sel]   # (n_sel, L) complejo

    X = _construir_canales(ventanas_sel, representacion)  # (n_sel, L, C)

    Y = np.concatenate([
        np.ones(len(idx_pos), dtype=np.int64),
        np.zeros(len(idx_neg), dtype=np.int64),
    ])

    w_neg = np.where(
        np.isin(idx_neg, idx_neg_hard), float(hard_neg_weight), 1.0
    ).astype(np.float32)
    W = np.concatenate([
        np.ones(len(idx_pos), dtype=np.float32),
        w_neg,
    ])

    return X, Y, W


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

def generar_dataset_desde_escenarios(
    representacion: str = "iq",
    modo_label: str = "onset_centro",
    n_escenarios_train: int = 200,
    n_escenarios_val: int = 50,
    lista_G: tuple = LISTA_G_DEFAULT,
    lista_SNR_dB: tuple = LISTA_SNR_DEFAULT,
    ventana_frame_times: int = 50,
    ratio_undersampling: int = RATIO_UNDERSAMPLING,
    semilla_base: int = SEMILLA_BASE,
    hard_neg_radius: int = HARD_NEG_RADIUS,
    hard_neg_weight: float = HARD_NEG_WEIGHT,
    usar_preambulo: bool = False,
) -> tuple:
    """
    Genera el dataset completo muestreando escenarios ALOHA reales.

    El split train/val se hace por escenario completo (sin data leakage).

    Retorna
    -------
    X_train, Y_train, W_train, X_val, Y_val, W_val
    X : (N, L, C) float32  donde C depende de 'representacion'
    Y : (N,) int64
    W : (N,) float32
    """
    if representacion not in REPRESENTACIONES_VALIDAS:
        raise ValueError(f"representacion debe ser uno de {REPRESENTACIONES_VALIDAS}")

    rng = np.random.default_rng(semilla_base)
    n_total = n_escenarios_train + n_escenarios_val
    n_canales = IN_CHANNELS[representacion]

    print(f"Dataset Fase 1")
    print(f"  Representación : {representacion} ({n_canales} canal(es))")
    print(f"  Modo label     : {modo_label}")
    print(f"  Balance        : 50/50  (ratio_undersampling={ratio_undersampling})")
    print(f"  Hard negatives : radio={hard_neg_radius}, peso={hard_neg_weight}")
    print(f"  Escenarios     : {n_escenarios_train} train + {n_escenarios_val} val")
    print(f"  G × SNR        : {lista_G} × {lista_SNR_dB}")

    X_tr, Y_tr, W_tr = [], [], []
    X_vl, Y_vl, W_vl = [], [], []

    for g in lista_G:
        for snr in lista_SNR_dB:
            for k in range(n_total):
                semilla_k = semilla_base + int(g * 1000) + int(snr * 100) + k
                esc = generar_escenario_phy(
                    carga_G=float(g),
                    ventana_frame_times=ventana_frame_times,
                    snr_db=float(snr),
                    num_bits_pre=NUM_BITS_PRE,
                    num_bits_datos=NUM_BITS_DATOS,
                    semilla=semilla_k,
                    usar_preambulo=usar_preambulo,
                )
                X_k, Y_k, W_k = _extraer_ventanas_y_etiquetas(
                    senal_rx=esc["senal_rx"],
                    instantes_llegada=esc["instantes_llegada_muestras"],
                    representacion=representacion,
                    modo_label=modo_label,
                    ratio_undersampling=ratio_undersampling,
                    hard_neg_radius=hard_neg_radius,
                    hard_neg_weight=hard_neg_weight,
                    rng=rng,
                )
                if len(X_k) == 0:
                    continue
                if k < n_escenarios_train:
                    X_tr.append(X_k); Y_tr.append(Y_k); W_tr.append(W_k)
                else:
                    X_vl.append(X_k); Y_vl.append(Y_k); W_vl.append(W_k)

    X_train = np.concatenate(X_tr, axis=0)
    Y_train = np.concatenate(Y_tr, axis=0)
    W_train = np.concatenate(W_tr, axis=0)
    X_val   = np.concatenate(X_vl, axis=0)
    Y_val   = np.concatenate(Y_vl, axis=0)
    W_val   = np.concatenate(W_vl, axis=0)

    # Mezcla aleatoria intra-conjunto
    idx_tr = rng.permutation(len(X_train))
    idx_vl = rng.permutation(len(X_val))
    X_train, Y_train, W_train = X_train[idx_tr], Y_train[idx_tr], W_train[idx_tr]
    X_val,   Y_val,   W_val   = X_val[idx_vl],   Y_val[idx_vl],   W_val[idx_vl]

    print(f"\n  X_train: {X_train.shape}  positivos: {Y_train.sum()} ({100*Y_train.mean():.1f}%)")
    print(f"  X_val  : {X_val.shape}    positivos: {Y_val.sum()} ({100*Y_val.mean():.1f}%)")

    return X_train, Y_train, W_train, X_val, Y_val, W_val


def guardar_dataset(
    directorio_salida: str,
    representacion: str = "iq",
    modo_label: str = "onset_centro",
    n_escenarios_train: int = 200,
    n_escenarios_val: int = 50,
    lista_G: tuple = LISTA_G_DEFAULT,
    lista_SNR_dB: tuple = LISTA_SNR_DEFAULT,
    ventana_frame_times: int = 50,
    ratio_undersampling: int = RATIO_UNDERSAMPLING,
    semilla_base: int = SEMILLA_BASE,
    hard_neg_radius: int = HARD_NEG_RADIUS,
    hard_neg_weight: float = HARD_NEG_WEIGHT,
    usar_preambulo: bool = False,
) -> None:
    os.makedirs(directorio_salida, exist_ok=True)
    X_tr, Y_tr, W_tr, X_vl, Y_vl, W_vl = generar_dataset_desde_escenarios(
        representacion=representacion,
        modo_label=modo_label,
        n_escenarios_train=n_escenarios_train,
        n_escenarios_val=n_escenarios_val,
        lista_G=lista_G,
        lista_SNR_dB=lista_SNR_dB,
        ventana_frame_times=ventana_frame_times,
        ratio_undersampling=ratio_undersampling,
        semilla_base=semilla_base,
        hard_neg_radius=hard_neg_radius,
        hard_neg_weight=hard_neg_weight,
        usar_preambulo=usar_preambulo,
    )
    for nombre, arr in [("X_train", X_tr), ("Y_train", Y_tr), ("W_train", W_tr),
                        ("X_val",   X_vl), ("Y_val",   Y_vl), ("W_val",   W_vl)]:
        np.save(os.path.join(directorio_salida, f"{nombre}.npy"), arr)

    tam_mb = (X_tr.nbytes + X_vl.nbytes) / 1e6
    print(f"\nDataset guardado en '{directorio_salida}' (~{tam_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset Fase 1 desde escenarios ALOHA reales."
    )
    parser.add_argument("--representacion", type=str, default="iq",
                        choices=list(REPRESENTACIONES_VALIDAS),
                        help="Representación de entrada: 'energia', 'iq' o 'iq_energia'.")
    parser.add_argument("--modo_label", type=str, default="onset_centro",
                        choices=["onset_centro", "ventana_llena"],
                        help="Definición del positivo. 'onset_centro' (Fase 1) o 'ventana_llena' (Fase 2).")
    parser.add_argument("--salida", type=str, default=None,
                        help="Directorio de salida. Por defecto: data/fase1_<representacion>_<modo_label>")
    parser.add_argument("--n_train",           type=int,   default=200)
    parser.add_argument("--n_val",             type=int,   default=50)
    parser.add_argument("--ventana_ft",        type=int,   default=50)
    parser.add_argument("--ratio_undersample", type=int,   default=RATIO_UNDERSAMPLING)
    parser.add_argument("--semilla",           type=int,   default=SEMILLA_BASE)
    parser.add_argument("--hard_neg_radius",   type=int,   default=HARD_NEG_RADIUS)
    parser.add_argument("--hard_neg_weight",   type=float, default=HARD_NEG_WEIGHT)
    parser.add_argument("--usar_preambulo",    action="store_true", default=False)
    args = parser.parse_args()

    salida = args.salida or f"data/fase1_{args.representacion}_{args.modo_label}"

    guardar_dataset(
        directorio_salida=salida,
        representacion=args.representacion,
        modo_label=args.modo_label,
        n_escenarios_train=args.n_train,
        n_escenarios_val=args.n_val,
        ventana_frame_times=args.ventana_ft,
        ratio_undersampling=args.ratio_undersample,
        semilla_base=args.semilla,
        hard_neg_radius=args.hard_neg_radius,
        hard_neg_weight=args.hard_neg_weight,
        usar_preambulo=args.usar_preambulo,
    )
