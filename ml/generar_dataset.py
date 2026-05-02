"""
Generación de datasets para Fase 1/2/3 con un pipeline único y trazable.

Representaciones soportadas
---------------------------
- energia      : 1 canal  (E normalizada)
- iq           : 2 canales (I, Q)
- iq_energia   : 3 canales (I, Q, E)

Modos de etiquetado
-------------------
- onset_centro      : binario, y=1 si onset cae en centro de ventana
- ventana_llena     : binario, y=1 si onset coincide con inicio de ventana
- multiclase_onset  : multiclase (C0/C1/C2) por distancia al onset más cercano
    C1: |d| <= k_c1
    C2: k_c1 < |d| <= k_c2
    C0: |d| > k_c2
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


LONG_VENTANA = LONG_VENTANA_CNN
DIANA = LONG_VENTANA // 2
TOLERANCIA_LABEL = 0
RATIO_UNDERSAMPLING = 1
HARD_NEG_RADIUS = 4
HARD_NEG_WEIGHT = 1.25
LISTA_G_DEFAULT = (0.2, 0.4, 0.6, 0.8)
LISTA_SNR_DEFAULT = (0.0, 3.0, 6.0, 10.0)
MODOS_LABEL_VALIDOS = ("onset_centro", "ventana_llena", "multiclase_onset")


def _construir_canales(ventanas_sel: np.ndarray, representacion: str) -> np.ndarray:
    I = ventanas_sel.real.astype(np.float32)
    Q = ventanas_sel.imag.astype(np.float32)

    if representacion == "iq":
        return np.stack([I, Q], axis=-1)

    E = I ** 2 + Q ** 2
    E_norm = (E / (E.mean(axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    if representacion == "energia":
        return E_norm[:, :, None]

    return np.stack([I, Q, E_norm], axis=-1)


def _referencia_ventana(centros: np.ndarray, indices_inicio: np.ndarray, modo_label: str) -> np.ndarray:
    if modo_label == "onset_centro":
        return centros
    if modo_label == "ventana_llena":
        return indices_inicio
    if modo_label == "multiclase_onset":
        # Multiclase inicial sobre onset-centro (fase 3 inicial)
        return centros
    raise ValueError(f"modo_label desconocido: {modo_label!r}")


def _etiquetar_binario(ref: np.ndarray, onsets: np.ndarray, tolerancia: int, hard_neg_radius: int):
    y_true = np.zeros(len(ref), dtype=bool)
    hard_neg_mask = np.zeros(len(ref), dtype=bool)
    for t in onsets:
        dist = np.abs(ref - int(t))
        y_true |= dist <= tolerancia
        if hard_neg_radius > 0:
            hard_neg_mask |= dist <= hard_neg_radius
    hard_neg_mask &= ~y_true
    return y_true, hard_neg_mask


def _etiquetar_multiclase(ref: np.ndarray, onsets: np.ndarray, k_c1: int, k_c2: int):
    if k_c1 < 0 or k_c2 <= k_c1:
        raise ValueError("Debe cumplirse: k_c1 >= 0 y k_c2 > k_c1")
    y = np.zeros(len(ref), dtype=np.int64)  # C0 por defecto
    if len(onsets) == 0:
        return y, np.full(len(ref), np.inf, dtype=float)

    # Distancia al onset más cercano
    dmin = np.full(len(ref), np.inf, dtype=float)
    for t in onsets:
        d = np.abs(ref - int(t))
        dmin = np.minimum(dmin, d)

    y[dmin <= k_c1] = 1
    y[(dmin > k_c1) & (dmin <= k_c2)] = 2
    return y, dmin


def _balancear_binario(idx_pos, idx_neg_hard, idx_neg_rest, ratio_undersampling, rng):
    n_neg_max = max(1, len(idx_pos)) * int(ratio_undersampling)
    if len(idx_neg_hard) >= n_neg_max:
        idx_neg = rng.choice(idx_neg_hard, size=n_neg_max, replace=False)
    else:
        n_faltan = n_neg_max - len(idx_neg_hard)
        rest_sel = rng.choice(idx_neg_rest, size=min(n_faltan, len(idx_neg_rest)), replace=False) \
            if len(idx_neg_rest) > 0 else np.empty(0, dtype=np.int64)
        idx_neg = np.concatenate([idx_neg_hard, rest_sel])
    idx_sel = np.concatenate([idx_pos, idx_neg])
    y_sel = np.concatenate([np.ones(len(idx_pos), dtype=np.int64), np.zeros(len(idx_neg), dtype=np.int64)])
    w_neg = np.where(np.isin(idx_neg, idx_neg_hard), HARD_NEG_WEIGHT, 1.0).astype(np.float32)
    w_sel = np.concatenate([np.ones(len(idx_pos), dtype=np.float32), w_neg])
    return idx_sel, y_sel, w_sel


def _balancear_multiclase(y_multi: np.ndarray, rng):
    idx0 = np.where(y_multi == 0)[0]
    idx1 = np.where(y_multi == 1)[0]
    idx2 = np.where(y_multi == 2)[0]
    # Balance 1:1:1 respecto a C1 (clase objetivo), con fallback robusto
    n_target = max(1, len(idx1))
    s0 = rng.choice(idx0, size=min(n_target, len(idx0)), replace=False) if len(idx0) else np.empty(0, dtype=np.int64)
    s1 = idx1
    s2 = rng.choice(idx2, size=min(n_target, len(idx2)), replace=False) if len(idx2) else np.empty(0, dtype=np.int64)
    idx_sel = np.concatenate([s0, s1, s2])
    y_sel = y_multi[idx_sel].astype(np.int64)
    w_sel = np.ones(len(idx_sel), dtype=np.float32)
    return idx_sel, y_sel, w_sel


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
    k_c1: int = 2,
    k_c2: int = 12,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(0)

    n_ventanas = len(senal_rx) - long_ventana + 1
    if n_ventanas <= 0:
        C = IN_CHANNELS[representacion]
        return np.empty((0, long_ventana, C), np.float32), np.empty(0, np.int64), np.empty(0, np.float32)

    idx_ini = np.arange(n_ventanas, dtype=np.int64)
    centros = idx_ini + diana
    ref = _referencia_ventana(centros, idx_ini, modo_label)

    dist_sel = None
    if modo_label in ("onset_centro", "ventana_llena"):
        y_true, hard_neg_mask = _etiquetar_binario(ref, instantes_llegada, tolerancia, hard_neg_radius)
        idx_pos = np.where(y_true)[0]
        idx_neg_hard = np.where(hard_neg_mask)[0]
        idx_neg_rest = np.where(~y_true & ~hard_neg_mask)[0]
        idx_sel, y_sel, w_sel = _balancear_binario(idx_pos, idx_neg_hard, idx_neg_rest, ratio_undersampling, rng)
        if len(idx_neg_hard) > 0:
            # aplica weight configurado en esta llamada
            mask_hard = np.isin(idx_sel[len(idx_pos):], idx_neg_hard)
            w_sel[len(idx_pos):][mask_hard] = float(hard_neg_weight)
    else:
        y_multi, dmin = _etiquetar_multiclase(ref, instantes_llegada, k_c1, k_c2)
        idx_sel, y_sel, w_sel = _balancear_multiclase(y_multi, rng)
        dist_sel = dmin[idx_sel].astype(np.float32)

    todas = sliding_window_view(senal_rx, window_shape=long_ventana)
    ventanas_sel = todas[idx_sel]
    X = _construir_canales(ventanas_sel, representacion)
    return X, y_sel, w_sel, dist_sel


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
    k_c1: int = 2,
    k_c2: int = 12,
):
    if representacion not in REPRESENTACIONES_VALIDAS:
        raise ValueError(f"representacion debe ser una de {REPRESENTACIONES_VALIDAS}")
    if modo_label not in MODOS_LABEL_VALIDOS:
        raise ValueError(f"modo_label debe ser uno de {MODOS_LABEL_VALIDOS}")

    rng = np.random.default_rng(semilla_base)
    n_total = n_escenarios_train + n_escenarios_val

    print("Dataset")
    print(f"  representacion={representacion}  modo_label={modo_label}")
    print(f"  canales={IN_CHANNELS[representacion]}  train/val={n_escenarios_train}/{n_escenarios_val}")
    if modo_label == "multiclase_onset":
        print(f"  multiclase: C1<=|d|{k_c1}, C2<=|d|{k_c2}, resto C0")
    else:
        print(f"  binario: ratio={ratio_undersampling} hard_neg_radius={hard_neg_radius} hard_neg_weight={hard_neg_weight}")

    X_tr, Y_tr, W_tr, X_vl, Y_vl, W_vl = [], [], [], [], [], []
    D_tr, D_vl = [], []

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
                Xk, Yk, Wk, Dk = _extraer_ventanas_y_etiquetas(
                    senal_rx=esc["senal_rx"],
                    instantes_llegada=esc["instantes_llegada_muestras"],
                    representacion=representacion,
                    modo_label=modo_label,
                    ratio_undersampling=ratio_undersampling,
                    hard_neg_radius=hard_neg_radius,
                    hard_neg_weight=hard_neg_weight,
                    k_c1=k_c1,
                    k_c2=k_c2,
                    rng=rng,
                )
                if len(Xk) == 0:
                    continue
                if k < n_escenarios_train:
                    X_tr.append(Xk); Y_tr.append(Yk); W_tr.append(Wk)
                    if modo_label == "multiclase_onset":
                        D_tr.append(Dk)
                else:
                    X_vl.append(Xk); Y_vl.append(Yk); W_vl.append(Wk)
                    if modo_label == "multiclase_onset":
                        D_vl.append(Dk)

    X_train = np.concatenate(X_tr, axis=0)
    Y_train = np.concatenate(Y_tr, axis=0)
    W_train = np.concatenate(W_tr, axis=0)
    X_val = np.concatenate(X_vl, axis=0)
    Y_val = np.concatenate(Y_vl, axis=0)
    W_val = np.concatenate(W_vl, axis=0)

    idx_tr = rng.permutation(len(X_train))
    idx_vl = rng.permutation(len(X_val))
    X_train, Y_train, W_train = X_train[idx_tr], Y_train[idx_tr], W_train[idx_tr]
    X_val, Y_val, W_val = X_val[idx_vl], Y_val[idx_vl], W_val[idx_vl]

    D_train = None
    D_val = None
    if modo_label == "multiclase_onset":
        D_train = np.concatenate(D_tr, axis=0)[idx_tr]
        D_val = np.concatenate(D_vl, axis=0)[idx_vl]

    if modo_label == "multiclase_onset":
        cls_tr = [int(np.sum(Y_train == c)) for c in (0, 1, 2)]
        cls_vl = [int(np.sum(Y_val == c)) for c in (0, 1, 2)]
        print(f"  train class counts [C0,C1,C2]={cls_tr}")
        print(f"  val   class counts [C0,C1,C2]={cls_vl}")
        # Diagnóstico clave: distribución de |d| dentro de C2 para detectar sesgos
        c2_tr = D_train[Y_train == 2]
        c2_vl = D_val[Y_val == 2]
        if len(c2_tr) > 0:
            print(f"  C2 |d| train: min={np.min(c2_tr):.1f} p25={np.percentile(c2_tr,25):.1f} "
                  f"p50={np.percentile(c2_tr,50):.1f} p75={np.percentile(c2_tr,75):.1f} max={np.max(c2_tr):.1f}")
        if len(c2_vl) > 0:
            print(f"  C2 |d| val  : min={np.min(c2_vl):.1f} p25={np.percentile(c2_vl,25):.1f} "
                  f"p50={np.percentile(c2_vl,50):.1f} p75={np.percentile(c2_vl,75):.1f} max={np.max(c2_vl):.1f}")
    else:
        print(f"  train positivos={int(Y_train.sum())} ({100*Y_train.mean():.2f}%)")
        print(f"  val   positivos={int(Y_val.sum())} ({100*Y_val.mean():.2f}%)")

    return X_train, Y_train, W_train, X_val, Y_val, W_val, D_train, D_val


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
    k_c1: int = 2,
    k_c2: int = 12,
):
    os.makedirs(directorio_salida, exist_ok=True)
    X_train, Y_train, W_train, X_val, Y_val, W_val, D_train, D_val = generar_dataset_desde_escenarios(
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
        k_c1=k_c1,
        k_c2=k_c2,
    )

    for name, arr in (("X_train", X_train), ("Y_train", Y_train), ("W_train", W_train),
                      ("X_val", X_val), ("Y_val", Y_val), ("W_val", W_val)):
        np.save(os.path.join(directorio_salida, f"{name}.npy"), arr)
    if modo_label == "multiclase_onset":
        np.save(os.path.join(directorio_salida, "D_train.npy"), D_train)
        np.save(os.path.join(directorio_salida, "D_val.npy"), D_val)
        print("Guardado diagnóstico multiclase: D_train.npy / D_val.npy (|d| a onset más cercano)")
    print(f"Dataset guardado en: {directorio_salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dataset ALOHA para fase experimental.")
    parser.add_argument("--representacion", type=str, default="iq", choices=list(REPRESENTACIONES_VALIDAS))
    parser.add_argument("--modo_label", type=str, default="onset_centro", choices=list(MODOS_LABEL_VALIDOS))
    parser.add_argument("--salida", type=str, default=None)
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--ventana_ft", type=int, default=50)
    parser.add_argument("--ratio_undersample", type=int, default=RATIO_UNDERSAMPLING)
    parser.add_argument("--semilla", type=int, default=SEMILLA_BASE)
    parser.add_argument("--hard_neg_radius", type=int, default=HARD_NEG_RADIUS)
    parser.add_argument("--hard_neg_weight", type=float, default=HARD_NEG_WEIGHT)
    parser.add_argument("--k_c1", type=int, default=2, help="Multiclase: radio de clase C1 (onset)")
    parser.add_argument("--k_c2", type=int, default=12, help="Multiclase: radio máximo de clase C2 (borde)")
    parser.add_argument("--usar_preambulo", action="store_true", default=False)
    args = parser.parse_args()

    salida = args.salida or f"data/fase_{args.representacion}_{args.modo_label}"
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
        k_c1=args.k_c1,
        k_c2=args.k_c2,
    )
