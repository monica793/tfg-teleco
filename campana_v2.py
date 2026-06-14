"""
campana_v2.py — Orquestador de la campaña experimental completa (9 modelos).

Genera datasets, entrena, evalúa en test de clasificación y en test de
detección (PHY Monte Carlo), y guarda tablas CSV + figuras para la memoria.

Uso en Colab:
    python campana_v2.py --modo todo
    python campana_v2.py --modo generar
    python campana_v2.py --modo entrenar
    python campana_v2.py --modo evaluar

Para un solo experimento piloto:
    python campana_v2.py --modo todo --rep iq --label multiclase_onset

Estructura de salida:
    data_v2/<rep>_<label>/          ← datasets .npy + metadata.json
    checkpoints_v2/<rep>_<label>/   ← mejor.ckpt + curvas_entrenamiento.png
    results_v2/
        figures/entrenamiento/      ← curvas PNG por experimento
        figures/seleccion/          ← comparativa 9 modelos
        figures/comparativa_final/  ← correlador vs campeón
        tables/seleccion_cnn.csv
        tables/comparativa_final.csv
"""

import argparse
import os
import json
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuración de la campaña
# ---------------------------------------------------------------------------

REPRESENTACIONES = ["energia", "iq", "iq_energia"]
MODOS_LABEL      = ["onset_centro", "ventana_llena", "multiclase_onset"]
NUM_CLASES_MAP   = {
    "onset_centro":     1,
    "ventana_llena":    1,
    "multiclase_onset": 3,
}
# ref_diana para inferencia: debe coincidir con el modo de etiquetado
REF_DIANA_MAP = {
    "onset_centro":     "centro",
    "ventana_llena":    "inicio",   # ← fix del bug previo
    "multiclase_onset": "centro",
}

# Hiperparámetros de generación (700/200/200 por celda G×SNR)
N_TRAIN         = 700
N_VAL           = 200
N_TEST          = 200
VENTANA_FT      = 80    # 80 duraciones de paquete → 80×128 = 10240 muestras/escenario
LISTA_G         = (0.2, 0.4, 0.6, 0.8)
LISTA_SNR       = (0.0, 3.0, 6.0, 10.0)

# Hiperparámetros de entrenamiento
MAX_EPOCHS      = 100
BATCH_SIZE      = 512
LR              = 1e-3
DROPOUT         = 0.3
PATIENCE        = 12    # EarlyStopping

# Rutas
DATA_DIR        = "data_v2"
CKPT_DIR        = "checkpoints_v2"
RESULTS_DIR     = "results_v2"
PROYECTO_WANDB  = "tfg-aloha-detector-v2"

# Protocolo de evaluación (mismos que protocolo_evaluacion.py)
TOLERANCIA_MC   = 4
N_ITER_MC       = 50    # iteraciones Monte Carlo para ROC de detección
SEMILLA_MC      = 999   # semilla distinta a la del dataset

# Escenario objetivo para selección de campeón
G_OBJ, SNR_OBJ  = 0.4, 6.0

os.makedirs(os.path.join(RESULTS_DIR, "figures", "entrenamiento"),  exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures", "seleccion"),      exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures", "comparativa_final"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nombre_exp(rep, label):
    return f"{rep}_{label}"


def _ruta_datos(rep, label):
    return os.path.join(DATA_DIR, _nombre_exp(rep, label))


def _ruta_ckpt(rep, label):
    return os.path.join(CKPT_DIR, _nombre_exp(rep, label), "mejor.ckpt")


def _ruta_curvas(rep, label):
    return os.path.join(RESULTS_DIR, "figures", "entrenamiento",
                        f"{_nombre_exp(rep, label)}.png")


def _existe_dataset(rep, label):
    d = _ruta_datos(rep, label)
    return os.path.exists(os.path.join(d, "X_train.npy"))


def _existe_ckpt(rep, label):
    return os.path.exists(_ruta_ckpt(rep, label))


# ---------------------------------------------------------------------------
# 1. GENERAR DATASETS
# ---------------------------------------------------------------------------

def generar_todos(reps=None, labels=None, forzar=False):
    """Genera los 9 datasets (o los seleccionados) y los guarda en data_v2/."""
    from ml.generar_dataset import guardar_dataset

    reps   = reps   or REPRESENTACIONES
    labels = labels or MODOS_LABEL

    for rep in reps:
        for label in labels:
            nombre = _nombre_exp(rep, label)
            ruta   = _ruta_datos(rep, label)
            if _existe_dataset(rep, label) and not forzar:
                print(f"[SKIP] Dataset ya existe: {nombre}")
                continue
            print(f"\n{'='*60}")
            print(f"GENERANDO dataset: {nombre}")
            print(f"{'='*60}")
            guardar_dataset(
                directorio_salida=ruta,
                representacion=rep,
                modo_label=label,
                n_escenarios_train=N_TRAIN,
                n_escenarios_val=N_VAL,
                n_escenarios_test=N_TEST,
                lista_G=LISTA_G,
                lista_SNR_dB=LISTA_SNR,
                ventana_frame_times=VENTANA_FT,
            )


# ---------------------------------------------------------------------------
# 2. ENTRENAR MODELOS
# ---------------------------------------------------------------------------

def entrenar_todos(reps=None, labels=None, usar_wandb=True, forzar=False):
    """Entrena los 9 modelos (o los seleccionados)."""
    from ml.entrenar_modelo import entrenar

    reps   = reps   or REPRESENTACIONES
    labels = labels or MODOS_LABEL

    for rep in reps:
        for label in labels:
            nombre = _nombre_exp(rep, label)
            if _existe_ckpt(rep, label) and not forzar:
                print(f"[SKIP] Checkpoint ya existe: {nombre}")
                continue
            if not _existe_dataset(rep, label):
                print(f"[ERROR] Dataset no encontrado para {nombre}, genera primero.")
                continue
            print(f"\n{'='*60}")
            print(f"ENTRENANDO: {nombre}")
            print(f"{'='*60}")
            entrenar(
                directorio_datos=_ruta_datos(rep, label),
                representacion=rep,
                modo_label=label,
                num_clases=NUM_CLASES_MAP[label],
                directorio_ckpt=CKPT_DIR,
                ruta_curvas=_ruta_curvas(rep, label),
                max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                dropout=DROPOUT,
                usar_wandb=usar_wandb,
                proyecto_wandb=PROYECTO_WANDB,
                nombre_run_wandb=nombre,
            )


# ---------------------------------------------------------------------------
# 3. EVALUAR — test de clasificación (ventanas)
# ---------------------------------------------------------------------------

def evaluar_clasificacion(rep, label):
    """
    Evalúa el modelo en X_test.npy: accuracy, F1, AUC (ventanas).
    Devuelve dict con métricas.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from ml.modelo_fase1 import cargar_checkpoint

    ruta_test_X = os.path.join(_ruta_datos(rep, label), "X_test.npy")
    ruta_test_Y = os.path.join(_ruta_datos(rep, label), "Y_test.npy")
    if not os.path.exists(ruta_test_X):
        return {"error": "sin test"}

    X = np.transpose(np.load(ruta_test_X), (0, 2, 1)).astype(np.float32)
    Y = np.load(ruta_test_Y)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint(_ruta_ckpt(rep, label), map_location=dispositivo)
    modelo.eval()

    num_clases = getattr(modelo, "num_clases", 1)
    X_t = torch.from_numpy(X).to(dispositivo)

    scores, preds = [], []
    with torch.no_grad():
        for i in range(0, len(X_t), 1024):
            logits = modelo(X_t[i:i+1024])
            if num_clases == 1:
                s = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
                p = (s >= 0.5).astype(int)
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                s = probs[:, 1]
                p = probs.argmax(axis=1)
            scores.append(s); preds.append(p)

    scores = np.concatenate(scores)
    preds  = np.concatenate(preds)

    # Etiquetas binarias C1-vs-resto para AUC
    Y_bin = (Y == 1).astype(int)

    auc  = float(roc_auc_score(Y_bin, scores)) if len(np.unique(Y_bin)) > 1 else float("nan")
    acc  = float(accuracy_score(Y, preds))
    # F1 binario: C1 vs resto
    f1   = float(f1_score(Y_bin, (preds == 1).astype(int), zero_division=0))

    return {"test_auc_clf": auc, "test_acc": acc, "test_f1_clf": f1,
            "n_test": int(len(Y))}


# ---------------------------------------------------------------------------
# 4. EVALUAR — test de detección (Monte Carlo PHY)
# ---------------------------------------------------------------------------

def evaluar_deteccion(rep, label, g=G_OBJ, snr=SNR_OBJ):
    """
    Evalúa el modelo en escenarios PHY nuevos (Monte Carlo).
    Devuelve AUC-PR, AUC-ROC, F1 best, TP/FP/FN.
    """
    from ml.modelo_fase1 import cargar_checkpoint
    from ml.evaluar import tabla_metricas
    from pipeline.escenario_phy import generar_escenario_phy, ejecutar_receptor_neuronal
    from pipeline.protocolo_evaluacion import (
        NUM_BITS_PRE, NUM_BITS_DATOS, TOLERANCIA_MUESTRAS, LONG_VENTANA_CNN,
    )

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint(_ruta_ckpt(rep, label), map_location=dispositivo)
    modelo.eval()

    ref_diana = REF_DIANA_MAP[label]
    taus = np.linspace(0.0, 1.0, 201)

    pr_aucs, roc_aucs, f1s = [], [], []
    for it in range(N_ITER_MC):
        esc = generar_escenario_phy(
            carga_G=g,
            ventana_frame_times=300,
            snr_db=snr,
            num_bits_pre=NUM_BITS_PRE,
            num_bits_datos=NUM_BITS_DATOS,
            semilla=SEMILLA_MC + it,
            usar_preambulo=False,
        )
        sal = ejecutar_receptor_neuronal(
            escenario=esc, modelo=modelo, umbral=0.5,
            dispositivo=dispositivo, stride=1,
            long_ventana=LONG_VENTANA_CNN, ref_diana=ref_diana,
        )
        score = np.asarray(sal["score_por_muestra"], dtype=np.float32)
        met = tabla_metricas(score, esc["instantes_llegada_muestras"],
                              TOLERANCIA_MUESTRAS, taus=taus)
        pr_aucs.append(met["pr_auc"])
        roc_aucs.append(met["roc_auc"])
        f1s.append(met["f1_best"])

    return {
        "pr_auc_det":  float(np.mean(pr_aucs)),
        "roc_auc_det": float(np.mean(roc_aucs)),
        "f1_det":      float(np.mean(f1s)),
        "n_iter_mc":   N_ITER_MC,
        "G":           g, "SNR": snr,
    }


# ---------------------------------------------------------------------------
# 5. TABLA DE SELECCIÓN (Anexo B)
# ---------------------------------------------------------------------------

def tabla_seleccion(reps=None, labels=None, usar_wandb=False):
    """
    Evalúa los 9 modelos y guarda results_v2/tables/seleccion_cnn.csv.
    Devuelve nombre del campeón (mejor AUC-PR de detección).
    """
    import csv

    reps   = reps   or REPRESENTACIONES
    labels = labels or MODOS_LABEL

    cabecera = ["representacion", "modo_label",
                "test_auc_clf", "test_acc", "test_f1_clf",
                "pr_auc_det", "roc_auc_det", "f1_det",
                "G_eval", "SNR_eval"]
    filas = []
    mejor_auc = -1.0
    campeon   = (None, None)

    for rep in reps:
        for label in labels:
            if not _existe_ckpt(rep, label):
                print(f"[SKIP] Sin checkpoint: {rep}_{label}")
                continue
            print(f"  Evaluando {rep}_{label} ...", end=" ", flush=True)
            m_clf = evaluar_clasificacion(rep, label)
            m_det = evaluar_deteccion(rep, label)
            print(f"PR-AUC_det={m_det['pr_auc_det']:.4f}  "
                  f"ROC-AUC_det={m_det['roc_auc_det']:.4f}  "
                  f"F1_det={m_det['f1_det']:.4f}")

            fila = {
                "representacion": rep,
                "modo_label":     label,
                **{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in m_clf.items() if k != "n_test"},
                **{k: round(v, 5) if isinstance(v, float) else v
                   for k, v in m_det.items() if k not in ("n_iter_mc",)},
            }
            filas.append(fila)

            # Criterio de selección: mayor AUC-PR en test de detección
            if m_det["pr_auc_det"] > mejor_auc:
                mejor_auc = m_det["pr_auc_det"]
                campeon   = (rep, label)

    ruta_csv = os.path.join(RESULTS_DIR, "tables", "seleccion_cnn.csv")
    with open(ruta_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cabecera)
        w.writeheader()
        for fila in filas:
            row = {k: fila.get(k, "") for k in cabecera}
            w.writerow(row)
    print(f"\nTabla selección guardada: {ruta_csv}")
    print(f"Campeón CNN: {campeon[0]}_{campeon[1]}  (PR-AUC det = {mejor_auc:.4f})")

    # Guardar campeón en JSON para que evaluar_comparativa lo lea
    with open(os.path.join(RESULTS_DIR, "tables", "campeon.json"), "w") as f:
        json.dump({"rep": campeon[0], "label": campeon[1], "pr_auc_det": mejor_auc}, f)

    return campeon


# ---------------------------------------------------------------------------
# 6. COMPARATIVA FINAL: correlador vs campeón CNN (§4.4)
# ---------------------------------------------------------------------------

def evaluar_comparativa_final(campeon_rep=None, campeon_label=None):
    """
    Genera figuras y tabla comparativa correlador vs campeón CNN para §4.4.
    Si no se pasa el campeón, lo lee de results_v2/tables/campeon.json.
    """
    import csv
    from ml.modelo_fase1 import cargar_checkpoint
    from ml.evaluar import tabla_metricas
    from pipeline.escenario_phy import (
        generar_escenario_phy, ejecutar_receptor_neuronal,
        ejecutar_monte_carlo_roc_correlador, ejecutar_monte_carlo_roc_neuronal,
    )
    from pipeline.correlator_decoder import correlador
    from pipeline.protocolo_evaluacion import (
        NUM_BITS_PRE, NUM_BITS_DATOS, TOLERANCIA_MUESTRAS, LONG_VENTANA_CNN,
    )
    from pipeline.visualization import (
        plot_roc_comparativa_correlador_vs_ml,
        plot_respuesta_correlador_vs_ml,
    )

    # Leer campeón si no se pasó
    if campeon_rep is None:
        ruta_json = os.path.join(RESULTS_DIR, "tables", "campeon.json")
        if not os.path.exists(ruta_json):
            raise FileNotFoundError("Ejecuta tabla_seleccion primero.")
        with open(ruta_json) as f:
            c = json.load(f)
        campeon_rep, campeon_label = c["rep"], c["label"]

    print(f"\nComparativa final: correlador vs {campeon_rep}_{campeon_label}")

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = cargar_checkpoint(_ruta_ckpt(campeon_rep, campeon_label),
                               map_location=dispositivo)
    modelo.eval()
    ref_diana = REF_DIANA_MAP[campeon_label]
    taus = np.linspace(0.0, 1.0, 201)

    resultados = {}

    for g, snr in [(0.2, 10.0), (0.4, 6.0), (0.4, 0.0), (0.8, 6.0)]:
        clave = f"G{g}_SNR{snr}"
        print(f"  Evaluando {clave} ...", end=" ", flush=True)

        # ROC correlador (Monte Carlo)
        roc_corr = ejecutar_monte_carlo_roc_correlador(
            carga_G=g, ventana_frame_times=300, snr_db=snr,
            tolerancia_muestras=TOLERANCIA_MUESTRAS,
            num_iteraciones=N_ITER_MC, semilla_base=SEMILLA_MC,
            num_bits_pre=NUM_BITS_PRE, num_bits_datos=NUM_BITS_DATOS,
            taus=taus,
        )
        # ROC CNN (Monte Carlo)
        roc_cnn = ejecutar_monte_carlo_roc_neuronal(
            carga_G=g, ventana_frame_times=300, snr_db=snr,
            tolerancia_muestras=TOLERANCIA_MUESTRAS,
            num_iteraciones=N_ITER_MC, modelo=modelo,
            semilla_base=SEMILLA_MC, num_bits_pre=NUM_BITS_PRE,
            num_bits_datos=NUM_BITS_DATOS, dispositivo=dispositivo,
            stride=1, long_ventana=LONG_VENTANA_CNN,
            ref_diana=ref_diana, taus=taus,
        )
        print(f"AUC corr={roc_corr['auc_media']:.4f}  AUC CNN={roc_cnn['auc_media']:.4f}")
        resultados[clave] = {"corr": roc_corr, "cnn": roc_cnn, "g": g, "snr": snr}

    # Figura ROC principal (G=0.4, SNR=6 dB)
    ref = resultados["G0.4_SNR6.0"]
    ruta_roc = os.path.join(RESULTS_DIR, "figures", "comparativa_final",
                            "roc_comparativa_correlador_vs_cnn.png")
    plot_roc_comparativa_correlador_vs_ml(
        fpr_corr=ref["corr"]["fpr_media"], tpr_corr=ref["corr"]["tpr_media"],
        auc_corr=ref["corr"]["auc_media"],
        fpr_ml=ref["cnn"]["fpr_media"],   tpr_ml=ref["cnn"]["tpr_media"],
        auc_ml=ref["cnn"]["auc_media"],
        ruta_salida=ruta_roc, carga_G=0.4, snr_db=6.0,
    )

    # Figura respuesta temporal (G=0.4, SNR=6 dB, una realización)
    esc_viz = generar_escenario_phy(
        carga_G=0.4, ventana_frame_times=300, snr_db=6.0,
        num_bits_pre=NUM_BITS_PRE, num_bits_datos=NUM_BITS_DATOS,
        semilla=SEMILLA_MC + 999, usar_preambulo=True,
    )
    corr_norm = correlador(esc_viz["senal_rx"], esc_viz["preambulo"])
    from pipeline.correlator_decoder import buscar_picos_preambulo
    det_corr = buscar_picos_preambulo(corr_norm, tau=0.65)
    sal_cnn = ejecutar_receptor_neuronal(
        escenario=esc_viz, modelo=modelo, umbral=0.5,
        dispositivo=dispositivo, stride=1,
        long_ventana=LONG_VENTANA_CNN, ref_diana=ref_diana,
    )
    ruta_temporal = os.path.join(RESULTS_DIR, "figures", "comparativa_final",
                                 "respuesta_temporal_correlador_vs_cnn.png")
    plot_respuesta_correlador_vs_ml(
        corr_norm=corr_norm,
        score_ml=sal_cnn["score_por_muestra"],
        instantes_reales=esc_viz["instantes_llegada_muestras"],
        detecciones_corr=det_corr,
        detecciones_ml=sal_cnn["instantes_detectados"],
        tau_corr=0.65, tau_ml=0.5,
        ruta_salida=ruta_temporal,
        titulo=f"Correlador vs CNN ({campeon_rep}/{campeon_label}) — G=0.4 SNR=6dB",
    )

    # Tabla CSV
    cabecera = ["condicion", "G", "SNR",
                "auc_corr", "auc_cnn", "delta_auc"]
    filas = []
    for clave, v in resultados.items():
        filas.append({
            "condicion": clave,
            "G":         v["g"], "SNR": v["snr"],
            "auc_corr":  round(v["corr"]["auc_media"], 4),
            "auc_cnn":   round(v["cnn"]["auc_media"],  4),
            "delta_auc": round(v["cnn"]["auc_media"] - v["corr"]["auc_media"], 4),
        })
    ruta_csv = os.path.join(RESULTS_DIR, "tables", "comparativa_final.csv")
    import csv
    with open(ruta_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cabecera)
        w.writeheader(); w.writerows(filas)
    print(f"Tabla comparativa guardada: {ruta_csv}")
    print(f"Figura ROC guardada:        {ruta_roc}")
    print(f"Figura temporal guardada:   {ruta_temporal}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Campaña experimental v2 (9 modelos).")
    parser.add_argument("--modo", choices=["todo", "generar", "entrenar",
                                           "evaluar", "comparativa"],
                        default="todo")
    parser.add_argument("--rep",   type=str, default=None,
                        help="Solo esta representación (p. ej. iq)")
    parser.add_argument("--label", type=str, default=None,
                        help="Solo este modo_label (p. ej. multiclase_onset)")
    parser.add_argument("--sin_wandb",  action="store_true")
    parser.add_argument("--forzar",     action="store_true",
                        help="Regenerar aunque ya existan datasets/checkpoints")
    args = parser.parse_args()

    reps   = [args.rep]   if args.rep   else None
    labels = [args.label] if args.label else None

    usar_wandb = not args.sin_wandb

    if args.modo in ("todo", "generar"):
        generar_todos(reps=reps, labels=labels, forzar=args.forzar)

    if args.modo in ("todo", "entrenar"):
        entrenar_todos(reps=reps, labels=labels,
                       usar_wandb=usar_wandb, forzar=args.forzar)

    if args.modo in ("todo", "evaluar"):
        tabla_seleccion(reps=reps, labels=labels)

    if args.modo in ("todo", "comparativa"):
        evaluar_comparativa_final()
