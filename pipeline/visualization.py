import os 
import numpy as np
import matplotlib.pyplot as plt
from aloha.pure_aloha import throughput_teorico

# ==========================================
# GRÁFICAS CAPA MAC (Tráfico Pure ALOHA)
# ==========================================
def plot_pure_aloha(G_valores, S_sim, FIGURES):
    G_fino = [g / 100 for g in range(1, 301)]
    S_fino = [throughput_teorico(g) for g in G_fino]
    G_opt, S_max = 0.5, throughput_teorico(0.5)

    plt.figure(figsize=(9, 5))
    plt.plot(G_fino, S_fino, color='steelblue', linewidth=2,
             label='Teórico: $S = G e^{-2G}$')
    plt.scatter(G_valores, S_sim, color='coral', s=40, zorder=5,
                label='Simulación Monte Carlo')
    plt.axvline(x=G_opt, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=S_max, color='gray', linestyle='--', linewidth=1)
    plt.annotate(f'G={G_opt}, S≈{S_max:.2f}',
                 xy=(G_opt, S_max), xytext=(G_opt + 0.15, S_max + 0.01),
                 fontsize=9, color='dimgray')
    plt.xlabel('G (intentos por frame time)')
    plt.ylabel('S (throughput por frame time)')
    plt.title('Pure ALOHA — Throughput teórico vs. simulado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 3.0)
    plt.ylim(0, 0.25)
    plt.tight_layout()

    ruta = os.path.join(FIGURES, 'pure_aloha.png')
    plt.savefig(ruta, dpi=150)
    plt.show()
    print(f"Gráfica guardada en {ruta}")

# ==========================================
# GRÁFICAS CAPA PHY (Señales y Correlador)
# ==========================================

def plot_deteccion_correlador_awgn(señal_rx, corr_norm, instante_llegada,
                         len_preambulo, tau=0.7, SNR_dB=None):
    """
    Genera la figura de dos subplots:
      - Arriba : señal recibida en el tiempo (solo se ve ruido)
      - Abajo  : salida del correlador con pico de detección y umbral τ
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    titulo = f"Pipeline del Correlador — SNR = {SNR_dB} dB" if SNR_dB else \
             "Pipeline del Correlador"
    fig.suptitle(titulo, fontsize=13)
 
    # --- subplot 1: señal recibida ---
    ax1.plot(señal_rx, color='steelblue', linewidth=0.7, alpha=0.85)
    ax1.axvline(x=instante_llegada, color='green', linestyle='--',
                linewidth=1, label=f'Inicio paquete (muestra {instante_llegada})')
    ax1.axvline(x=instante_llegada + len_preambulo, color='orange',
                linestyle='--', linewidth=1, label='Fin preámbulo')
    ax1.set_ylabel('Amplitud')
    ax1.set_title('Señal recibida — el paquete queda oculto en el ruido')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
 
    # --- subplot 2: salida del correlador ---
    muestras_corr = np.arange(len(corr_norm))
    ax2.plot(muestras_corr, corr_norm, color='coral', linewidth=0.9,
             alpha=0.9, label='Correlación normalizada')
 
    # umbral de detección
    ax2.axhline(y=tau, color='red', linestyle=':', linewidth=1.5,
                label=f'Umbral τ = {tau}')
 
    # pico esperado (posición teórica)
    ax2.axvline(x=instante_llegada, color='green', linestyle='--',
                linewidth=1, label=f'Pico esperado (muestra {instante_llegada})')
 
    # pico real detectado
    pico_idx = np.argmax(corr_norm)
    pico_val = corr_norm[pico_idx]
    ax2.plot(pico_idx, pico_val, 'r*', markersize=12,
             label=f'Pico detectado: {pico_val:.3f} (muestra {pico_idx})')
 
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('Magnitud normalizada')
    ax2.set_xlabel('Muestras')
    ax2.set_title('Salida del correlador — el pico sobresale del suelo de ruido')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('results/figures/correlador_resultado.png', dpi=150)
    plt.show()
    print("Gráfica guardada en results/figures/correlador_resultado.png")


# ============================================================
# AÑADIR a pipeline/visualization.py
# ============================================================

def plot_colision_correlador(senal_rx, corr_norm, instantes_reales,
                              len_preambulo, tau=0.7, SNR_dB=None):
    """
    Visualiza una colisión PHY y la respuesta del correlador.

    Subplot superior : señal recibida ruidosa con marcadores de inicio
                       de cada paquete real.
    Subplot inferior : salida del correlador normalizada, umbral τ y
                       estrellas en los picos que superan el umbral.

    Parámetros
    ----------
    senal_rx        : array — señal recibida del canal (compuesta + ruido)
    corr_norm       : array — salida normalizada del correlador
    instantes_reales: list of int — muestras donde empieza cada paquete
    len_preambulo   : int — longitud L del preámbulo (para marcar su fin)
    tau             : float — umbral de detección (default 0.7)
    SNR_dB          : float o None — para el título

    Decisión de diseño:
    -------------------
    Los picos se detectan con un mínimo de separación entre ellos igual
    a len_preambulo//2 para evitar contar el mismo pico dos veces cuando
    el correlador sube y baja alrededor del máximo real.
    """
    colores_paquetes = ['#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

    titulo = (f"Colisión Pure ALOHA — Capa PHY  |  SNR = {SNR_dB} dB"
              if SNR_dB is not None else "Colisión Pure ALOHA — Capa PHY")
    fig.suptitle(titulo, fontsize=13, fontweight='bold')

    # ------------------------------------------------------------------
    # Subplot 1 — señal recibida
    # ------------------------------------------------------------------
    ax1.plot(senal_rx, color='steelblue', linewidth=0.6, alpha=0.8,
             label='Señal recibida (ruido + paquetes solapados)')

    for i, t in enumerate(instantes_reales):
        color = colores_paquetes[i % len(colores_paquetes)]
        ax1.axvline(x=t, color=color, linestyle='--', linewidth=1.4,
                    label=f'Paquete {i+1}: inicio muestra {t}')
        ax1.axvline(x=t + len_preambulo, color=color, linestyle=':',
                    linewidth=0.9, alpha=0.6,
                    label=f'Paquete {i+1}: fin preámbulo ({t+len_preambulo})')

    # región de solapamiento (si hay exactamente 2 paquetes)
    if len(instantes_reales) == 2:
        t1, t2 = sorted(instantes_reales)
        solap_inicio = t2
        solap_fin    = t1 + len_preambulo  # fin del preámbulo del primero
        if solap_fin > solap_inicio:
            ax1.axvspan(solap_inicio, solap_fin, alpha=0.12, color='red',
                        label=f'Zona de solapamiento ({solap_fin-solap_inicio} muestras)')

    ax1.set_ylabel('Amplitud')
    ax1.set_title('Señal recibida — los paquetes quedan ocultos en el ruido')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Subplot 2 — salida del correlador
    # ------------------------------------------------------------------
    ax2.plot(corr_norm, color='coral', linewidth=0.8, alpha=0.9,
             label='Correlación normalizada |Cµ| / L')

    # umbral
    ax2.axhline(y=tau, color='red', linestyle=':', linewidth=1.6,
                label=f'Umbral τ = {tau}')

    # picos que superan el umbral con separación mínima
    picos = _encontrar_picos(corr_norm, tau, separacion_min=len_preambulo // 2)

    for i, idx in enumerate(picos):
        ax2.plot(idx, corr_norm[idx], '*', markersize=14,
                 color=colores_paquetes[i % len(colores_paquetes)],
                 zorder=5,
                 label=f'Detección {i+1}: muestra {idx} '
                       f'(valor={corr_norm[idx]:.3f})')

    # marcadores de posición real
    for i, t in enumerate(instantes_reales):
        color = colores_paquetes[i % len(colores_paquetes)]
        ax2.axvline(x=t, color=color, linestyle='--',
                    linewidth=1.2, alpha=0.5)

    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel('Magnitud normalizada')
    ax2.set_xlabel('Muestras')
    ax2.set_title('Salida del correlador — detección de picos sobre umbral τ')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()

    ruta = 'results/figures/colision_correlador.png'
    plt.savefig(ruta, dpi=150)
    plt.show()
    print(f"Gráfica guardada en {ruta}")

    # resumen por consola
    print(f"\nResumen detección:")
    print(f"  Paquetes reales en:  {instantes_reales}")
    print(f"  Picos detectados en: {list(picos)}")
    for i, t_real in enumerate(instantes_reales):
        detectado = any(abs(p - t_real) <= len_preambulo for p in picos)
        print(f"  Paquete {i+1} (muestra {t_real}): "
              f"{'✓ DETECTADO' if detectado else '✗ PERDIDO'}")


def _encontrar_picos(corr_norm, tau, separacion_min=5):
    """
    Encuentra picos en corr_norm que superen tau con separación mínima
    entre ellos. Evita contar el mismo pico varias veces.

    Retorna lista de índices de picos ordenados por valor descendente.
    """
    candidatos = np.where(corr_norm >= tau)[0]
    if len(candidatos) == 0:
        return []

    picos = []
    ultimo = -separacion_min - 1

    # recorre los candidatos en orden y filtra los demasiado cercanos
    for idx in candidatos:
        if idx - ultimo >= separacion_min:
            # toma el máximo local en una ventana alrededor de idx
            ventana_inicio = max(0, idx - separacion_min)
            ventana_fin    = min(len(corr_norm), idx + separacion_min)
            idx_max = ventana_inicio + np.argmax(
                corr_norm[ventana_inicio:ventana_fin]
            )
            if not picos or idx_max != picos[-1]:
                picos.append(idx_max)
                ultimo = idx_max

    return picos


def plot_roc_correlador(fpr, tpr, auc=None, ruta_salida=None):
    """
    Dibuja la curva ROC (TPR vs FPR) para el correlador.
    """
    fpr = np.asarray(fpr, dtype=float).ravel()
    tpr = np.asarray(tpr, dtype=float).ravel()
    if fpr.size != tpr.size:
        raise ValueError("fpr y tpr deben tener la misma longitud")

    orden = np.argsort(fpr)
    fpr_ord = fpr[orden]
    tpr_ord = tpr[orden]

    plt.figure(figsize=(7, 6))
    label = "ROC correlador"
    if auc is not None:
        label = f"ROC correlador (AUC={float(auc):.3f})"

    plt.plot(fpr_ord, tpr_ord, color="royalblue", linewidth=2.0, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Azar")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC del correlador (por índice)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if ruta_salida is not None:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=150)
        print(f"Gráfica ROC guardada en {ruta_salida}")
    plt.show()


def plot_roc_familia_por_snr(curvas_por_snr, ruta_salida=None, carga_G=None):
    """
    Dibuja una familia de curvas ROC para varios SNR a carga G fija.
    """
    plt.figure(figsize=(7, 6))
    for snr_db, curva in sorted(curvas_por_snr.items(), key=lambda x: float(x[0])):
        fpr = np.asarray(curva["fpr"], dtype=float).ravel()
        tpr = np.asarray(curva["tpr"], dtype=float).ravel()
        orden = np.argsort(fpr)
        auc_txt = ""
        if "auc" in curva and curva["auc"] is not None:
            auc_txt = f", AUC={float(curva['auc']):.3f}"
        plt.plot(
            fpr[orden],
            tpr[orden],
            linewidth=1.8,
            label=f"SNR={float(snr_db):g} dB{auc_txt}",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Azar")
    titulo = "Familia ROC por SNR"
    if carga_G is not None:
        titulo += f" (G={float(carga_G):g})"
    plt.title(titulo)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if ruta_salida is not None:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=150)
        print(f"Familia ROC guardada en {ruta_salida}")
    plt.show()


def plot_roc_4panel_comparativa(curvas_por_condicion, ruta_salida=None):
    """
    Figura única 2×2 con curvas ROC del correlador vs CNN para 4 condiciones (G,SNR).

    Parámetros
    ----------
    curvas_por_condicion : dict con claves (G, SNR) y valores dict:
        {
          'fpr_corr', 'tpr_corr', 'auc_corr',
          'fpr_ml',   'tpr_ml',   'auc_ml',
          'titulo': str  (etiqueta del subplot)
        }
    ruta_salida : str opcional — ruta donde guardar la figura.
    """
    condiciones = list(curvas_por_condicion.keys())
    if len(condiciones) != 4:
        raise ValueError("Se necesitan exactamente 4 condiciones para la figura 2×2.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for ax, cond in zip(axes_flat, condiciones):
        c = curvas_por_condicion[cond]

        fpr_corr = np.asarray(c["fpr_corr"], dtype=float).ravel()
        tpr_corr = np.asarray(c["tpr_corr"], dtype=float).ravel()
        fpr_ml   = np.asarray(c["fpr_ml"],   dtype=float).ravel()
        tpr_ml   = np.asarray(c["tpr_ml"],   dtype=float).ravel()

        # Ordenar por FPR ascendente para plot correcto
        ord_corr = np.argsort(fpr_corr)
        ord_ml   = np.argsort(fpr_ml)

        ax.plot(fpr_corr[ord_corr], tpr_corr[ord_corr],
                linewidth=2.0, color="royalblue",
                label=f"Correlador (AUC={float(c['auc_corr']):.3f})")
        ax.plot(fpr_ml[ord_ml], tpr_ml[ord_ml],
                linewidth=2.0, color="darkorange",
                label=f"CNN 1D (AUC={float(c['auc_ml']):.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
                linewidth=1.0, label="Azar")

        ax.set_title(c.get("titulo", str(cond)), fontsize=11)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Comparativa ROC: Correlador vs CNN 1D (protocolo congelado)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if ruta_salida is not None:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=150)
        print(f"Figura 4-panel ROC guardada en {ruta_salida}")
    plt.show()


def plot_roc_comparativa_correlador_vs_ml(
    fpr_corr,
    tpr_corr,
    auc_corr,
    fpr_ml,
    tpr_ml,
    auc_ml,
    ruta_salida=None,
    carga_G=None,
    snr_db=None,
):
    """
    Dibuja en la misma figura la ROC del correlador y la del detector ML.
    """
    fpr_corr = np.asarray(fpr_corr, dtype=float).ravel()
    tpr_corr = np.asarray(tpr_corr, dtype=float).ravel()
    fpr_ml = np.asarray(fpr_ml, dtype=float).ravel()
    tpr_ml = np.asarray(tpr_ml, dtype=float).ravel()

    if fpr_corr.size != tpr_corr.size:
        raise ValueError("fpr_corr y tpr_corr deben tener la misma longitud")
    if fpr_ml.size != tpr_ml.size:
        raise ValueError("fpr_ml y tpr_ml deben tener la misma longitud")

    ord_corr = np.argsort(fpr_corr)
    ord_ml = np.argsort(fpr_ml)

    plt.figure(figsize=(7.5, 6.2))
    plt.plot(
        fpr_corr[ord_corr],
        tpr_corr[ord_corr],
        linewidth=2.0,
        color="royalblue",
        label=f"Correlador (AUC={float(auc_corr):.3f})",
    )
    plt.plot(
        fpr_ml[ord_ml],
        tpr_ml[ord_ml],
        linewidth=2.0,
        color="darkorange",
        label=f"Red neuronal (AUC={float(auc_ml):.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Azar")

    titulo = "Comparativa ROC: correlador vs red neuronal"
    if carga_G is not None and snr_db is not None:
        titulo += f" (G={float(carga_G):g}, SNR={float(snr_db):g} dB)"
    plt.title(titulo)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if ruta_salida is not None:
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=150)
        print(f"Comparativa ROC guardada en {ruta_salida}")
    plt.show()