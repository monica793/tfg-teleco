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