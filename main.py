"""
TFG — Comunicaciones digitales
Punto de entrada principal. Comenta/descomenta lo que quieras ejecutar.
"""
import os

# ── rutas ──────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(__file__)
FIGURES = os.path.join(ROOT, 'results', 'figures')
MODELS  = os.path.join(ROOT, 'models', 'trained')
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(MODELS,  exist_ok=True)

# ── imports de tus módulos ─────────────────────────────────────────────────
from aloha.pure_aloha import simular_pure_aloha, throughput_teorico
# from pipeline.correlator_decoder import simular_correlador   # cuando lo tengas
# from pipeline.nn_decoder          import cargar_y_evaluar    # cuando lo tengas

# ── helpers de visualización ───────────────────────────────────────────────
import matplotlib.pyplot as plt

def plot_pure_aloha(G_valores, S_sim):
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

# ── ejecución ──────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("=== Pure ALOHA ===")
    G_valores = [round(g * 0.1, 1) for g in range(1, 31)]
    # Ejecutamos la simulación real para cada punto de G
    S_sim = [simular_pure_aloha(g, num_frames=50_000) for g in G_valores]
    plot_pure_aloha(G_valores, S_sim)

    # print("=== Correlador ===")      # descomenta cuando llegues aquí
    # simular_correlador(...)

    # print("=== Red neuronal ===")    # descomenta cuando llegues aquí
    # cargar_y_evaluar(...)