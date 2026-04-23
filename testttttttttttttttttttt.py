import matplotlib.pyplot as plt
from pipeline.escenario_phy import generar_escenario_phy, ejecutar_receptor_correlador

# 1) Escenario Pure ALOHA real (no hardcode)
esc = generar_escenario_phy(
    carga_G=0.2,
    ventana_frame_times=50,
    snr_db=0.0,
    semilla=123,
    num_bits_pre=13,
    num_bits_datos=20,
)

# 2) Receptor correlador
sal = ejecutar_receptor_correlador(
    esc,
    tau=0.65,
    separacion_minima=len(esc["preambulo"]),
)

corr_norm = sal["corr_norm"]
instantes = esc["instantes_llegada_muestras"]
detectados = sal["instantes_detectados"]

print("Instantes reales:", instantes.tolist())
print("Detectados:", detectados.tolist())

# 3) Plot de corr_norm + marcas de verdad terreno
plt.figure(figsize=(12,4))
plt.plot(corr_norm, label="corr_norm")
for t in instantes:
    plt.axvline(t, linestyle="--", alpha=0.5, color="green")
for d in detectados:
    plt.plot(d, corr_norm[d], "r*")
plt.title("Correlación en escenario Pure ALOHA")
plt.xlabel("Muestra")
plt.ylabel("|C|/L")
plt.legend(["corr_norm", "instante real", "detección"])
plt.grid(alpha=0.3)
plt.show()