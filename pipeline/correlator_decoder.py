import numpy as np 

def correlador(señal_rx, preambulo):
    """
    Correlación cruzada normalizada entre la señal recibida y el preámbulo.
 
    En un caso ideal (sin ruido, paquete alineado), el pico máximo vale 1.0.
    En presencia de ruido el pico baja, y el suelo de correlación sube.
 
    Parámetros
    ----------
    señal_rx  : señal recibida del canal (ruidosa)
    preambulo : plantilla del preámbulo conocida por el receptor
 
    Retorna
    -------
    corr_norm : array con la correlación normalizada (valores entre ~0 y ~1)
 
    Uso del umbral:
        detecciones = np.where(corr_norm >= tau)[0]
    """
    L = len(preambulo)
 
    # np.correlate en modo 'full' devuelve longitud len(señal) + len(preámbulo) - 1
    # usamos 'valid' para obtener solo las posiciones donde el preámbulo cabe entero
    corr = np.correlate(señal_rx, preambulo, mode='valid')
 
    # Normalización: divide por L para que el pico ideal sea exactamente 1
    corr_norm = np.abs(corr) / L
 
    return corr_norm