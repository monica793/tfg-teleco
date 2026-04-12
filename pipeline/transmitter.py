import numpy as np
def generar_preambulo(num_bits=13, semilla=42):
    """
    Genera un preámbulo conocido modulado en BPSK.
 
    Parámetros
    ----------
    num_bits : longitud del preámbulo en símbolos (recomendado: primo, ej. 13)
    semilla  : semilla fija para que el preámbulo sea siempre el mismo
 
    Retorna
    -------
    preambulo : array de +1/-1 de longitud num_bits
 
    Nota: transmisor y receptor usan la misma semilla para conocer la secuencia.
    """
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=num_bits)   # bits aleatorios 0/1
    preambulo = 2 * bits - 1                    # BPSK: 0 → -1, 1 → +1
    return preambulo
 
 
def generar_paquete(preambulo, num_bits_datos=20):
    """
    Construye un paquete completo: [preámbulo | datos aleatorios].
 
    Parámetros
    ----------
    preambulo      : array BPSK del preámbulo (conocido)
    num_bits_datos : longitud de la parte de datos (desconocida para el receptor)
 
    Retorna
    -------
    paquete : array [preámbulo | datos] en BPSK
    """
    datos = np.random.choice([-1, 1], size=num_bits_datos)
    paquete = np.concatenate([preambulo, datos])
    return paquete