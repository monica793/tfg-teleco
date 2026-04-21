import numpy as np


def generar_preambulo_zc(longitud=13, raiz_u=1):
    """
    Genera una secuencia Zadoff-Chu compleja de módulo constante.

    Parámetros
    ----------
    longitud : int
        Longitud de la secuencia ZC.
    raiz_u : int
        Índice de raíz (debe ser coprimo con longitud para buenas propiedades).

    Retorna
    -------
    preambulo_zc : np.ndarray complejo de longitud `longitud`.
    """
    if longitud <= 1:
        raise ValueError("longitud debe ser mayor que 1")
    if np.gcd(raiz_u, longitud) != 1:
        raise ValueError("raiz_u debe ser coprimo con longitud")

    n = np.arange(longitud, dtype=np.float64)
    fase = -np.pi * raiz_u * n * (n + 1) / longitud
    preambulo_zc = np.exp(1j * fase).astype(np.complex128)
    return preambulo_zc


def generar_preambulo(num_bits=13, semilla=42, tipo="zc", raiz_u=1):
    """
    Genera un preámbulo conocido.

    Modos:
    - tipo='zc'   : secuencia Zadoff-Chu compleja (por defecto, Fase 1 I/Q).
    - tipo='bpsk' : secuencia BPSK real legado para compatibilidad.
    """
    if tipo == "zc":
        return generar_preambulo_zc(longitud=num_bits, raiz_u=raiz_u)

    if tipo == "bpsk":
        rng = np.random.default_rng(semilla)
        bits = rng.integers(0, 2, size=num_bits)
        return (2 * bits - 1).astype(np.float64)

    raise ValueError("tipo debe ser 'zc' o 'bpsk'")


def generar_paquete(preambulo, num_bits_datos=20):
    """
    Construye un paquete completo: [preámbulo | datos].

    Si el preámbulo es complejo (ZC), los datos BPSK se promocionan a complejo.
    """
    datos = np.random.choice([-1, 1], size=num_bits_datos).astype(np.float64)
    if np.iscomplexobj(preambulo):
        datos = datos.astype(np.complex128)
    paquete = np.concatenate([preambulo, datos])
    return paquete
