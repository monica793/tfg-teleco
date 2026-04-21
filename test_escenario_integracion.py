"""
Prueba mínima (smoke): escenario PHY + correlador + métricas sin errores.

Ejecutar: python -m unittest test_escenario_integracion -v
"""
import unittest

import numpy as np

from pipeline.escenario_phy import (
    ejecutar_receptor_correlador,
    generar_escenario_phy,
)
from pipeline.metricas_receptor import evaluar_detecciones


class TestEscenarioIntegracion(unittest.TestCase):
    def test_smoke_escenario_y_correlador(self):
        esc = generar_escenario_phy(
            carga_G=0.35,
            ventana_frame_times=120,
            snr_db=8.0,
            semilla=123,
            num_bits_pre=13,
            num_bits_datos=20,
        )
        self.assertEqual(esc["senal_rx"].ndim, 1)
        self.assertEqual(esc["senal_rx"].shape[0], esc["longitud_total"])

        salida = ejecutar_receptor_correlador(
            esc,
            tau=0.45,
            separacion_minima=len(esc["preambulo"]),
        )
        metricas = evaluar_detecciones(
            esc["instantes_llegada_muestras"],
            salida["instantes_detectados"],
            tolerancia_muestras=5,
        )
        self.assertEqual(metricas["tp"] + metricas["fn"], metricas["num_verdaderos"])
        self.assertEqual(metricas["tp"] + metricas["fp"], metricas["num_detectados"])
        self.assertTrue(np.isfinite(salida["corr_norm"]).all())


if __name__ == "__main__":
    unittest.main()
