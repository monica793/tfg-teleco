# Capítulo 3. Fundamentos Teóricos

## 3.1 Modelo de tráfico Pure ALOHA

En Pure ALOHA, los intentos de acceso al medio se modelan como un proceso de Poisson con carga ofrecida \(G\). Si observamos una ventana temporal normalizada en unidades de duración de paquete, el número de intentos en esa ventana es una variable Poisson con media proporcional a \(G\). Cada intento tiene un instante de inicio aleatorio dentro de la ventana.

La colisión clásica de Pure ALOHA se define con un periodo vulnerable de dos duraciones de paquete. En representación por instantes de inicio, dos paquetes se consideran en conflicto cuando sus inicios quedan a menos de una duración de paquete.

## 3.2 Modelo de señal física y ruido, canal (AWGN y colisiones)

La señal recibida se modela como suma de contribuciones de múltiples transmisores más ruido aditivo blanco gaussiano:

\[
r[n] = \sum_k s_k[n-\tau_k] + w[n]
\]

donde \(\tau_k\) es el instante de llegada del paquete \(k\). En colisión, varias contribuciones se solapan y se suman linealmente.

En dominio complejo I/Q, el ruido AWGN debe ser circularmente simétrico:

\[
w[n] = w_I[n] + j\,w_Q[n], \quad
w_I, w_Q \sim \mathcal{N}(0, \sigma^2/2)
\]

de forma que la potencia compleja de ruido cumpla \(E[|w[n]|^2] = \sigma^2\).

## 3.3 Formulación del problema de detección

Dada una señal recibida \(r[n]\), se busca estimar los instantes de inicio de preámbulo \(\{\hat{\tau}_m\}\). En un entorno de acceso múltiple y ruido, el problema consiste en distinguir picos válidos de sincronización frente a:

- interferencia multiusuario por solapamiento,
- ruido térmico del receptor,
- picos espurios por fluctuaciones aleatorias.

La evaluación se realiza comparando detecciones estimadas con instantes verdaderos del escenario.

En esta memoria se emplean dos niveles complementarios de evaluación que comparten
la misma verdad terreno: (i) evaluación por evento (instantes detectados frente a
instantes verdaderos, con matching 1-a-1 y tolerancia temporal) y (ii) evaluación
ROC canónica por índice del estadístico de correlación. En la ROC, cada índice
temporal se etiqueta como positivo si cae dentro de \(\pm \Delta\) muestras de una
llegada real, y negativo en caso contrario (resto de índices), lo que permite
calcular TPR/FPR con TN explícitos.

## 3.4 Detector clásico de referencia: el correlador

El detector clásico se basa en filtro adaptado/correlación cruzada con un preámbulo conocido \(p[n]\):

\[
C[k] = \sum_{n=0}^{L-1} r[k+n]\,p^*[n]
\]

donde \(p^*[n]\) es el conjugado complejo y \(L\) la longitud del preámbulo. La decisión se toma sobre \(|C[k]|\) (o \(|C[k]|^2\)) con umbral. Para escenarios multipaquete, se aplica un posprocesado de picos para evitar múltiples detecciones del mismo evento.

# Capítulo 4. Implementación del Entorno de Simulación y Receptores

## 4.1 Arquitectura cross-layer MAC-PHY (traducción de tiempos a muestras y creación de escenarios inmutables)

Implementación actual en Python:

1. `simular_pure_aloha` genera instantes de llegada en tiempo MAC.
2. Esos tiempos se traducen a muestras PHY mediante:
   `instante_muestra = int(tiempo_mac * muestras_por_paquete)`.
3. Se crea un escenario con:
   - `senal_rx` (suma de paquetes en canal + AWGN),
   - `instantes_llegada_muestras` (verdad de referencia),
   - metadatos (SNR, G, longitud, etc.).

El escenario es común para cualquier receptor (correlador hoy, ML en el futuro), lo que garantiza comparativas justas.

### 4.1.1 Protocolo común de evaluación (congelado)

Para evitar sesgos metodológicos, el protocolo de evaluación se fija de forma
común antes de comparar detectores:

1. **Definición de evento**: un paquete verdadero es cada inicio en
   `instantes_llegada_muestras`.
2. **Evaluación por evento**: matching 1-a-1 dentro de \(\pm \Delta\) muestras,
   con métricas `TP/FP/FN`, `Recall`, `Precision` y `F1`.
3. **Evaluación ROC por índice**: positivos en \(\pm \Delta\) de llegadas reales,
   negativos como el resto del eje temporal de correlación; se reportan `TPR/FPR`
   y `AUC`.
4. **Reproducibilidad**: grid fijo \((G,\mathrm{SNR})\), número de iteraciones
   Monte Carlo fijo y semillas controladas.

Este protocolo aplica tanto al correlador como al detector ML futuro: cambia el
motor de decisión, pero no cambia el escenario ni el árbitro de métricas.

## 4.2 Pipeline del Correlador (Generación I/Q, umbral y árbitro de métricas TP/FP/FN)

### Generación I/Q y canal

- `pipeline/transmitter.py` genera ahora preámbulo Zadoff-Chu complejo (`generar_preambulo_zc`).
- `generar_preambulo(..., tipo='zc')` activa por defecto el modo complejo.
- `pipeline/channel.py` detecta automáticamente si la señal es compleja y, en ese caso, usa ruido AWGN complejo circularmente simétrico.

### Detección por umbral

Tras la correlación, se declaran detecciones en todas las muestras con `corr_norm >= tau` (sin fusionar picos vecinos).

### Árbitro de métricas: TP/FP/FN

`evaluar_detecciones` compara `instantes_detectados` frente a `instantes_verdaderos` con una tolerancia en muestras:

- TP: detección asignada a un verdadero dentro de la ventana.
- FP: detección sin correspondencia válida.
- FN: verdadero sin detección asignada.

Ejemplo simple con tolerancia \(\pm 2\):

- Verdaderos: `{100, 500}`
- Detectados: `{101, 250, 502}`

Resultado:

- TP = 2 (`101` con `100`, `502` con `500`)
- FP = 1 (`250`)
- FN = 0

## 4.3 Pipeline del Detector ML (Generación de dataset y red neuronal que haremos en el futuro)

El entorno actual ya deja preparada la interfaz:

- Entrada común al receptor: `senal_rx` del escenario.
- Verdad de referencia: `instantes_llegada_muestras`.
- Métrica común: `evaluar_detecciones`.

Para el detector ML, la integración prevista es:

1. reutilizar exactamente el mismo generador de escenarios;
2. entrenar/inferir una red que produzca instantes detectados;
3. evaluar con el mismo árbitro TP/FP/FN.

Así se desacopla el problema de ingeniería del canal/escenario de la comparación de receptores.
