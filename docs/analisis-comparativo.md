# Análisis comparativo de aceleradores de hardware para inteligencia artificial: GPU, TPU, NPU y FPGA

**Asignatura:** Arquitectura de Computadores · Unidad 11 · UFPS
**Autor:** Diego Alferez
**Fecha de consulta de las fuentes:** noviembre de 2026

## Introducción

Este documento compara cuatro familias de aceleradores usadas en cargas de trabajo de inteligencia artificial: las GPU (representadas por NVIDIA H100/A100), las TPU (Google Cloud TPU v5e y sucesoras), las NPU (Apple Neural Engine del M3 Ultra) y las FPGA (AMD Alveo U55C). El análisis evalúa sus diferencias arquitectónicas, su rendimiento en el benchmark MLPerf Inference v4.1, su relación costo-rendimiento y su costo total de propiedad (TCO), el efecto de la cuantización INT8 sobre el throughput y la precisión, y los escenarios donde cada una resulta óptima. Todas las cifras provienen de hojas de especificaciones oficiales, de los resultados publicados de MLPerf o de artículos técnicos, y cada una lleva su fuente; las cifras derivadas por cálculo del autor se rotulan como tales, y los supuestos no respaldados por fuentes se declaran explícitamente. Cuando una cifra de la guía de contenido no coincide con las fuentes consultadas, el documento lo indica.

## 1. Caracterización de cada acelerador

La GPU NVIDIA H100 SXM sigue el paradigma SIMT masivo: miles de threads agrupados en warps se ejecutan sobre multiprocesadores que combinan núcleos CUDA con Tensor Cores de cuarta generación. Su hoja de datos reporta 80 GB de memoria HBM3 con 3,35 TB/s de ancho de banda, un TDP configurable de hasta 700 W y, para FP16/BF16, 1 979 TFLOPS en los Tensor Cores y 3 958 TOPS en INT8, cifras que NVIDIA declara con *sparsity* y que son la mitad sin ella (NVIDIA, s. f.), es decir, 989 TFLOPS en FP16 y 1 979 TOPS en INT8 en modo denso. Se programa con CUDA, cuDNN y TensorRT. Su predecesora A100 ofrece 312 TFLOPS en FP16 y 624 TOPS en INT8 en modo denso (NVIDIA, 2021).

La TPU de Google es un acelerador de propósito específico basado en arreglos sistólicos. Cada chip TPU v5e contiene un TensorCore con cuatro unidades de multiplicación de matrices (MXU), una unidad vectorial y una escalar; entrega un pico de 197 TFLOPS en bf16 y dispone de 16 GB de HBM2 a 819 GB/s, y un pod de 256 chips alcanza 100 PetaOps en INT8 (Google Cloud, s. f.-a). La guía menciona MXU de 128×128, dato que la documentación consultada no detalla. Se programa a través de XLA, JAX o TensorFlow y solo está disponible como servicio en Google Cloud.

La NPU se ilustra con el Neural Engine del Apple M3 Ultra, de flujo de datos fijo y orientado a inferencia en el dispositivo. Las fuentes consultadas indican que el M3 Ultra incluye un Neural Engine de 32 núcleos, el doble del de M4 Max, y más de 800 GB/s de ancho de banda de memoria unificada (9to5Mac, 2025; MacStories, 2025; MacRumors, 2025); la guía señala 18 núcleos, cifra que no coincide con esas fuentes. No se encontró una cifra oficial de TOPS para este Neural Engine, por lo que no se reporta. Se programa con Core ML.

La FPGA se ilustra con la AMD Alveo U55C: según su hoja de datos, 1 304 K tablas de búsqueda (LUT), 9 024 bloques DSP, 16 GB de HBM2 a 460 GB/s y un TDP de 115 W (150 W de carga eléctrica máxima) (AMD, 2023). Es reconfigurable: la lógica se define con HLS (Vitis) o VHDL/Verilog, lo que permite pipelines de latencia determinística, a cambio de un costo de desarrollo mayor.

| Acelerador | Paradigma | Unidad de cómputo | Memoria | Potencia | Programación |
|---|---|---|---|---|---|
| GPU NVIDIA H100 SXM | SIMT masivo | CUDA Core + Tensor Core | 80 GB HBM3, 3,35 TB/s | hasta 700 W | CUDA / cuDNN / TensorRT |
| TPU v5e (Google) | Arreglo sistólico | 4 MXU por TensorCore | 16 GB HBM2, 819 GB/s | no reportada en la fuente | XLA / JAX / TensorFlow |
| NPU Apple M3 Ultra | Flujo de datos fijo | Neural Engine de 32 núcleos | >800 GB/s (unificada) | no reportada en la fuente | Core ML |
| FPGA AMD Alveo U55C | Reconfigurable | LUT + DSP + BRAM/HBM | 16 GB HBM2, 460 GB/s | 115 W (TDP) | HLS (Vitis) / VHDL / Verilog |

## 2. Benchmarks MLPerf Inference

MLPerf Inference define dos escenarios principales: *Offline*, donde todas las muestras están disponibles a la vez y se mide el throughput, y *Server*, donde las consultas llegan de forma aleatoria y se mide el máximo de consultas por segundo que cumple un límite de latencia. En el escenario Server, ResNet-50 v1.5 debe cumplir una exactitud de al menos 99 % del modelo FP32 de referencia (76,46 %) con una latencia máxima de 15 ms (Dell Technologies, 2024), y BERT-Large sobre SQuAD v1.1 una exactitud de 99 % o 99,9 % de la referencia (F1 = 90,874) con un límite de 130 ms (Dell Technologies, 2021). Estos límites son el equivalente en MLPerf de la latencia p99 solicitada: el benchmark no publica una latencia p99 por acelerador, sino el throughput máximo que respeta el límite.

NVIDIA publicó en la ronda v4.1 resultados con ocho GPU H200 (la misma arquitectura Hopper que la H100, con memoria HBM3e) configuradas a 700 W (NVIDIA, 2024). Las cifras por GPU de la tabla siguiente son una división entre ocho hecha por el autor; MLPerf no usa el throughput por GPU como métrica primaria, y la eficiencia energética se estimó dividiendo entre el TDP configurado de 700 W, sin que corresponda a una medición de potencia de MLPerf.

| Modelo (escenario Offline) | Sistema 8×H200 | Por GPU (derivado) | Eficiencia derivada (muestras/s/W) |
|---|---|---|---|
| ResNet-50 v1.5 | 756 960 muestras/s | 94 620 muestras/s | 135,2 |
| BERT-Large (99 %) | 73 310 muestras/s | 9 164 muestras/s | 13,1 |
| Stable Diffusion XL | 17,42 muestras/s | 2,18 muestras/s | 0,0031 |

En el escenario Server, el mismo sistema alcanzó 632 229 consultas/s en ResNet-50 (79 029 por GPU), 57 609 en BERT-99 y 16,78 en SDXL (NVIDIA, 2024).

La cobertura de las demás familias en esa ronda es limitada. Google presentó la TPU en la v4.1 únicamente en Stable Diffusion: el chip Trillium (TPU v6e, vista previa) obtuvo 5,44 muestras/s en Offline y 4,49 consultas/s en Server, alrededor de tres veces el rendimiento de la TPU v5e (EE Times Japan, 2024); no se verificó la cantidad de chips de esa configuración, por lo que la cifra no es comparable por acelerador con la de la H200. No se encontraron resultados de Apple ni de FPGA en la sección de centros de datos de esa ronda. Entre las NPU, la SpeedAI240 de Untether AI reportó ResNet-50 con aproximadamente 88 % del rendimiento de una H200 y fue, junto con NVIDIA, de los únicos participantes que reportaron la prueba de potencia (xpu.pub, 2024).

Desde el punto de vista arquitectónico, la GPU lidera en throughput absoluto porque combina miles de núcleos, Tensor Cores con soporte nativo de FP8/INT8 y un ancho de banda de memoria de varios TB/s. La eficiencia energética favorece a las arquitecturas especializadas (flujo de datos fijo o arreglos sistólicos) porque eliminan la lógica de propósito general, aunque la falta de resultados comparables impide cuantificarlo con MLPerf para TPU, Apple y FPGA. Esta ronda no es la más reciente: Google ya reportó resultados posteriores (v5.0) para Trillium y para GPU H200/B200 (Google Cloud, 2025), pero la guía exige la v4.1.

## 3. Costo-rendimiento y TCO

**Precios consultados.** NVIDIA no publica precio de lista de la H100; las guías de mercado de 2026 la sitúan entre USD 25 000 y 40 000 por tarjeta según la variante (IntuitionLabs, 2026), y el alquiler en proveedores especializados va de unos USD 2,7 a 4,3 por GPU-hora (Tech Insider, 2026). La guía usa USD 30 000, valor dentro de ese rango. Para la TPU, Google Cloud cobra por chip-hora bajo demanda USD 1,20 (v5e, us-central1), 4,20 (v5p), 3,22 (v4) y 2,70 (Trillium), y un host de cuatro chips v4 aparece facturado como USD 12,88 por hora (Google Cloud, s. f.-b); la cifra de ~USD 32/hora de la guía no coincide con esas tarifas. La AMD Alveo U55C figura en el sitio de AMD a USD 4 747, con precios de distribuidor que van de unos CHF 4 321 a USD 10 134 según el canal (AMD, s. f.), frente a ~USD 8 000 de la guía. El Mac Studio con M3 Ultra parte de USD 3 999 (MacRumors, 2025).

**Supuestos del cálculo.** TCO a 3 años (26 280 h) con operación 24/7, electricidad a USD 0,12/kWh y un factor de refrigeración de 1,5 aplicado a la energía de equipos propios; ambos valores son supuestos ilustrativos del autor, no cifras de fuentes. La energía se estima con el TDP o la potencia típica reportada. Los costos de adquisición no incluyen el servidor anfitrión ni los costos de desarrollo y mantenimiento del software (CUDA/TensorRT, XLA o HLS), que no se cuantifican por falta de fuentes.

| Opción (3 años, 24/7) | Adquisición / alquiler | Energía (supuesto) | Total aproximado |
|---|---|---|---|
| H100 propia (USD 30 000, 700 W) | 30 000 | ≈ 3 311 | ≈ 33 300 |
| H100 alquilada (USD 2,69/GPU-h) | ≈ 70 700 | incluida | ≈ 70 700 |
| TPU v5e, 1 chip bajo demanda (USD 1,20/h) | ≈ 31 500 | incluida | ≈ 31 500 |
| TPU v5e, 1 chip con compromiso de 3 años (USD 0,54/h) | ≈ 14 200 | incluida | ≈ 14 200 |
| FPGA U55C propia (USD 4 747, 115 W típicos) | 4 747 | ≈ 544 | ≈ 5 300 |

**Escenario (a): startup con inferencia en la nube a escala media (menos de 100 solicitudes/s).** A esa carga, un solo acelerador sobra con holgura: la H100 sostiene decenas de miles de consultas/s de ResNet-50 por GPU (sección 2). La decisión la domina el costo y la ingeniería, no el throughput. La TPU v5e con compromiso es la opción más barata por hora y la GPU alquilada la más flexible. Se prefiere la TPU v5e si el modelo ya está en JAX o TensorFlow, porque elimina la inversión de capital; se prefiere la GPU si el equipo depende del ecosistema CUDA/TensorRT, pues el costo de portar el código a XLA, que no se cuantifica aquí por falta de fuentes, puede borrar el ahorro.

**Escenario (b): inferencia en el borde (smartphone, cámara industrial, vehículo).** Una H100 de 700 W y una U55C de 115 W exceden el presupuesto térmico de estos dispositivos, y la U55C es una tarjeta de centro de datos. La opción natural es la NPU integrada en el SoC, cuyo costo marginal es casi nulo; la FPGA solo se justifica cuando el requisito es latencia determinística. Este análisis no cubre FPGA de borde de menor potencia, que serían las candidatas realistas en una cámara industrial.

## 4. Recomendación fundamentada por perfil

**Perfil A: laboratorio universitario, USD 10 000, visión y NLP hasta 7B parámetros.** Se recomienda alquilar GPU NVIDIA en la nube (H100 o A100) en lugar de comprar hardware. Los criterios son: (1) *presupuesto*: USD 10 000 no alcanzan para una H100 (USD 25 000–40 000) pero financian unas 3 700 horas de H100 a USD 2,69/h; (2) *memoria*: un modelo de 7B en FP16 ocupa unos 14 GB y el ajuste fino necesita de 3 a 4 veces más (hasta unos 56 GB), que caben en los 80 GB de una H100 (Codesota, 2026); (3) *ecosistema*: CUDA y PyTorch cubren visión y NLP sin portar código, a diferencia de XLA en TPU o Core ML en Apple. Limitaciones: no hay propiedad del hardware, el costo escala con el uso y un Mac Studio M3 Ultra (desde USD 3 999) sería una alternativa local viable para inferencia, aunque sin CUDA.

**Perfil B: manufactura, inspección visual en tiempo real (<5 ms), 50 líneas.** Se recomienda una FPGA por línea si la inspección gobierna el rechazo de piezas con implicaciones de seguridad, y una GPU centralizada con redundancia si el control de calidad no es crítico para la seguridad. El razonamiento detallado está en las decisiones técnicas de las secciones 6 y 7.

**Perfil C: proveedor en la nube, modelos de 70B en clústeres de 512 aceleradores con comunicación colectiva intensiva.** Se recomienda GPU NVIDIA H100/H200 en clústeres con NVLink e InfiniBand. Los criterios son: (1) *interconexión*: 900 GB/s por GPU de NVLink 4.0 dentro del nodo (NVIDIA, s. f.), crítica para los *all-reduce* del entrenamiento; (2) *ecosistema y demanda*: un proveedor que revende capacidad debe ofrecer el software que sus clientes ya usan, CUDA y NCCL; (3) *flexibilidad*: la misma flota sirve entrenamiento e inferencia. Limitaciones: 512 GPU a 700 W suponen unos 358 kW solo en aceleradores (cálculo del autor), una inversión de unos USD 15,4 millones a USD 30 000 por GPU sin red ni servidores, y plazos de entrega de 6 a 12 meses en compras grandes (GMI Cloud, 2026). La TPU es competitiva, pues Google reporta 99 % de eficiencia de escalado en entrenamiento con Trillium (Google Cloud, 2024), pero solo se obtiene alquilándola a Google, lo que no sirve a un proveedor que quiere ser dueño de su infraestructura.

## 5. Cuantización INT8 frente a FP32: efecto en throughput y precisión

El throughput de los Tensor Cores se duplica al pasar de FP16/BF16 a INT8 en la misma GPU: la A100 entrega 312 TFLOPS en FP16 y 624 TOPS en INT8 (NVIDIA, 2021), y la H100 989 TFLOPS frente a 1 979 TOPS en modo denso (NVIDIA, s. f.). Wu et al. (2020) muestran el mismo patrón en la arquitectura Turing: respecto a FP32, FP16 logra 8× de throughput matemático e INT8 16×, es decir, el doble que FP16. La ganancia no es una cifra de un solo modelo de GPU, sino una propiedad de la aritmética entera de los Tensor Cores.

| Configuración | Throughput relativo | Modelo | Exactitud reportada | Fuente |
|---|---|---|---|---|
| FP32 (referencia) | 1× | ResNet-50 v1.5 | 76,16 % top-1 (checkpoint de Wu et al.); 76,46 % en MLPerf | Wu et al. (2020); Dell Technologies (2024) |
| INT8, PTQ calibrado con entropía | ~2× vs FP16 (Tensor Core) | ResNet-50 v1.5 | 76,05 % (−0,14 %) | Wu et al. (2020) |
| INT8, QAT | ~2× vs FP16 (Tensor Core) | ResNet-50 v1.5 | 76,85 % (+0,91 %) | Wu et al. (2020) |
| FP32 (referencia) | 1× | BERT-Large (SQuAD v1.1) | F1 = 91,01 (checkpoint de Wu et al.); 90,874 en MLPerf | Wu et al. (2020); Dell Technologies (2021) |
| INT8, PTQ con calibración *max* | ~2× vs FP16 (Tensor Core) | BERT-Large | F1 = 85,92 | Wu et al. (2020) |
| INT8, PTQ con calibración *entropy* | ~2× vs FP16 (Tensor Core) | BERT-Large | F1 = 37,40 | Wu et al. (2020) |
| INT8, PTQ con mejor calibración (percentil 99,999) | ~2× vs FP16 (Tensor Core) | BERT-Large | F1 = 90,20 (−0,89 %) | Wu et al. (2020) |
| INT8, QAT | ~2× vs FP16 (Tensor Core) | BERT-Large | F1 = 90,67 (−0,37 %) | Wu et al. (2020) |

**La cuantización no es un simple *cast*.** Los resultados de BERT-Large lo demuestran: con una calibración inadecuada la exactitud cae de F1 = 91,01 a 85,92 (*max*) o incluso a 37,40 (*entropy*), mientras que una calibración por percentil la recupera a 90,20 y el entrenamiento consciente de la cuantización (QAT) a 90,67. En ResNet-50 el efecto es mucho menor (−0,14 % con PTQ), lo que muestra que la sensibilidad depende de la arquitectura: en el mismo estudio, EfficientNet-b0 pierde 6,23 % con PTQ y solo se recupera con QAT. Wu et al. (2020) recomiendan pesos con cuantización por canal y calibración *max*, activaciones con calibración elegida entre varias, y recurrir a cuantización parcial o QAT cuando la exactitud no alcanza.

**Restricción de exactitud de MLPerf.** MLPerf exige que un envío cuantizado conserve al menos 99 % de la exactitud FP32 de referencia en ResNet-50 (76,46 % × 0,99 ≈ 75,70 %), y 99 % o 99,9 % en BERT-Large (F1 de 89,97 o de 90,78 sobre 90,874; cálculo del autor sobre la referencia de MLPerf). Con esas reglas, ResNet-50 en INT8 pasa con holgura incluso con PTQ; BERT-Large pasa el umbral de 99 % solo con la mejor calibración o con QAT, y los 90,20 y 90,67 de Wu et al. (2020) quedan por debajo del umbral de 99,9 %. Las referencias de Wu et al. (F1 = 91,01) y de MLPerf (90,874) usan checkpoints distintos, de modo que la comparación solo ilustra el orden de magnitud. Por eso las cifras de TOPS de un fabricante son un techo teórico y no una promesa de rendimiento con exactitud preservada.

