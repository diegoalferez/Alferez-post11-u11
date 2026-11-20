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

