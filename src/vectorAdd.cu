// vectorAdd.cu - Suma de vectores en CUDA: benchmark CPU vs GPU
// Arquitectura de Computadores - Unidad 11 (Post-contenido, Parte 1)
//
// Compilar: nvcc -O2 -o vectorAdd src/vectorAdd.cu
// Ejecutar: ./vectorAdd
//
// Convencion de nombres: h_ = memoria del host, d_ = memoria del dispositivo (GPU).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256   // threads por bloque
#define CPU_REPS   5     // repeticiones para promediar el tiempo de CPU
#define GPU_REPS   10    // repeticiones para promediar los tiempos de GPU
#define TOLERANCE  1e-4f // error maximo permitido entre CPU y GPU

// Tamanos del benchmark: 1M, 4M y 16M elementos
static const int SIZES[] = { 1 << 20, 1 << 22, 1 << 24 };
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// Verifica el resultado de cada llamada a la API de CUDA
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "Error CUDA en %s:%d -> %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err_));            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel CUDA: cada thread calcula un elemento de C
__global__ void vectorAdd(const float *d_A, const float *d_B,
                          float *d_C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_C[idx] = d_A[idx] + d_B[idx];
}

// Suma de vectores en CPU (referencia)
static void vectorAddCPU(const float *h_A, const float *h_B,
                         float *h_C, int n) {
    for (int i = 0; i < n; i++) h_C[i] = h_A[i] + h_B[i];
}

// Ejecuta el benchmark para un tamano n e imprime una fila de tabla Markdown
// (lista para copiar al README). Devuelve el numero de errores encontrados.
static int runBenchmark(int n) {
    size_t bytes = (size_t)n * sizeof(float);

    // Memoria del host
    float *h_A     = (float *)malloc(bytes);
    float *h_B     = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Error: sin memoria en el host para n = %d\n", n);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // ---- Benchmark CPU (promedio de CPU_REPS ejecuciones) ----
    double cpuMs = 0.0;
    for (int r = 0; r < CPU_REPS; r++) {
        clock_t t0 = clock();
        vectorAddCPU(h_A, h_B, h_C_cpu, n);
        cpuMs += (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
    }
    cpuMs /= CPU_REPS;

    // ---- Benchmark GPU ----
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t ev[4]; // 0: inicio, 1: tras H2D, 2: tras kernel, 3: tras D2H
    for (int e = 0; e < 4; e++) CUDA_CHECK(cudaEventCreate(&ev[e]));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Calentamiento (no se mide): evita contar la inicializacion del contexto
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double kernelMs = 0.0, totalMs = 0.0;
    for (int r = 0; r < GPU_REPS; r++) {
        CUDA_CHECK(cudaEventRecord(ev[0]));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev[1]));
        vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev[2]));
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev[3]));
        CUDA_CHECK(cudaEventSynchronize(ev[3]));

        float tKernel = 0.0f, tTotal = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&tKernel, ev[1], ev[2])); // solo kernel
        CUDA_CHECK(cudaEventElapsedTime(&tTotal,  ev[0], ev[3])); // H2D + kernel + D2H
        kernelMs += tKernel;
        totalMs  += tTotal;
    }
    kernelMs /= GPU_REPS;
    totalMs  /= GPU_REPS;

    // ---- Verificar resultados contra la CPU ----
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > TOLERANCE) errors++;

    printf("| %9d | %9.2f | %14.3f | %20.3f | %13.1fx | %12.1fx | %7d |\n",
           n, cpuMs, kernelMs, totalMs, cpuMs / kernelMs, cpuMs / totalMs, errors);

    // ---- Liberar memoria ----
    for (int e = 0; e < 4; e++) CUDA_CHECK(cudaEventDestroy(ev[e]));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return errors;
}

int main(void) {
    cudaDeviceProp prop;
    int rtVersion = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaRuntimeGetVersion(&rtVersion));
    printf("GPU: %s (compute capability %d.%d) | CUDA runtime %d.%d\n\n",
           prop.name, prop.major, prop.minor,
           rtVersion / 1000, (rtVersion % 100) / 10);

    printf("| N (elementos) | CPU (ms) | GPU kernel (ms) | GPU total c/memcpy (ms) | Speedup kernel | Speedup total | Errores |\n");
    printf("|---------------|----------|-----------------|-------------------------|----------------|---------------|---------|\n");

    int totalErrors = 0;
    for (int s = 0; s < NUM_SIZES; s++) totalErrors += runBenchmark(SIZES[s]);

    printf("\nErrores totales: %d\n", totalErrors);
    return totalErrors == 0 ? 0 : 1;
}
