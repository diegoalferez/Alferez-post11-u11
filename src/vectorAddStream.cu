// vectorAddStream.cu - Suma de vectores con transferencia asincrona y streams
// Compara una version sincrona (cudaMemcpy) contra una version que divide los
// datos en chunks y los reparte en streams (cudaMemcpyAsync) para solapar
// transferencia PCIe y computo.
// Arquitectura de Computadores - Unidad 11 (Post-contenido, Parte 1)
//
// Compilar: nvcc -O2 -o vectorAddStream src/vectorAddStream.cu
// Otro numero de streams: nvcc -O2 -DNSTREAMS=8 -o vectorAddStream src/vectorAddStream.cu
// Ejecutar: ./vectorAddStream
//
// Convencion de nombres: h_ = memoria del host, d_ = memoria del dispositivo (GPU).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 16777216  // 1 << 24 = 16M elementos, igual que vectorAdd.cu

#ifndef NSTREAMS
#define NSTREAMS 4  // numero de streams (N debe ser divisible entre NSTREAMS)
#endif

#define CHUNK        (N / NSTREAMS)
#define BYTES        ((size_t)N * sizeof(float))
#define CHUNK_BYTES  ((size_t)CHUNK * sizeof(float))
#define BLOCK_SIZE   256
#define REPS         5      // repeticiones medidas por version (promedio)
#define TOLERANCE    1e-4f

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

// Kernel: cada thread suma un elemento del chunk que recibe
__global__ void vectorAddChunk(const float *d_A, const float *d_B,
                               float *d_C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_C[idx] = d_A[idx] + d_B[idx];
}

// Cuenta los elementos que difieren de la referencia de CPU
static int countErrors(const float *h_out, const float *h_ref, int n) {
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(h_out[i] - h_ref[i]) > TOLERANCE) errors++;
    return errors;
}

typedef struct {
    float h2dMs;     // dos copias H2D (A y B)
    float kernelMs;  // kernel
    float d2hMs;     // copia D2H (C)
    float totalMs;   // H2D + kernel + D2H
} SyncTimes;

// Version sincrona: una transferencia H2D por arreglo, un kernel, una D2H
static SyncTimes runSync(const float *h_A, const float *h_B, float *h_C,
                         float *d_A, float *d_B, float *d_C, cudaEvent_t ev[4]) {
    SyncTimes t;
    CUDA_CHECK(cudaEventRecord(ev[0]));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev[1]));
    vectorAddChunk<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev[2]));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaEventSynchronize(ev[3]));

    CUDA_CHECK(cudaEventElapsedTime(&t.h2dMs,    ev[0], ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernelMs, ev[1], ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2hMs,    ev[2], ev[3]));
    CUDA_CHECK(cudaEventElapsedTime(&t.totalMs,  ev[0], ev[3]));
    return t;
}

// Version con streams: cada stream procesa un chunk (H2D -> kernel -> D2H),
// de modo que la copia de un chunk se solapa con el computo de otro
static float runStreams(const float *h_A, const float *h_B, float *h_C,
                        float *d_A, float *d_B, float *d_C,
                        cudaStream_t streams[NSTREAMS],
                        cudaEvent_t start, cudaEvent_t stop) {
    int chunkGrid = (CHUNK + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaEventRecord(start));
    for (int s = 0; s < NSTREAMS; s++) {
        size_t off = (size_t)s * CHUNK;
        CUDA_CHECK(cudaMemcpyAsync(d_A + off, h_A + off, CHUNK_BYTES,
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_B + off, h_B + off, CHUNK_BYTES,
                                   cudaMemcpyHostToDevice, streams[s]));
        vectorAddChunk<<<chunkGrid, BLOCK_SIZE, 0, streams[s]>>>(
            d_A + off, d_B + off, d_C + off, CHUNK);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_C + off, d_C + off, CHUNK_BYTES,
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    for (int s = 0; s < NSTREAMS; s++) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | N = %d | NSTREAMS = %d | copy engines = %d\n",
           prop.name, N, NSTREAMS, prop.asyncEngineCount);

    // Memoria host "pinned" (page-locked): obligatoria para que cudaMemcpyAsync
    // se ejecute realmente de forma asincrona
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaHostAlloc((void **)&h_A, BYTES, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_B, BYTES, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_C, BYTES, cudaHostAllocDefault));

    // Referencia de CPU para verificar ambas versiones
    float *h_ref = (float *)malloc(BYTES);
    if (!h_ref) {
        fprintf(stderr, "Error: sin memoria en el host\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
        h_ref[i] = h_A[i] + h_B[i];
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, BYTES));
    CUDA_CHECK(cudaMalloc((void **)&d_B, BYTES));
    CUDA_CHECK(cudaMalloc((void **)&d_C, BYTES));

    cudaEvent_t ev[4], start, stop;
    for (int e = 0; e < 4; e++) CUDA_CHECK(cudaEventCreate(&ev[e]));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Calentamiento de ambas versiones (no se mide)
    runSync(h_A, h_B, h_C, d_A, d_B, d_C, ev);
    runStreams(h_A, h_B, h_C, d_A, d_B, d_C, streams, start, stop);

    // ---- Version sincrona ----
    memset(h_C, 0, BYTES);
    SyncTimes avg = {0, 0, 0, 0};
    for (int r = 0; r < REPS; r++) {
        SyncTimes t = runSync(h_A, h_B, h_C, d_A, d_B, d_C, ev);
        avg.h2dMs += t.h2dMs;       avg.kernelMs += t.kernelMs;
        avg.d2hMs += t.d2hMs;       avg.totalMs  += t.totalMs;
    }
    avg.h2dMs /= REPS;  avg.kernelMs /= REPS;
    avg.d2hMs /= REPS;  avg.totalMs  /= REPS;
    int errSync = countErrors(h_C, h_ref, N);

    // ---- Version con streams ----
    memset(h_C, 0, BYTES);
    float asyncMs = 0.0f;
    for (int r = 0; r < REPS; r++)
        asyncMs += runStreams(h_A, h_B, h_C, d_A, d_B, d_C, streams, start, stop);
    asyncMs /= REPS;
    int errAsync = countErrors(h_C, h_ref, N);

    // ---- Resultados ----
    double totalGB = 3.0 * (double)BYTES / 1e9;  // A y B (H2D) + C (D2H)
    printf("\nDesglose sincrono: H2D %.2f ms | kernel %.2f ms | D2H %.2f ms\n",
           avg.h2dMs, avg.kernelMs, avg.d2hMs);
    printf("Volumen transferido: %.1f MB | ancho de banda efectivo: %.1f GB/s\n",
           totalGB * 1000.0, totalGB / (avg.h2dMs + avg.d2hMs) * 1000.0);
    printf("Sincrono (H2D + kernel + D2H completos): %.2f ms\n", avg.totalMs);
    printf("Con streams (%d, solapado): %.2f ms\n", NSTREAMS, asyncMs);
    printf("Reduccion: %.1f%%\n", 100.0 * (avg.totalMs - asyncMs) / avg.totalMs);
    printf("Errores (sincrono): %d | Errores (streams): %d\n", errSync, errAsync);

    // ---- Liberar recursos ----
    for (int s = 0; s < NSTREAMS; s++) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    for (int e = 0; e < 4; e++) CUDA_CHECK(cudaEventDestroy(ev[e]));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B)); CUDA_CHECK(cudaFreeHost(h_C));
    free(h_ref);
    return (errSync == 0 && errAsync == 0) ? 0 : 1;
}
