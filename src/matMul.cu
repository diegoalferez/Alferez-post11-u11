// matMul.cu - Multiplicacion de matrices C = A x B en CUDA
// Compara la version naive (solo memoria global) contra tiling con shared memory.
// Arquitectura de Computadores - Unidad 11 (Post-contenido, Parte 1)
//
// Compilar: nvcc -O2 -o matMul src/matMul.cu
// Ejecutar: ./matMul
//
// Convencion de nombres: h_ = memoria del host, d_ = memoria del dispositivo (GPU).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE       16     // lado del tile (bloque de TILE x TILE threads)
#define GPU_REPS   10     // repeticiones para promediar el tiempo de cada kernel
#define TOLERANCE  1e-3f  // error maximo permitido entre CPU y GPU

// Tamanos del benchmark
static const int SIZES[] = { 512, 1024 };
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

// Kernel naive: cada thread calcula un elemento de C leyendo de memoria global
__global__ void matMulNaive(const float *d_A, const float *d_B,
                            float *d_C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += d_A[row * n + k] * d_B[k * n + col];
        d_C[row * n + col] = sum;
    }
}

// Kernel con tiling: cada bloque carga tiles de A y B en shared memory y los
// reutiliza TILE veces, reduciendo los accesos a memoria global en un factor TILE
__global__ void matMulTiled(const float *d_A, const float *d_B,
                            float *d_C, int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        // Cargar tile de A en shared memory
        if (row < n && t * TILE + threadIdx.x < n)
            sA[threadIdx.y][threadIdx.x] = d_A[row * n + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Cargar tile de B en shared memory
        if (col < n && t * TILE + threadIdx.y < n)
            sB[threadIdx.y][threadIdx.x] = d_B[(t * TILE + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // esperar a que el tile este cargado

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads(); // esperar antes de cargar el siguiente tile
    }

    if (row < n && col < n) d_C[row * n + col] = sum;
}

// Multiplicacion en CPU (referencia). Orden i-k-j para recorrer B por filas.
static void matMulCPU(const float *h_A, const float *h_B, float *h_C, int n) {
    for (int i = 0; i < n * n; i++) h_C[i] = 0.0f;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = h_A[i * n + k];
            for (int j = 0; j < n; j++)
                h_C[i * n + j] += a * h_B[k * n + j];
        }
}

// Error absoluto maximo entre dos matrices n x n
static float maxAbsError(const float *h_X, const float *h_Y, int n) {
    float maxErr = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float e = fabsf(h_X[i] - h_Y[i]);
        if (e > maxErr) maxErr = e;
    }
    return maxErr;
}

// Mide el tiempo promedio (ms) de un kernel con cudaEvent
static float timeKernel(int useTiled, const float *d_A, const float *d_B,
                        float *d_C, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Calentamiento (no se mide)
    if (useTiled) matMulTiled<<<grid, block>>>(d_A, d_B, d_C, n);
    else          matMulNaive<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < GPU_REPS; r++) {
        if (useTiled) matMulTiled<<<grid, block>>>(d_A, d_B, d_C, n);
        else          matMulNaive<<<grid, block>>>(d_A, d_B, d_C, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / GPU_REPS;
}

// Ejecuta el benchmark para un tamano n e imprime una fila de tabla Markdown.
// Devuelve 1 si algun kernel supera la tolerancia respecto a la CPU.
static int runBenchmark(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A       = (float *)malloc(bytes);
    float *h_B       = (float *)malloc(bytes);
    float *h_C_cpu   = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_naive || !h_C_tiled) {
        fprintf(stderr, "Error: sin memoria en el host para n = %d\n", n);
        exit(EXIT_FAILURE);
    }

    // Entradas asimetricas con valores multiplos de 0.25: todos los productos y
    // sumas parciales son exactos en float, asi el resultado no depende del
    // orden de acumulacion y un error de indexacion (p. ej. transposicion) se nota.
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = (float)((i * 3 + j) % 5) * 0.25f;
            h_B[i * n + j] = (float)((i + 2 * j) % 7) * 0.25f;
        }

    // ---- CPU (referencia) ----
    clock_t t0 = clock();
    matMulCPU(h_A, h_B, h_C_cpu, n);
    double cpuMs = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

    // ---- GPU ----
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    float naiveMs = timeKernel(0, d_A, d_B, d_C, n);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    float tiledMs = timeKernel(1, d_A, d_B, d_C, n);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    float errNaive = maxAbsError(h_C_naive, h_C_cpu, n);
    float errTiled = maxAbsError(h_C_tiled, h_C_cpu, n);

    printf("| %5d | %9.1f | %10.3f | %10.3f | %14.2fx | %12.1fx | %10.2e | %10.2e |\n",
           n, cpuMs, naiveMs, tiledMs, naiveMs / tiledMs, cpuMs / tiledMs,
           errNaive, errTiled);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_naive); free(h_C_tiled);
    return (errNaive > TOLERANCE || errTiled > TOLERANCE) ? 1 : 0;
}

int main(void) {
    cudaDeviceProp prop;
    int rtVersion = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaRuntimeGetVersion(&rtVersion));
    printf("GPU: %s (compute capability %d.%d) | CUDA runtime %d.%d | TILE = %d\n\n",
           prop.name, prop.major, prop.minor,
           rtVersion / 1000, (rtVersion % 100) / 10, TILE);

    printf("|   N   | CPU (ms)  | naive (ms) | tiling (ms) | speedup tiling/naive | speedup tiling/CPU | err naive  | err tiling |\n");
    printf("|-------|-----------|------------|-------------|----------------------|--------------------|------------|------------|\n");

    int failed = 0;
    for (int s = 0; s < NUM_SIZES; s++) failed += runBenchmark(SIZES[s]);

    printf("\nVerificacion (error < %.0e): %s\n", (double)TOLERANCE,
           failed == 0 ? "OK" : "FALLO");
    return failed == 0 ? 0 : 1;
}
