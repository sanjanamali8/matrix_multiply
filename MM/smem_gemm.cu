#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define cudaCheck(err) (cudaCheckImpl(err, __FILE__, __LINE__))
#define BLOCKSIZE 32

__global__ void gemm_naive(int M, int N, int K, const float *A, const float *B, float *C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        //printf("Thread - Row = %d, Col = %d \n", row, col);
        float tmp = 0.0;
        for (int k = 0; k < K; k++)
        {
            //int i = row * K + k, j = k * N + col;
            tmp += A[row * K + k] * B[k * N + col];
           // printf("Thread[%d][%d] - A = %5.0f at %d \t B = %5.0f at %d\n",row, col, A[i], i, B[j], j);
        }
        C[row * N + col] = tmp;
        //printf("C at %d is %5.0f\n", row * N + col, tmp);
    }
}

__global__ void smem_gemm(int M, int N, int K,
                          const float *A,
                          const float *B,
                          float *C)
{
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * BLOCKSIZE + tx;
    int row = blockIdx.y * BLOCKSIZE + ty;

    float tmp = 0.0f;

    for (int phase = 0; phase < CEIL_DIV(K, BLOCKSIZE); phase++) {
        int a_row = row;
        int a_col = phase * BLOCKSIZE + tx;

        int b_row = phase * BLOCKSIZE + ty;
        int b_col = col;

        if (a_row < M && a_col < K)
            As[ty][tx] = A[a_row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        if (b_row < K && b_col < N)
            Bs[ty][tx] = B[b_row * N + b_col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k++) {
            tmp += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = tmp;
    }
}

void cudaCheckImpl(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n",
               file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void randomize_matrix(float *mat, int size)
{
    for (int i = 0; i < size; i++) {
        mat[i] = rand() % 5;
    }
}

float run_one_size(int max_size)
{
    int M = max_size;
    int N = max_size;
    int K = max_size;

    size_t bytes = sizeof(float) * max_size * max_size;

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);

    if (!A || !B || !C) {
        std::cerr << "Host malloc failed for size " << max_size << "\n";
        exit(EXIT_FAILURE);
    }

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);

    float *dA, *dB, *dC;

    cudaCheck(cudaMalloc((void **)&dA, bytes));
    cudaCheck(cudaMalloc((void **)&dB, bytes));
    cudaCheck(cudaMalloc((void **)&dC, bytes));

    cudaCheck(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));

    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    float elapsed_time = 0.0f;

    cudaCheck(cudaEventRecord(beg));

    smem_gemm<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end));
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));

    cudaCheck(cudaEventDestroy(beg));
    cudaCheck(cudaEventDestroy(end));

    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));

    free(A);
    free(B);
    free(C);

    return elapsed_time;
}

int main()
{
    int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16284};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("BLOCKSIZE = %d\n\n", BLOCKSIZE);
    printf("%15s\n", "Time(ms)");  

    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];

        float time_ms = run_one_size(size);

        printf("%15.3f\n", time_ms);
    }

    return 0;
}