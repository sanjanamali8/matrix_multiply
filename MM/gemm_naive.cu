#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define cudaCheck(err) (cudaCheckImpl(err, __FILE__, __LINE__))

__global__ void gemm_naive(int M, int N, int K, const float *A, const float *B, float *C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        //printf("Thread - Row = %d, Col = %d \n", row, col);
        float tmp = 0.0;
        for (int k = 0; k < K; k++)
        {
            int i = row * K + k, j = k * N + col;
            tmp += A[row * K + k] * B[k * N + col];
            printf("Thread[%d][%d] - A = %5.0f at %d \t B = %5.0f at %d\n",row, col, A[i], i, B[j], j);
        }
        C[row * N + col] = tmp;
        printf("C at %d is %5.0f\n", row * N + col, tmp);
    }
}

void cudaCheckImpl(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 5;
    }
}

bool verify(float *C, float *C_ref, int n)
{
    float diff = 0.0;
    for (int i = 0; i < n; i++)
    {
        diff = std::fabs(C[i] - C_ref[i]);
        if (std::isnan(diff) || diff > 0.01)
        {
            printf("Divergence! Should be %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   C_ref[i], C[i], diff, i);
            return false;
        }
    }
    return true;
}

void print_matrix(float *mat, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%5.0f\t", mat[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    if(argc != 2) {
        std::cerr<<"Please enter matrix size!";
        exit(EXIT_FAILURE);
    }
    int max_size = std::stoi(argv[1]);
    std::cout<<"Size = " << max_size <<"\n";
    int M, N, K;
    M = N = K = max_size;
    float *A, *B, *C, *C_ref;
    float *dA, *dB, *dC;

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);

    printf("A:\n");
    print_matrix(A, M, K);

    printf("B:\n");
    print_matrix(B, K, N);

    for (int i = 0; i < M * N; i++)
    {
        C[i] = 0;
    }

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    // dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    // dim3 blockDim(32, 32);

    dim3 gridDim(1, 1);
    dim3 blockDim(4, 4);
    cudaEventRecord(beg);
    gemm_naive<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC);

    cudaCheck(cudaGetLastError());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    // elapsed_time /= 1000.; // Convert to seconds

    printf("GPU elapsed time: %7.6f ms\n", elapsed_time);

    cudaCheck(cudaMemcpy(C, dC, sizeof(float) * max_size * max_size, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < M; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         float tmp = 0.0f;
    //         for (int k = 0; k < K; k++)
    //         {
    //             tmp += A[i * K + k] * B[k * N + j];
    //         }
    //         C_ref[i * N + j] = tmp;
    //     }
    // }

    // if (verify(C, C_ref, M * N))
    // {
    //     std::cout << "Correct!\n";
    // }

//     if(max_size < 32){
    //     printf("A:\n");
    //     print_matrix(A, M, K);

    //     printf("B:\n");
    //     print_matrix(B, K, N);

    //     printf("GPU C:\n");
    //     print_matrix(C, M, N);

    //     printf("CPU C_ref:\n");
    //     print_matrix(C_ref, M, N);
//     }
    free(A);
    free(B);
    free(C);
    free(C_ref);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
