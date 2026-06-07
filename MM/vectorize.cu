#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define cudaCheck(err) (cudaCheckImpl(err, __FILE__, __LINE__))
#define BLOCKSIZE 16
#define TM 4
#define TN 4

__global__ void twod_tiling_vec(int M, int N, int K,
                          const float *A,
                          const float *B,
                          float *C)
{
    __shared__ float As[BLOCKSIZE][BLOCKSIZE * TM];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE * TN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * (BLOCKSIZE * TN) + tx; //output row & col
    int row = blockIdx.y * (BLOCKSIZE * TM) + ty;

    float tmp[TM][TN] = {0.0};

    for (int phase = 0; phase < CEIL_DIV(K, BLOCKSIZE); phase++) {
        int a_row = row;
        int a_col = phase * BLOCKSIZE + tx * 4;

       // int b_row = phase * BLOCKSIZE + ty;
       // int b_col = col;

        if(tx < BLOCKSIZE/4) {
            for (int i = 0; i < TM; i++) {
                int load_row = a_row + i * BLOCKSIZE;
                //int newty = ty + i * BLOCKSIZE;
                if (load_row < M && a_col+3 < K) {
                    int idx = load_row * K + a_col;
                    float4 Avec = reinterpret_cast<const float4*>(&A[idx])[0];
                    As[tx * 4 + 0][ty + i * BLOCKSIZE] = Avec.x;
                    As[tx * 4 + 1][ty + i * BLOCKSIZE] = Avec.y;
                    As[tx * 4 + 2][ty + i * BLOCKSIZE] = Avec.z;
                    As[tx * 4 + 3][ty + i * BLOCKSIZE] = Avec.w;
                }
                else {
                    As[tx * 4 + 0][ty + i * BLOCKSIZE] = 0.0f;
                    As[tx * 4 + 1][ty + i * BLOCKSIZE] = 0.0f;
                    As[tx * 4 + 2][ty + i * BLOCKSIZE] = 0.0f;
                    As[tx * 4 + 3][ty + i * BLOCKSIZE] = 0.0f;
                }
            }
        }

        if (tx < BLOCKSIZE / 4) {
            int b_row = phase * BLOCKSIZE + ty;

            for (int j = 0; j < TN; j++) {
                int b_col =
                    blockIdx.x * (BLOCKSIZE * TN)
                    + j * BLOCKSIZE
                    + tx * 4;

                if (b_row < K && b_col + 3 < N) {
                    reinterpret_cast<float4*>(&Bs[ty][j * BLOCKSIZE + tx * 4])[0] =
                        reinterpret_cast<const float4*>(&B[b_row * N + b_col])[0];
                } else {
                    Bs[ty][j * BLOCKSIZE + tx * 4 + 0] = 0.0f;
                    Bs[ty][j * BLOCKSIZE + tx * 4 + 1] = 0.0f;
                    Bs[ty][j * BLOCKSIZE + tx * 4 + 2] = 0.0f;
                    Bs[ty][j * BLOCKSIZE + tx * 4 + 3] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k++) {
            for(int i=0;i<TM;i++) {
                float tmpA = As[k][ty + i * BLOCKSIZE];
                for(int j = 0;j<TN;j++) {
                    float tmpB = Bs[k][tx + j * BLOCKSIZE];
                    tmp[i][j] +=  tmpA * tmpB;
                }
            }
        }

        __syncthreads();
    }

    for(int i=0;i<TM;i++) {
        for(int j=0;j<TN;j++) {
            int out_row = row + i*BLOCKSIZE;
            int out_col = col + j*BLOCKSIZE;
            if(out_row < M && out_col < N)
                C[(out_row)*N + out_col] = tmp[i][j];
        }
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

float run_one_size(int max_size)
{
    int M = max_size;
    int N = max_size;
    int K = max_size;

    size_t bytes = sizeof(float) * max_size * max_size;

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);
    float *C_ref = (float *)malloc(bytes);

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
    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE * TN), CEIL_DIV(M, BLOCKSIZE * TM));

    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    float elapsed_time = 0.0f;

    cudaCheck(cudaEventRecord(beg));

    twod_tiling_vec<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(end));
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));

    cudaCheck(cudaEventDestroy(beg));
    cudaCheck(cudaEventDestroy(end));

    // cudaCheck(cudaMemcpy(C, dC, sizeof(float) * max_size * max_size, cudaMemcpyDeviceToHost));

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

    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return elapsed_time;
}

int main()
{
    int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("BLOCKSIZE = %d\n", BLOCKSIZE);
    printf("TM = %d\n", TM);
    printf("TN = %d\n", TN);
    printf("%15s\n", "Time(ms)");  

    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];

        float time_ms = run_one_size(size);

        printf("%15.3f\n", time_ms);
    }

    return 0;
}