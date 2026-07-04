#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#define CEIL_DIV(M, N) ((M + N - 1) / N)
#define TM 8
#define BM 64
#define BN 64
#define BK 8


__global__ void oned_tiling(int M, int N, int K, float* A, float *B, float *C) 
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int cThreadRow = threadIdx.x / BN;
    const int cThreadCol = threadIdx.x % BN;

    const int aThreadRow = threadIdx.x / BK;
    const int aThreadCol = threadIdx.x % BK;
    
    const int bThreadRow = threadIdx.x / BN;
    const int bThreadCol = threadIdx.x % BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    float tmp[TM] = {0.0};

    for(int bIdx = 0; bIdx < K; bIdx += BK) {

        As[aThreadRow * BK + aThreadCol] = A[aThreadRow * K + aThreadCol];
        Bs[bThreadRow * BN + bThreadCol] = B[bThreadRow * N + bThreadCol];

        __syncthreads();

        A += BK;
        B += BK * N;

        for(int k = 0; k < BK; k++) {
            float tmpB = Bs[k * BN + cThreadCol];

            for(int tid = 0; tid < TM; tid++) {
                tmp[tid] += As[(cThreadRow * TM + tid) * BK + k] * tmpB;
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM; i++) {
        C[(cThreadRow * TM + i) * N + cThreadCol] = tmp[i];
    }
}


void randomize_matrix(float *mat, int N) {
    for(int i=0;i<N;i++) {
        mat[i] = rand() % 10;
    }
}

bool verify(float *C, float *C_ref, int n) {
    float diff = 0.0f;
    for(int i=0;i<n;i++) {
        diff = std::fabs(C[i] - C_ref[i]);
        if(std::isnan(diff) || diff > 0.01) {
            printf("Divergence! Should be %5.2f is %5.2f at %d\n", C_ref[i], C[i], i);
            return false;
        }
    }
    return true;
}

void printDeviceProperties() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Using CUDA device %d: %s\n", device, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f MB\n", prop.totalGlobalMem / (1024.0f * 1024.0f));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads dimensions: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid dimensions: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Memory clock rate: %.2f MHz\n", prop.memoryClockRate / 1000.0f);
    printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("  L2 cache size: %d KB\n", prop.l2CacheSize / 1024);
    printf("\n");
}

void cpu_run(float *A, float*B, float *C_ref, int M, int N, int K) {
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0.0f;
            for (int k = 0; k < K; k++)
            {
                tmp += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = tmp;
        }
    }
}

int main(int argc, char **argv) {
    if(argc != 2) {
        std::cerr<<"Please enter matrix size!";
        exit(EXIT_FAILURE);
    }

    int ipsize = std::stoi(argv[1]);
    std::cout<<"Size = " <<ipsize<< "\n";

    int M, N, K;
    M = N = K = ipsize;
    int size = ipsize * ipsize;

   // printDeviceProperties();

    float *A, *B, *C, *C_ref;
    float *dA, *dB, *dC;

    float elapsed_time;
    cudaEvent_t begin;
    cudaEvent_t end;

    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    A = (float *)malloc(sizeof(float) * size);
    B = (float *)malloc(sizeof(float) * size);
    C = (float *)malloc(sizeof(float) * size);
    C_ref = (float *)malloc(sizeof(float) * size);

    randomize_matrix(A, size);
    randomize_matrix(B, size);

    for(int i=0;i<M*N;i++) {
        C[i] = 0;
    }

    cudaMalloc((void **)&dA, sizeof(float) * size);
    cudaMalloc((void **)&dB, sizeof(float) * size);
    cudaMalloc((void **)&dC, sizeof(float) * size);

    cudaMemcpy(dA, A, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(float) * size, cudaMemcpyHostToDevice);

    dim3 blockDim((BM * BN)/TM);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEventRecord(begin);

    oned_tiling<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEventRecord(end);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, begin, end);

    printf("GPU elapsed time: %7.6f ms\n", elapsed_time);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsed_time / 1000.0) / 1e9;

    printf("GFLOPS/s = %6.4f\n", gflops);

    cudaMemcpy(C, dC, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // cpu_run(A, B, C_ref, M, N, K);
    // if(verify(C, C_ref, M * N)) {
    //     std::cout<<"Correct!\n";
    // }

    free(A);
    free(B);
    free(C);
    free(C_ref);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
