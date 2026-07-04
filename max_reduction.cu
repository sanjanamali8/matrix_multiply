__global__ void findmax(float *A, const int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x % N;
    if (tid >= N)
        return;

    for (int stride = 1; stride <= N / 2; stride *= 2)
    {
        if (tid % (stride * 2) == 0)
        {
            A[tid] = max(A[tid], A[tid + stride]);
        }
        __syncthreads();
    }
} // max element stored at A[0];