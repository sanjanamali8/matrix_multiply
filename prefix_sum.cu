__global__ void prefix_sum(float *A, int N)
{
    int tid = threadIdx.x;

    for (int stride = 1; stride <= N / 2; stride *= 2)
    {
        if ((tid + 1) % (2 * stride) == 0)
        {
            A[tid] += A[tid - stride];
        }
        __syncthreads();
    }

    __syncthreads();

    if (tid == N - 1)
    {
        A[tid] = 0;
    }

    for (int stride = N / 2; stride >= 1; stride /= 2)
    {
        if ((tid + 1) % (2 * stride) == 0)
        {
            int left = A[tid - stride];
            A[tid - stride] = A[tid];
            A[tid] = A[tid] + left;
        }
        __syncthreads();
    }
}