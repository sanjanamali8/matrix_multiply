__global__ void bfs(const int *row_ptr, const int *col_indices, bool *F, bool *F_next, int *dist, const int V, int level, int *has_active)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V)
        return;

    if (F[idx])
    {
        for (int i = row_ptr[idx]; i < row_ptr[idx + 1]; i++)
        {
            if (dist[col_indices[i]] == -1)
            {
                atomicOr(has_active, 1);
                dist[col_indices[i]] = level + 1;
                F_next[col_indices[i]] = true;
            }
        }
        F[idx] = false;
    }
}

__global__ void pull_bfs(const int *col_ptr, const int *row_indices, bool *F, bool *F_next, int *dist, const int V, int level, int *has_active)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V)
        return;

    if (dist[idx] == -1) // means its still unvisted!
    {
        for (int i = col_ptr[idx]; i < col_ptr[idx + 1]; i++) // finding incoming edges i.e. parents
        {
            if (F[row_indices[i]]) // check if in frontier
            {
                atomicOr(has_active, 1);
                dist[idx] = dist[row_indices[i]] + 1;
                F_next[idx] = true; // add child to frontier
                break;
            }
        }
    }
}