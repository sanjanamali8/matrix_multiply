#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>


__global__ void bfs(const bool *G, bool *F, bool *F_next, int *dist, const int V, int level, int *has_active)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= V)
        return;

    if (F[idx])
    {
        for (int i = 0; i < V; i++)
        {
            if (G[idx * V + i] && dist[i] == -1)
            {
                atomicOr(has_active, 1);
                dist[i] = level + 1;
                F_next[i] = true;
            }
        }
        F[idx] = false;
    }
}

void add_edge(bool *G, int u, int v, int V)
{
    G[u * V + v] = true;
    // G[v * V + u] = true; // uncomment for undirected graph
}

void populate_graph(bool *G, int V)
{
    // Initialize all to false
    memset(G, false, V * V * sizeof(bool));

    // Add edges - simple test graph
    //     0
    //    / \
    //   1   2
    //  / \
    // 3   4
    add_edge(G, 0, 1, V);
    add_edge(G, 0, 2, V);
    add_edge(G, 1, 3, V);
    add_edge(G, 1, 4, V);
}

int main()
{
    int V = 5, src = 0;
    int level = 0;
    bool *G = (bool *)malloc(V * V * sizeof(bool));
    bool *F = (bool *)malloc(V * sizeof(bool));
    int *dist = (int *)malloc(V * sizeof(int));
    int *has_active = (int *)malloc(sizeof(int));

    populate_graph(G, V);

    for (int i = 0; i < V; i++)
    {
        dist[i] = -1;
        F[i] = false;
    }
    F[src] = true;
    dist[src] = 0;

    bool *d_G, *d_F, *d_F_next;
    int *d_dist;
    int *d_has_active;

    cudaMalloc(&d_G, V * V * sizeof(bool));
    cudaMalloc(&d_F, V * sizeof(bool));
    cudaMalloc(&d_F_next, V * sizeof(bool));
    cudaMalloc(&d_dist, V * sizeof(int));
    cudaMalloc(&d_has_active, sizeof(int));

    cudaMemcpy(d_G, G, V * V * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, V * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, V * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridDim(((V + 31) / 32), 1);
    dim3 blockDim(32, 1);

    do
    {
        cudaMemset(d_has_active, 0, sizeof(int));
        cudaMemset(d_F_next, false, V * sizeof(bool));

        bfs<<<gridDim, blockDim>>>(d_G, d_F, d_F_next, d_dist, V, level, d_has_active);

        cudaDeviceSynchronize();
        std::swap(d_F, d_F_next);

        cudaMemcpy(has_active, d_has_active, sizeof(int), cudaMemcpyDeviceToHost);

        level++;

    } while (has_active[0]);

    cudaMemcpy(dist, d_dist, V * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Distances from source:\n");
    for (int i = 0; i < V; i++)
    {
        printf("dist[%d] = %d\n", i, dist[i]);
    }

    cudaFree(d_G);
    cudaFree(d_F);
    cudaFree(d_F_next);
    cudaFree(d_dist);
    cudaFree(d_has_active);

    free(G);
    free(F);
    free(dist);
}