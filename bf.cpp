vector<int> bf(vector<vector<pair<int, int>>> &adj, int src, int n)
{
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    bool flag = false;

    for (int i = 0; i < n - 1; i++)
    {
        for (int u = 0; u < n; u++)
        {
            for (auto [v, w] : adj[u])
            {
                if (dist[u] == INT_MAX && dist[v] > dist[u] + w)
                {
                    dist[v] = dist[u] + w;
                }
            }
        }
    }
    for (int u = 0; u < n; u++)
    {
        for (auto [v, w] : adj[u])
        {
            if (dist[v] > dist[u] + w)
            {
                flag = true; // negative cycle
            }
        }
    }
    return dist;
}