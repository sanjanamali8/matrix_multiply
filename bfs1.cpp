vector<int> bfs(vector<vector<int>> &adj, int n, int src)
{
    vector<int> dist(n, -1);
    dist[src] = 0;

    queue<int> fron;
    fron.push(src);

    while (!fron.empty())
    {
        int v = fron.front();
        fron.pop();

        for (int nei : adj[v])
        {
            if (dist[nei] == -1)
            {
                dist[nei] = dist[v] + 1;
                fron.push(nei);
            }
        }
    }
    return dist;
}