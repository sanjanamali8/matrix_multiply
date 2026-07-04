void dfs(vector<vector<int>> &adj, int u, vector<bool> &visited, vector<bool> &st)
{
    visited[u] = true;
    st[u] = true;

    for (int nei : adj[u])
    {
        if (!visited[nei])
        {
            dfs(adj, nei, visited, st);
        }
        else if (st[nei])
        {
            // cycle detected
        }
    }
    st[u] = false;
    return;
}