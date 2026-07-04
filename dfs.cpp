void dfs(vector<vector<int>> &adj, int u, vector<bool> &visited, vector<bool> &stack)
{
    stack[u] = true;
    visited[u] = true;

    for (int nei : adj[u])
    {
        if (!visited[nei])
        {
            dfs(adj, nei, visited, stack);
        }
        else if (stack[nei] == true)
        {
            // has cycle
        }
    }

    stack[u] = false;
}