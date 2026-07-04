vector<int> topo(vector<vector<int>> &adj, int n)
{
    vector<int> ind(n, 0);

    for (int i = 0; i < adj.size(); i++)
    {
        for (int out : adj[i])
        {
            ind[out]++;
        }
    }

    queue<int> q;
    vector<int> ans;

    for (int i = 0; i < n; i++)
    {
        if (ind[i] == 0)
        {
            q.push(i);
        }
    }

    while (!q.empty())
    {
        int v = q.front();
        q.pop();
        ans.push_back(v);

        for (int nei : adj[v])
        {
            ind[nei]--;
            if (ind[nei] == 0)
            {
                q.push(nei);
            }
        }
    }

    if (ans.size() != n)
    {
        // cycle detected
    }
    return ans;
}