vector<int> topoSort(vector<vector<int>> &adj, int n)
{
    vector<int> ind(n, 0);
    for (vector<int> u : adj)
    {
        for (int v : u)
        {
            ind[v]++;
        }
    }

    queue<int> q;
    for (int i = 0; i < n; i++)
    {
        if (ind[i] == 0)
        {
            q.push(i);
        }
    }

    vector<int> ans;

    while (!q.empty())
    {
        int top = q.front();
        ans.push_back(top);
        q.pop();
        for (int v : adj[top])
        {
            ind[v]--;
            if (ind[v] == 0)
            {
                q.push(v);
            }
        }
    }
    return ans;
}