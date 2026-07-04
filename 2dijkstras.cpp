vector < int > 2dijkstras(vector<vector<pair<int, int>>> &adj, int n, int src)
{
    vector<int> dist1(n, INT_MAX);
    vector<int> dist2(n, INT_MAX);

    dist1[src] = 0;

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, src});

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();

        if (dist2[u] < d)
            continue;

        for (auto [v, w] : adj[u])
        {
            int newd = d + w;

            if (newd < dist1[v])
            {
                dist2[v] = dist1[v];
                dist1[v] = newd;
                pq.push({dist1[v], v});
                if (dist2[v] != INT_MAX)
                    pq.push({dist2[v], v});
            }
            else if (newd < dist2[v] && newd != dist1[v])
            {
                dist2[v] = newd;
                pq.push({dist2[v], v});
            }
        }
    }
    return dist2;
}