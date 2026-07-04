vector<int> bfs(vector<vector<int>> &adj, int source, int n)
{
    queue<int> frontier;
    frontier.push(source);

    vector<int> distances(n);
    fill(distances.begin(), distances.end(), -1);
    distances[source] = 0;

    while (!frontier.empty())
    {
        int i = frontier.front();
        frontier.pop();
        for (int ne : adj[i])
        {
            if (distances[ne] == -1)
            {
                frontier.push(ne);
                distances[ne] = distances[i] + 1;
            }
        }
    }
    return distances;
}