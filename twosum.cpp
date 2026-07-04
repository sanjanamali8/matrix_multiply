vector<pair<int, int>> pairsum(vector<int> &arr, int target)
{
    unordered_map<int, int> hm;

    vector<pair<int, int>> ans;

    for (int i = 0; i < arr.size(); i++)
    {
        if (hm.find(target - arr[i]) != hm.end())
        {
            ans.push_back({arr[i], target - arr[i]});
        }
        hm[arr[i]] = i;
    }
    return ans;
}
