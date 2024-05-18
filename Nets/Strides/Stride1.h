#ifndef HOSTINDEXER1
#define HOSTINDEXER1

using namespace std;

namespace HostIndexer1
{
    int internodal_data_list_indexer(int node_ID, int layer_ID, int connection_ID, vector<int> &architecture);
    int nodal_data_list_indexer(int node_ID, int layer_ID, vector<int> &architecture);
}

#endif