#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector> 
#include <list>

using namespace std;

namespace HostIndexer1
{
    int internodal_data_list_indexer(int node_ID, int layer_ID, int connection_ID, vector<int> &architecture)
    {
        int IDX = (node_ID*architecture[layer_ID+1]) + connection_ID; // architecture[layer] returns the number of nodes in a given layer. Since network is fully connected, the number of connections from a node is equal to the number of nodes in the next layer, which is indexable using architecture[layer + 1].
        int iter = 0;
    start:
        if (iter == (layer_ID))
        {
            return IDX;
        }
        else
        {   
            IDX += architecture[iter]*architecture[iter+1]; // Totals the number of connections in layers prior to indexed layer. Adds to number of connections in nodes prior to index.
            iter++;
            goto start;
        }
    }

    int nodal_data_list_indexer(int node_ID, int layer_ID, vector<int> &architecture)        
    {
        int IDX = node_ID;
        int iter = 0;
    start:
        if (iter == (layer_ID))
        {
            return IDX;
        }
        else
        {   
            IDX += architecture[iter]; // Totals the number of nodes in layers prior to indexed layer. Adds to node index.
            iter++;
            goto start;
        }
    }
}
