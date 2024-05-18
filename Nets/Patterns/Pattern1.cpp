#include "Pattern1.h"
#include "../Strides/Stride1.h"

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;
namespace NeuralNetPattern1
{
    Pattern1::Pattern1(int type, vector<int> _architecture, int _architecture_len)
    {
        connectType = type;
        architecture = _architecture;
        architecture_len = _architecture_len;
    }

    int Pattern1::terminate_outgoing_connection_to_center_nodes(int node, int layer)
    {
        int connectionIDX = 0;
    start:
        if (connectionIDX == architecture[layer + 1]) // architecture[layer + 1] give the number of connections between a node in a layer and its neighbors in the next layer.
        {
            return 0;
        }
        else
        {
            if (connectionIDX != 0 && connectionIDX != architecture[layer + 1]) // Only disconnect inner connections.
            {
                int IDX = HostIndexer1::internodal_data_list_indexer(node, layer, connectionIDX, architecture);

                active_connections[IDX] = 1;
            };
            connectionIDX++;
            goto start;
        };
    }

    int Pattern1::terminate_all_outgoing_connections_from_node(int node, int layer)
    {
        int connectionIDX = 0;
    start:
        if (connectionIDX == architecture[layer + 1]) 
        {
            return 0;
        }
        else
        {
            int IDX = HostIndexer1::internodal_data_list_indexer(node, layer, connectionIDX, architecture);

            active_connections[IDX] = 1;
            connectionIDX++;
            goto start;
        }
    }

    int Pattern1::loop_all_nodes(int layerIDX)
    {
        int nodeIDX = 0;
    start:
        if (nodeIDX == architecture[layerIDX])
        {
            return 0;
        }
        else
        {
            terminate_outgoing_connection_to_center_nodes(nodeIDX, layerIDX);
            if (nodeIDX != 0 && nodeIDX != (architecture[layerIDX]-1)) // Only disconnect inner nodes.
            {
                terminate_all_outgoing_connections_from_node(nodeIDX, layerIDX);
            };
            nodeIDX++;
            goto start;
        };
    }

    int Pattern1::loop_all_layers()
    {
        int layerIDX = 0;
    start: 
        if (layerIDX == (architecture_len-1))
        {
            return 0;
        }
        else 
        {
            loop_all_nodes(layerIDX);
            layerIDX++;
            goto start;
        };
    }

    int Pattern1::flip_nodes_loop()
    {
        int IDX = 0;
    start:
        if (IDX == active_connections.size())
        {
            return 0;
        }
        else
        {
            if (active_connections[IDX] == 0)
            {
                active_connections[IDX] = 1;
            }
            else
            {
                active_connections[IDX] = 0;
            };
            IDX++;
            goto start;
        };
    }

    vector<int> Pattern1::get_active_nodes(int _length)
    {
        vector<int> _active_connections(_length, 0);
        active_connections = _active_connections;

        // TODO connection types are depreciated.
        if (connectType == 1)
        {
            loop_all_layers();
            std::cout<<"This connection type is broken, use another!";
        }
        else if (connectType == 2)
        {
            loop_all_layers();
            flip_nodes_loop(); // output must be larger than 3. //TODO fix issues with connections to inactive nodes.
            std::cout<<"This connection type is broken, use another!";
        };
        return active_connections;
    }
}