#ifndef PATTERN1
#define PATTERN1

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector> 
#include <list>

using namespace std;

namespace NeuralNetPattern1
{
    class Pattern1
    {
        public:
            Pattern1(int type, vector<int> _architecture, int _architecture_len);
            vector<int> get_active_nodes(int _length);

            int terminate_outgoing_connection_to_center_nodes(int node, int layer);
            int terminate_all_outgoing_connections_from_node(int node, int layer);

            int loop_all_nodes(int layerIDX);
            int loop_all_layers();
            int flip_nodes_loop();
        
        private:
            vector<int> active_connections;
            int connectType;
            vector<int> architecture;
            int architecture_len;
    };
}

#endif