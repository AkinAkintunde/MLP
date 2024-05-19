#ifndef RUNNET1 
#define RUNNET1 

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class RunNet1 
{
    public:
        RunNet1();
        void set_net(vector<double> _inputs, vector<double> _weights, vector<double> _biases, vector<int> _architecture, int activation_type);
        void propagate_network(vector<double> &old_nodes, vector<double> &old_weights, vector<double> &old_biases, vector<double> &new_nodes, vector<int> &architecture, int &layerIDX, int numLayers);

        int nodes_l(int number_of_layers, vector<int> architecture);
        int weights_l(int number_of_layers, vector<int> architecture);
        int clear_set_input_nodes(vector<double> inputs);

        int propagate_layer();
        void propagate();

        double get_node(int layerID, int nodeID);
    private:
        vector<double> biases;
        vector<double> weights;
        vector<double> nodes;
        vector<int> architecture;
        bool hasRan;
        bool isSet;
        int nodes_length;
        int weights_length;
        int act_type;
        int input_node_size;
};

#endif