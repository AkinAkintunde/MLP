#include "Net1.h"
#include "Patterns/Pattern1.h"
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector> 
#include <list>

using namespace std;
using namespace NeuralNet1;
using namespace NeuralNetPattern1;

Net1::Net1()
{
    netIsSet = false;
}

int Net1::allocate_nodes()
{
    int nodeID = 0;
start:
    if (nodeID == input_node_size)
    {
        return 0;
    }
    else
    {
        data.nodes[nodeID] = input[nodeID];
        nodeID++;
        goto start;
    };
}

int Net1::clear_set_input_nodes(vector<double> _training_data, vector<double> _fit_data)
{
    int ID = 0;
    fit_nodes = _fit_data;
start:
    if (ID == data.nodes.size())
    {
        return 0;
    }
    else
    {
        if (ID < input_node_size)
        {
            data.nodes[ID] = _training_data[ID]; // Sets the values of the nodes in the first layer.
        }
        else
        {
            data.nodes[ID] = 0; // Sets the nodes in all other layers to zero.
        };
        ID++;
        goto start;
    };
}

vector<double> Net1::set_output_nodes()
{
    int ID = 0;
    
start:
    if (ID == number_of_fit_nodes)
    {
        
        return data.nodes; 
    }
    else
    {
        data.nodes[(data.nodes.size()-number_of_fit_nodes)+ID] = fit_nodes[ID];
    };
    ID++;
    goto start;
}

int Net1::weights_length(int number_of_layers, vector<int> architecture)
{
    int idx = 0;
    int the_length = 0;
start:
    if (idx == (number_of_layers - 1))
    {
        return the_length;
    }
    else
    {
        the_length += architecture[idx]*architecture[idx + 1]; // Add number of connections in each layer. Architecture hold the number of nodes in each layer.
        idx++;
        goto start;
    };
}

int Net1::nodes_length(int number_of_layers, vector<int> architecture)
{
    int idx = 0;
    int the_length = 0;
start:
    if (idx == (number_of_layers))
    {
        return the_length;
    }
    else
    {
        the_length += architecture[idx]; // Add number of modes in each layer.
        idx++;
        goto start;
    };
}

void Net1::set_net(vector<double> network_input_data, vector<double> network_fit_data, int input_size, int fit_size, int type, vector<int> architecture, int architecture_len)
{
    fit_nodes = network_fit_data;
    input = network_input_data;

    number_of_fit_nodes = fit_size;
    input_node_size = input_size;

    number_of_layers = architecture_len;

    Pattern1 NNP(type, architecture, architecture_len);

    int weights_shape = weights_length(number_of_layers, architecture);
    int nodes_shape = nodes_length(number_of_layers, architecture);

    vector<double> _nodes(nodes_shape, 0.);
    data.nodes = _nodes;
    vector<double> _weights(weights_shape, 0.1);
    data.weights = _weights;
    vector<double> _biases(nodes_shape, 0.03);  // Biases vector is equal in length to the weights vector.
    data.biases = _biases;

    allocate_nodes();

    vector<int> _active_connections = NNP.get_active_nodes(weights_shape); // Active connectioncs vector is the same shape as weights vector.
    active_connections = _active_connections;

    netIsSet = true;
} 

void Net1::update_nodes(vector<double> nodes_new_values)
{
    data.nodes = nodes_new_values;
}

void Net1::update_weights(vector<double> weights_new_values)
{
    data.weights = weights_new_values;
}

void Net1::update_biases(vector<double> biases_new_values)
{
    data.biases = biases_new_values;
}

vector<double> Net1::get_nodes()
{
    return data.nodes;
}

vector<double> Net1::get_weights()
{
    return data.weights;
}

vector<double> Net1::get_biases()
{
    return data.biases;
}

int Net1::get_number_of_layers()
{
    return number_of_layers;
}

vector<int> Net1::get_active_connections()
{
    return active_connections;
}