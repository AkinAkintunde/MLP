#ifndef PROPAGATOR1
#define PROPAGATOR1

#include "../Nets/Net1.h"

namespace Propagator1
{
    void propagate_network(vector<double> &old_nodes, vector<double> &old_weights, vector<double> &old_biases, vector<double> &new_nodes, vector<double> &new_weights, vector<double> &new_biases, vector<int> &active_internode_connections, vector<int> &architecture, int &numLayers, int &layerIDX, bool &forward, int &sampleIDX, int &setIDX, NeuralNet1::Net1 &NET, vector<vector<vector<double>>> &raw_fit_data, int &num_layers, int &activation_type);
}

#endif