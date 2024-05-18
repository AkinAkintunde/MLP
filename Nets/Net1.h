#ifndef NET1
#define NET1

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector> 
#include <list>

using namespace std;

namespace NeuralNet1
{
    class Net1
    {
        public:
            Net1();
            void set_net(vector<double> network_input_data, vector<double> network_fit_data, int input_size, int fit_size, int type, vector<int> architecture, int architecture_len);
            void update_nodes(vector<double> nodes_new_values);
            void update_weights(vector<double> weights_new_values);
            void update_biases(vector<double> biases_new_values);
            int allocate_nodes();

            int clear_set_input_nodes(vector<double> _training_data, vector<double> _fit_data);

            vector<double> get_nodes();
            vector<double> get_weights();
            vector<double> get_biases();

            vector<int> get_active_connections();
            vector<double> set_output_nodes();

            int weights_length(int number_of_layers, vector<int> architecture);
            int nodes_length(int number_of_layers, vector<int> architecture);
            int get_number_of_layers();

        private:
            vector<double> nodes;
            vector<double> weights;
            vector<double> biases;

            vector<double> cost_weights;
            vector<double> cost_bias;

            vector<double> fit_nodes;
            vector<double> input;

            vector<int> active_connections;

            bool netIsSet;

            int input_node_size;
            int number_of_layers;
            int number_of_fit_nodes;
    };
}
#endif