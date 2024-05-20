#ifndef TRAINNET
#define TRAINNET

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include "Nets/Net1.h"

using namespace std;

class TrainNet 
{
    public:
        void set_training_data(vector<vector<vector<double>>> training_data, vector<vector<vector<double>>> fit_data, int _type, vector<int> _architecture, double _learning_rate, int _activation_type);
        int update_cost(int sampleIDX, int setIDX); 

        void compute_gpu(int iterations);
        int iterarions(int iters);
        int compute_samples(int set_IDX);
        int compute_layers(int sampleIDX, int setIDX, bool forward);

        vector<vector<vector<vector<string>>>> get_active_net_connections();
        int init_return_connection_data();
        int fill_return_layers();
        int fill_return_nodes(int layer_ID);
        int fill_return_connections(int node_ID, int layer_ID);

        vector<vector<string>> get_active_interconnections();
        int interconnections_layers_loop();
        int interconnections_nodes_loop(int layerID);
        int interconnections_connections_loop(int layerID, int nodeID);
        string getLetters(int num);

        double get_net_weights(int layerID, int nodeID, int connectionID);
        double get_net_biases(int layerID, int nodeID);
        double get_net_nodes(int layerID, int nodeID);
        vector<double> get_net_cost();

        vector<double> get_all_weights();
        vector<double> get_all_biases();

        void setOfficialWeightsBiases(vector<double> weights, vector<double> biases);
        double get_cost();

    private:
        vector<vector<vector<double>>> raw_training_data;
        vector<vector<vector<double>>> raw_fit_data;

        vector<double> results_delta_weights;
        vector<double> results_delta_biases;

        vector<double> results_weights;
        vector<double> results_biases;

        vector<int> architecture;

        vector<vector<vector<vector<string>>>> return_connection_data;
        vector<vector<string>>  active_interconnections;

        vector<vector<vector<double>>> return_weights_data;

        int type; // Network type
        int activation_type; // Type of activation function
        int max_iter;

        int prevIter;

        vector<double> cost;
        double learning_rate;

        int num_layers;
        int num_of_samples;
        double ccost;

        bool isSet = false;
        NeuralNet1::Net1 NET;
};

#endif