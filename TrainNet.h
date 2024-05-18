#ifndef TRAINNET
#define TRAINNET

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class TrainNet 
{
    public:
        void set_training_data(vector<vector<vector<double>>> training_data, vector<vector<vector<double>>> fit_data, int _type, vector<int> _architecture, double _learning_rate, int _activation_type);
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
        // NeuralNet1 NET;
};

#endif