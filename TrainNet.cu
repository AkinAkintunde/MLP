#include "TrainNet.h"
#include "../Nets/Net1.h"
#include "../Checker/Checker.h"

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include <csignal> 

using namespace std;

void TrainNet::set_training_data(vector<vector<vector<double>>> training_data, vector<vector<vector<double>>> fit_data, int _type, vector<int> _architecture, double _learning_rate, int _activation_type)
{
    // TODO architecture should match input and fit data
    
    raw_training_data = training_data;
    raw_fit_data = fit_data;
    type = _type;
    activation_type = _activation_type;

    max_iter = raw_training_data.size();
    prevIter = 0;

    num_of_samples = training_data[0].size();
    architecture = _architecture;
    learning_rate = _learning_rate;

    vector<double> _training_data;
    vector<double> _fit_data;
        
    _training_data = raw_training_data[0][0];
    _fit_data = raw_fit_data[0][0];

    int raw_training_data_size = Checker::set_checker(raw_training_data, raw_training_data.size());
    int raw_fit_data_size = Checker::set_checker(raw_fit_data, raw_fit_data.size());

    if (_fit_data.size() != architecture[architecture.size()-1])
    {
        isSet = false;
        std::cout<<"architecture has incorrect dimensions!\n";
    }
    else if (_training_data.size() != architecture[0])
    {
        isSet = false;
        std::cout<<"architecture has incorrect dimensions!\n";
    }
    else if (raw_fit_data_size != (max_iter*num_of_samples*_fit_data.size()))
    {
        isSet = false;
        std::cout<<"Fit data has incorrect dimensions!\n";
    }
    else if (raw_training_data_size != (max_iter*num_of_samples*_training_data.size()))
    {
        isSet = false;
        std::cout<<"Training data has incorrect dimensions!\n";
    }
    else
    {
        NeuralNet1::Net1 _NET;
        NET = _NET;
        NET.set_net(_training_data, _fit_data, _training_data.size(), _fit_data.size(), type, architecture, architecture.size());

        num_layers = NET.get_number_of_layers();

        vector<double> _cost(raw_fit_data[0][0].size(), 0.0);
        cost = _cost;
        isSet = true;
    };
}