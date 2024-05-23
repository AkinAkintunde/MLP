#include "TrainNet.cuh"
#include "Nets/Net1.h"
#include "Nets/Strides/Stride1.h"
#include "Unpacker/Unpacker.h"
#include "Operations/VecAdd.cuh"
#include "Operations/Clear.cuh"
#include "Activations/Activation1.h"
#include "Propagators/Propagator1.cuh"

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include <csignal> 

using namespace std;

void signal_handler(int signal_num) 
{ 
    std::cout<<"\r";
    std::cout<<"User terminated the program!\n"; 
    cudaDeviceReset();
    // It terminates the  program 
    exit(signal_num); 
} 

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

    int raw_training_data_size = Unpacker::vec_3d_size(raw_training_data, raw_training_data.size());
    int raw_fit_data_size = Unpacker::vec_3d_size(raw_fit_data, raw_fit_data.size());

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
        results_weights = NET.get_weights();
        results_biases = NET.get_biases();

        vector<double> _cost(raw_fit_data[0][0].size(), 0.0);
        cost = _cost;
        isSet = true;
    };
}

vector<double> updateOfficialWeightsBiases(vector<double> vecA, vector<double> delta_vecA, double t_rate, int num_samps)
{
    int iter = 0;
    vector<double> summ;
start:
    if (iter == vecA.size())
    {
        return summ;
    }
    else
    {
        summ.push_back(vecA[iter] - (t_rate * (delta_vecA[iter]/num_samps)));
        iter++;
        goto start;
    };
}

int TrainNet::update_cost(int sampleIDX, int setIDX) 
{
    int last_nodes_id = 0;
    int _max_iter = raw_fit_data[setIDX][sampleIDX].size();
    ccost = 0.0;
start:
    if (last_nodes_id == _max_iter)
    {
        std::cout<<" cost:"<<ccost<<"\r"<<std::flush;
        return 0;
    }
    else
    {
        int IDX = HostIndexer1::nodal_data_list_indexer(last_nodes_id, (num_layers - 1), architecture);
        double AA = HostActivation1::activation_function(NET.get_nodes()[IDX] + NET.get_biases()[IDX], activation_type, false);
        double BB = (AA - raw_fit_data[setIDX][sampleIDX][last_nodes_id]);
        double _ccost = BB*BB;
        cost[last_nodes_id] = _ccost;
        ccost += _ccost/_max_iter;
        last_nodes_id++;
        goto start;
    };
    return 0;
}

void TrainNet::compute_gpu(int iterations) // num_samples is the number of samples, a is the number of iter, 
                                                                              // c len of each test, d len of result, e number of layers
{
    int ITR = prevIter;

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = signal_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

start:
    if (ITR < (iterations+prevIter))
    {
        if (isSet)
        {
            iterarions(ITR);
        }
        else 
        {
            std::cout<<"Data must be set first!\n";
        };
        ITR++;
        goto start;
    }
    else
    {
        prevIter = ITR;
        std::cout<<"Done | final cost:"<<ccost;
    };
}

int TrainNet::iterarions(int iters)
{
    int iterID = 0;
start:
    if (iterID == max_iter)
    {
        return 0;
    }
    else
    {
        if (iterID == 0 && iters == 0)
        {
            results_weights = NET.get_weights();
            results_biases = NET.get_biases();
        };
        compute_samples(iterID);
        iterID++;
        goto start;
    };
}

int TrainNet::compute_samples(int set_IDX) 
{
    int sample_index = 0;
start:
    if (sample_index == num_of_samples)
    {
        vector<double> _results_weights = updateOfficialWeightsBiases(results_weights, results_delta_weights, learning_rate, num_of_samples);
        results_weights = _results_weights ; //offically update weights
        vector<double> _results_biases = updateOfficialWeightsBiases(results_biases, results_delta_biases, learning_rate, num_of_samples);
        results_biases = _results_biases ; //offically update biases
        return 0;
    } 
    else
    {
        vector<double> _training_data;
        vector<double> _fit_data;
        
        _training_data = raw_training_data[set_IDX][sample_index];
        _fit_data = raw_fit_data[set_IDX][sample_index];

        NET.clear_set_input_nodes(_training_data, _fit_data);

        NET.update_biases(results_biases); // Reset weights and biases for each sample run.
        NET.update_weights(results_weights);

        compute_layers(sample_index, set_IDX, true);
        compute_layers(sample_index, set_IDX, false);

        sample_index++;
        goto start;
    };
}

int TrainNet::compute_layers(int sampleIDX, int setIDX, bool forward)
{
    int layerID = 0;
start:
    if (layerID == (num_layers - 1))
    {
        if (!forward)
        {
            vector<double> _results_delta_weights;
            vector<double> _results_delta_biases;
            if (sampleIDX == 0)
            {
                _results_delta_weights = NET.get_weights();
                _results_delta_biases = NET.get_biases();
            }
            else
            {
                _results_delta_weights = AddVec::addVecGPU(NET.get_weights(), results_delta_weights); 
                _results_delta_biases = AddVec::addVecGPU(NET.get_biases(), results_delta_biases);
            };
            results_delta_weights = _results_delta_weights;
            results_delta_biases = _results_delta_biases;
        }
        else
        {
            if (sampleIDX == (num_of_samples - 1))
            {
                update_cost(sampleIDX, setIDX);
            };
        };
        return 0;
    }
    else
    {
        vector<double> temp_nodes = NET.get_nodes();
        vector<double> temp_weights = NET.get_weights();
        vector<double> temp_biases = NET.get_biases();

        vector<int> active_connections = NET.get_active_connections();

        int nodes_length = NET.nodes_length(num_layers, architecture);
        int weights_length = NET.weights_length(num_layers, architecture);

        vector<double> new_nodes = temp_nodes;
        vector<double> new_weights = temp_weights;
        vector<double> new_biases = temp_biases;

        bool tTrue = true;
        bool fFalse = false;
        
        if (forward)
        {
            // Clears data in next layer.
            new_nodes = Clear::clear_layer(new_nodes, layerID, architecture, num_layers, false, true);

            Propagator1::propagate_network(temp_nodes, temp_weights, temp_biases, new_nodes, new_weights, new_biases, active_connections, architecture, num_layers, layerID, tTrue, sampleIDX, setIDX, NET, raw_fit_data, num_layers, activation_type);
        }
        else
        {
            // Clears data in previous layer. Here, propagation is done backwards.
            new_nodes = Clear::clear_layer(new_nodes, layerID, architecture, num_layers, false, false);

            Propagator1::propagate_network(temp_nodes, temp_weights, temp_biases, new_nodes, new_weights, new_biases, active_connections, architecture, num_layers, layerID, fFalse, sampleIDX, setIDX, NET, raw_fit_data, num_layers, activation_type); // Propagate backwards, fFalse ensures layer is indexed backwards.
        };

        NET.update_nodes(new_nodes);
        NET.update_weights(new_weights);
        NET.update_biases(new_biases);

        layerID++;
        goto start;
    };
};

vector<vector<vector<vector<string>>>> TrainNet::get_active_net_connections()
{
    if (isSet)
    {
        return_connection_data.clear();
        init_return_connection_data();
        fill_return_layers();
        return return_connection_data;
    }
    else
    {
        std::cout<<"Must set data first!\n";
    };
    return {{{{{}}}}};
}

int TrainNet::init_return_connection_data()
{
    int _idx = 0;
start:
    if (_idx == (num_layers-1))
    {   
        return 0;
    }
    else
    {
        vector<string> tup(2, "**");
        vector<vector<string>> vectup(architecture[_idx+1], tup);

        vector<vector<vector<string>>> __return(architecture[_idx], vectup);
        return_connection_data.push_back(__return);
        _idx++;
        goto start;
    }
}

int TrainNet::fill_return_layers()
{
    int layer_ID = 0;
start:
    if (layer_ID == (num_layers-1))
    {
        return 0;
    }
    else
    {
        fill_return_nodes(layer_ID);
        layer_ID++;
        goto start;
    };
}

int TrainNet::fill_return_nodes(int layer_ID)
{
    int node_ID = 0;
start:
    if (node_ID == architecture[layer_ID])
    {
        return 0;
    }
    else
    {
        fill_return_connections(node_ID, layer_ID);
        node_ID++;
        goto start;
    };
}

int TrainNet::fill_return_connections(int node_ID, int layer_ID)
{
    int connection_ID = 0;
start:
    if (connection_ID == architecture[layer_ID+1])
    {
        return 0;
    }
    else
    {
        vector<vector<string>> active_connections = get_active_interconnections();
        
        int IDX = HostIndexer1::internodal_data_list_indexer(node_ID, layer_ID, connection_ID, architecture);

        return_connection_data[layer_ID][node_ID][connection_ID] = active_connections[IDX];

        connection_ID++;
        goto start;
    };
}

vector<vector<string>> TrainNet::get_active_interconnections()
{
    interconnections_layers_loop();
    return active_interconnections;
}

int TrainNet::interconnections_layers_loop()
{
    int i_layer = 0;
start:
    if (i_layer == (num_layers-1))
    {
        return 0;
    }
    else
    {
        interconnections_nodes_loop(i_layer);
        i_layer++;
        goto start;
    };
}

int TrainNet::interconnections_nodes_loop(int layerID)
{
    int i_node = 0;
start:
    if (i_node == architecture[layerID])
    {
        return 0;
    }
    else
    {
        interconnections_connections_loop(layerID, i_node);
        i_node++;
        goto start;
    };
}

int TrainNet::interconnections_connections_loop(int layerID, int nodeID)
{
    int i_connect = 0;
start:
    if (i_connect == architecture[layerID+1])
    {
        return 0;
    }
    else
    {
        int IDX = HostIndexer1::internodal_data_list_indexer(nodeID, layerID, i_connect, architecture);
        int isConnected = NET.get_active_connections()[IDX];
        if (isConnected == 0)
        {
            active_interconnections.push_back({getLetters(layerID)+getLetters(nodeID), getLetters(layerID+1)+getLetters(i_connect)});
        };
        i_connect++;
        goto start;
    };
}

string TrainNet::getLetters(int num)
{
    string strings = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int num2 = (int)(num/26);
    string string2;
    string string3(num2, (char) strings[num2]);
    string string4 = string2 + string3;
    string4.push_back(strings[num%26]);
    return string4;
}

double TrainNet::get_net_weights(int layerID, int nodeID, int connectionID)
{
    if (layerID < (architecture.size()-1) && nodeID < architecture[layerID])
    {
        if (connectionID < architecture[layerID + 1])
        {
            int IDX = HostIndexer1::internodal_data_list_indexer(nodeID, layerID, connectionID, architecture);
            return results_weights[IDX];
        };
    };
    std::cout<<"You have supplied an incorrect index. No connection between nodes exist.\n";
    return 0;
}

double TrainNet::get_net_biases(int layerID, int nodeID)
{
    if (layerID < architecture.size() && nodeID < architecture[layerID])
    {
        int IDX = HostIndexer1::nodal_data_list_indexer(nodeID, layerID, architecture);
        return results_biases[IDX];
    };
    std::cout<<"You have supplied an incorrect index. No connection between nodes exist.\n";
    return 0;
}

double TrainNet::get_net_nodes(int layerID, int nodeID)
{
    if (layerID < (architecture.size()) && nodeID < architecture[layerID])
    {
        int IDX = HostIndexer1::nodal_data_list_indexer(nodeID, layerID, architecture);
        double pre_a = NET.get_nodes()[IDX];
        if (layerID != 0)
        {
            int IDX2 = HostIndexer1::nodal_data_list_indexer(nodeID, layerID-1, architecture);
            std::cout<<pre_a;
            return HostActivation1::activation_function(pre_a + NET.get_biases()[IDX2], activation_type, false);
        }
        return pre_a;
    };
    std::cout<<"You have supplied an incorrect index. No connection between nodes exist.\n";
    return 0;
}

vector<double> TrainNet::get_net_cost()
{
    return cost;
}

vector<double> TrainNet::get_all_weights()
{
    return results_weights;
}

vector<double> TrainNet::get_all_biases()
{
    return results_biases;
}

void TrainNet::setOfficialWeightsBiases(vector<double> weights, vector<double> biases)
{
    if (isSet)
    {
        vector<double> _biases = biases;
        vector<double> _weights = weights;

        double length = NET.weights_length(num_layers, architecture);
        double length2 = NET.nodes_length(num_layers, architecture);

        if (_biases.size() == length2 && _weights.size() == length)
        {
            NET.update_biases(_biases);
            NET.update_weights(_weights);
        }
        else
        {
            std::cout<<"Incorrect configuration!\n";
        };
    }
    else
    {
        std::cout<<"Must set the network first!\n";
    };
}

double TrainNet::get_cost()
{
    return ccost;
}