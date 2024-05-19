#include "RunNet1.cuh"

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ int internodal_data_list_indexer_(int node_ID, int layer_ID, int connection_ID, int *architecture)
{
    int IDX = (node_ID*architecture[layer_ID+1]) + connection_ID; // architecture[layer] returns the number of nodes in a given layer. Since network is fully connected, the number of connections from a node is equal to the number of nodes in the next layer, which is indexable using architecture[layer + 1].
    int iter = 0;
start:
    if (iter == (layer_ID))
    {
        return IDX;
    }
    else
    {   
        IDX += architecture[iter]*architecture[iter+1]; // Totals the number of connections in layers prior to indexed layer. Adds to number of connections in nodes prior to index.
        iter++;
        goto start;
    }
}

__device__ int nodal_data_list_indexer_(int node_ID, int layer_ID, int *architecture)
{
    int IDX = node_ID;
    int iter = 0;
start:
    if (iter == (layer_ID))
    {
        return IDX;
    }
    else
    {   
        IDX += architecture[iter]; // Totals the number of nodes in layers prior to indexed layer. Adds to node index.
        iter++;
        goto start;
    }
}

__device__ double activation_function(double w_val, int type)
{
    if (type == 0)
    {
        return 1./(1. + exp(-w_val));
    }
    else if (type == 1)
    {
        if (w_val <= 0)
        {
            return 0.0;
        } 
        else
        {
            return w_val;
        }; 
    };
    return 0.;
}

double _activation_function(double w_val, int type)
{
    if (type == 0)
    {
        return 1./(1. + exp(-w_val));
    }
    else if (type == 1)
    {
        if (w_val <= 0)
        {
            return 0.0;
        } 
        else
        {
            return w_val;
        }; 
    }
    else
    {
        std::cout<<"cannot process type!";
    };
    return 0.;
}

// Function that runs in the GPU
__global__ void updateNetwork(double *aa, double *bb, double *cc, double *dd, int *hh, int layer_IDX, int activation_type)
{
    int nodeIDX = blockIdx.x*blockDim.x + threadIdx.x; // Index of a node.
    int connectionIDX = blockIdx.y*blockDim.y + threadIdx.y; // Index of connection leaving the node.
    if (nodeIDX < hh[layer_IDX] && connectionIDX < hh[layer_IDX + 1])
    {
        int IDX1 = internodal_data_list_indexer_(nodeIDX,layer_IDX,connectionIDX,hh); // Index weights and biases in previus layer.
        int IDX2 = nodal_data_list_indexer_(nodeIDX,layer_IDX,hh); // Index nodes in previous layer.

        int IDX3 = nodal_data_list_indexer_(connectionIDX,layer_IDX+1,hh); // Index node in next layer.
        double w;
        if (layer_IDX == 0)
        {
            w = bb[IDX1]*aa[IDX2]; // Activation function is not applied to input nodes.
        }
        else
        { 
            double prev_a = activation_function(aa[IDX2]+cc[IDX2], activation_type); // Compute activation of previous layer.
            w = bb[IDX1]*prev_a; // Compute data for next node withouth appling the activation. 
        };

        dd[IDX3] = 0; // Clears node data. Will only be executed once due to race conditions, other threads will ignore.
        atomicAdd(&dd[IDX3], w); // All threads add to existing node data after node has been cleared (due to code above). 
    };
}

void RunNet1::propagate_network(vector<double> &old_nodes, vector<double> &old_weights, vector<double> &old_biases, vector<double> &new_nodes, vector<int> &architecture, int &layerIDX, int numLayers)
{   
    //create points to the GPU
    double *cudaA;
    double *cudaB;
    double *cudaC;
    double *cudaD;

    int *cudaH;

    // Allocate memory in the GPU
    if (cudaMalloc((void**)&cudaA, nodes_length*sizeof(double)) != cudaSuccess)
    {
	    cout<<"Data could not be allocated!";
    return;
    }
    if (cudaMalloc((void**)&cudaB, weights_length*sizeof(double)) != cudaSuccess)
    {
	    cout<<"Data could not be allocated!";
	    cudaFree(cudaA);
    return;
    }
    if (cudaMalloc((void**)&cudaC, nodes_length*sizeof(double)) != cudaSuccess)
    {
	    cout<<"Data could not be allocated!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
    return;
    }

    if (cudaMalloc((void**)&cudaD, nodes_length*sizeof(double)) != cudaSuccess)
    {
	    cout<<"Data could not be allocated!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
    return;
    }
    
    if (cudaMalloc((void**)&cudaH, numLayers*sizeof(int)) != cudaSuccess)
    {
	    cout<<"Data could not be allocated!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
    return;
    }

    // Add vectors to GPU for processing
    if (cudaMemcpy(cudaA, old_nodes.data(), nodes_length*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
	    cout<<"Alloacted memory did not accept supplied data!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
        cudaFree(cudaH);
	    return;
    }
    if (cudaMemcpy(cudaB, old_weights.data(), weights_length*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
	    cout<<"Alloacted memory did not accept supplied data!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
        cudaFree(cudaH);
	    return;
    }       
    if (cudaMemcpy(cudaC, old_biases.data(), nodes_length*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
	    cout<<"Alloacted memory did not accept supplied data!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
        cudaFree(cudaH);
	    return;
    }
    if (cudaMemcpy(cudaD, new_nodes.data(), nodes_length*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    {
	    cout<<"Alloacted memory did not accept supplied data!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
        cudaFree(cudaH);
	    return;
    }

    if (cudaMemcpy(cudaH, architecture.data(), numLayers*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    {
	    cout<<"Alloacted memory did not accept supplied data!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
        cudaFree(cudaH);
	    return;
    } 
    
    // Run function. Specify grid size and block size
    int threadsPerBlock = 16;
    int blocksPerGrid1;
    int blocksPerGrid2;

    // Number of threads are determined based on the number of connections between current layer and next layer (propagating forward), or previous layer and current layer (propagating backwards).
    blocksPerGrid1 = architecture[layerIDX] / threadsPerBlock + 1; 
    blocksPerGrid2 = architecture[layerIDX+1] / threadsPerBlock + 1;

    dim3 blockDIM(threadsPerBlock, threadsPerBlock);
    dim3 gridDIM(blocksPerGrid1, blocksPerGrid2);
    
    updateNetwork<<< gridDIM, blockDIM >>>(cudaA, cudaB, cudaC, cudaD, cudaH, layerIDX, act_type);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy vectors from GPU
    if (cudaMemcpy(new_nodes.data(), cudaD, nodes_length*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
	    cout<<"Alloacted memory did not accept supplied data!";
	    cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
        cudaFree(cudaH);
	    return;
    }

    // //Unallocate data on device
    cudaFree(cudaA);
	cudaFree(cudaB);
    cudaFree(cudaC);
    cudaFree(cudaD);
    cudaFree(cudaH);
}

RunNet1::RunNet1()
{
    hasRan = false;
    isSet = false;
}

int RunNet1::weights_l(int number_of_layers, vector<int> architecture)
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

int RunNet1::nodes_l(int number_of_layers, vector<int> architecture)
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
        the_length += architecture[idx]; // Add number of nodes in each layer.
        idx++;
        goto start;
    };
}

int RunNet1::clear_set_input_nodes(vector<double> inputs)
{
    int ID = 0;
start:
    if (ID == nodes.size())
    {
        return 0;
    }
    else
    {
        if (ID < input_node_size)
        {
            nodes[ID] = inputs[ID]; // Sets the values of the nodes in the first layer.
        }
        else
        {
            nodes[ID] = 0; // Sets the nodes in all other layers to zero.
        };
        ID++;
        goto start;
    };
}

void RunNet1::set_net(vector<double> _inputs, vector<double> _weights, vector<double> _biases, vector<int> _architecture, int activation_type)
{
    architecture = _architecture;
    weights = _weights;
    biases = _biases;
    act_type = activation_type;
    vector<double> inp = _inputs;
    input_node_size = inp.size();

    nodes_length = nodes_l(architecture.size(), architecture);
    weights_length = weights_l(architecture.size(), architecture);

    vector<double> _nodes(nodes_length, 0.);
    nodes = _nodes;

    if (weights.size() != weights_length)
    {
        std::cout<<"Incorrect inputs";
    }
    else if (biases.size() != nodes_length)
    {
        std::cout<<"Incorrect inputs";
    }
    else
    {
        clear_set_input_nodes(_inputs);
        isSet = true;
    };
}

int _nodal_data_list_indexer_(int node_ID, int layer_ID, vector<int> architecture)
{
    int IDX = node_ID;
    int iter = 0;
start:
    if (iter == (layer_ID))
    {
        return IDX;
    }
    else
    {   
        IDX += architecture[iter]; // Totals the number of nodes in layers prior to indexed layer. Adds to node index.
        iter++;
        goto start;
    }
}

double RunNet1::get_node(int layerID, int nodeID)
{
    if (layerID < (architecture.size()) && nodeID < architecture[layerID])
    {
        int IDX = _nodal_data_list_indexer_(nodeID, layerID, architecture);
        double pre_a = nodes[IDX];
        if (layerID != 0)
        {
            int IDX2 = _nodal_data_list_indexer_(nodeID, layerID-1, architecture);
            std::cout<<pre_a;
            return _activation_function(pre_a + biases[IDX2], act_type);
        }
        return pre_a;
    };
    std::cout<<"You have supplied an incorrect index. No connection between nodes exist.";
    return 0;
}

int RunNet1::propagate_layer()
{
    int layerIDX = 0;
start:
    if (layerIDX == architecture.size())
    {
        return 0;
    }
    else
    {
        vector<double> old_nodes;
        vector<double> old_weights;
        vector<double> old_biases;

        vector<double> new_nodes(nodes_length, 0.);

        old_nodes = nodes;
        old_weights = weights;
        old_biases = biases;

        propagate_network(old_nodes, old_weights, old_biases, new_nodes, architecture, layerIDX, architecture.size());
        layerIDX++;
        goto start;
    };
}

void RunNet1::propagate()
{
    if (isSet)
    {
        propagate_layer();
        hasRan = true;
    }
    else
    {
        std::cout<<"Must set data first!";
    };
}