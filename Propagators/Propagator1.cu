#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <csignal> 
#include "../Nets/Strides/Stride1.cuh"
#include "../Activations/Activation1.cuh"
#include "Propagator1.cuh"
#include "../Nets/Net1.h"

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

// Function that runs in the GPU
__global__ void updateNetwork(double *aa, double *bb, double *cc, double *dd, double* ee, double *ff, int *gg, int *hh, double *ii, int numLayers, int layer_IDX, int activation_type, bool forward)
{
    if (forward)
    {
        int nodeIDX = blockIdx.x*blockDim.x + threadIdx.x; // Index of a node.
        int connectionIDX = blockIdx.y*blockDim.y + threadIdx.y; // Index of connection leaving the node.
        if (nodeIDX < hh[layer_IDX] && connectionIDX < hh[layer_IDX + 1])
        {
            int IDX1 = DeviceIndexer1::internodal_data_list_indexer(nodeIDX,layer_IDX,connectionIDX,hh); // Index weights in previous(current) layer.
            int IDX2 = DeviceIndexer1::nodal_data_list_indexer(nodeIDX,layer_IDX,hh); // Index nodes in previous layer.

            int IDX3 = DeviceIndexer1::nodal_data_list_indexer(connectionIDX,layer_IDX+1,hh); // Index node in next layer.
            double w;
            if (layer_IDX == 0)
            {
                w = bb[IDX1]*aa[IDX2]; // Activation function is not applied to input nodes.
            }
            else
            { 
                double prev_a = DeviceActivation1::activation_function(aa[IDX2]+cc[IDX2], activation_type, false); // Compute activation of previous layer.
                w = bb[IDX1]*prev_a; // Compute data for next node withouth appling the activation. 
            };

            atomicAdd(&dd[IDX3], w); // All threads add to existing node data after node has been cleared (due to code above). 
        };
    }
    else // Layer indexing is done backwards.
    {
        int connectionIDX = blockIdx.x*blockDim.x + threadIdx.x; // Index of connection leaving a node.
        int nodeIDX = blockIdx.y*blockDim.y + threadIdx.y; // Node's index.

        int IDX2 = DeviceIndexer1::nodal_data_list_indexer(connectionIDX,(numLayers-1)-layer_IDX,hh); // Node's index in next(current) layer.

        int IDX3 = DeviceIndexer1::internodal_data_list_indexer(nodeIDX,(numLayers-1)-(layer_IDX+1),connectionIDX,hh); // Index weights in previous layer.
        int IDX4 = DeviceIndexer1::nodal_data_list_indexer(nodeIDX,(numLayers-1)-(layer_IDX+1),hh); // Index node and biases in previous layer.

        int IDXX = DeviceIndexer1::nodal_data_list_indexer(nodeIDX,(numLayers-1)-layer_IDX,hh); // For indexing bias vector at last layer.

        if (nodeIDX < hh[(numLayers-1)-(layer_IDX+1)] && connectionIDX < hh[(numLayers-1)-layer_IDX])
        {
            if (layer_IDX == 0)
            {
                double w_val_last = DeviceActivation1::activation_function(aa[IDX2] + cc[IDX2], activation_type, false); // Computes activation of last layer
                double delta_a_last = 2.*(w_val_last - ii[connectionIDX]); // compute dC/da of last layer.
                dd[IDX2] = delta_a_last;

                ff[IDXX] = delta_a_last*DeviceActivation1::activation_function(aa[IDX4]*bb[IDX3], activation_type, true); // compute dC/df of last layer;
            };
            double w_val;
            double a_val;
            if (layer_IDX != (numLayers-1)) // Biases and activations should not be used in determining the values of nodes in first layer. (Indexing layers backwards)
            {
                w_val = DeviceActivation1::activation_function(aa[IDX4] + cc[IDX4], activation_type, false); //Computes activation of node in previous layer. 
                a_val = w_val*bb[IDX3] + cc[IDX4];
            }
            else
            {
                w_val = aa[IDX4];
                a_val = w_val*bb[IDX3];
            };
            double delta_a = dd[IDX2]*DeviceActivation1::activation_function(a_val, activation_type, true)*bb[IDX3]; // Derivative of cost function wrt previous node.
            double delta_b = dd[IDX2]*DeviceActivation1::activation_function(a_val, activation_type, true)*w_val;
            double delta_c = dd[IDX2]*DeviceActivation1::activation_function(a_val, activation_type, true);

            atomicAdd(&ee[IDX3], delta_b);
            atomicAdd(&dd[IDX4], delta_a); // All treads add to existing node and biases.

            if ((layer_IDX+1) < (numLayers-1))
            {
                atomicAdd(&ff[IDX4], delta_c); // Dont update biases in first layer. They are placeholders. 
            };
        }
        else
        {
            // Do nothing.
        };
    };
}

namespace Propagator1
{
    void propagate_network(vector<double> &old_nodes, vector<double> &old_weights, vector<double> &old_biases, vector<double> &new_nodes, vector<double> &new_weights, vector<double> &new_biases, vector<int> &active_internode_connections, vector<int> &architecture, int &numLayers, int &layerIDX, bool &forward, int &sampleIDX, int &setIDX, NeuralNet1::Net1 &NET, vector<vector<vector<double>>> &raw_fit_data, int &num_layers, int &activation_type)
    {   
        //create points to the GPU
        double *cudaA;
        double *cudaB;
        double *cudaC;
        double *cudaD;
        double *cudaE;
        double *cudaF;
        double *cudaI;
    
        int *cudaG;
        int *cudaH;

        int numNodesTotal = NET.nodes_length(numLayers, architecture);
        int numConnectTotal = NET.weights_length(numLayers, architecture);

        int fitSize = raw_fit_data[setIDX][sampleIDX].size();

        // Allocate memory in the GPU
        if (cudaMalloc((void**)&cudaA, numNodesTotal*sizeof(double)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
        return;
        }
        if (cudaMalloc((void**)&cudaB, numConnectTotal*sizeof(double)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
        return;
        }
        if (cudaMalloc((void**)&cudaC, numNodesTotal*sizeof(double)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
        return;
        }

        if (cudaMalloc((void**)&cudaD, numNodesTotal*sizeof(double)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
        return;
        }
        if (cudaMalloc((void**)&cudaE, numConnectTotal*sizeof(double)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
        return;
        }
        if (cudaMalloc((void**)&cudaF, numNodesTotal*sizeof(double)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
        return;
        }
    
        if (cudaMalloc((void**)&cudaG, numConnectTotal*sizeof(int)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
        return;
        }
        if (cudaMalloc((void**)&cudaH, numLayers*sizeof(int)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
        return;
        }
        if (cudaMalloc((void**)&cudaI, fitSize*sizeof(double)) != cudaSuccess)
        {
	        cout<<"Data could not be allocated!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
        return;
        }

        // Add vectors to GPU for processing
        if (cudaMemcpy(cudaA, old_nodes.data(), numNodesTotal*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }
        if (cudaMemcpy(cudaB, old_weights.data(), numConnectTotal*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }       
        if (cudaMemcpy(cudaC, old_biases.data(), numNodesTotal*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }
        if (cudaMemcpy(cudaD, new_nodes.data(), numNodesTotal*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }
        if (cudaMemcpy(cudaE, new_weights.data(), numConnectTotal*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }
        if (cudaMemcpy(cudaF, new_biases.data(), numNodesTotal*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }
        if (cudaMemcpy(cudaG, active_internode_connections.data(), numConnectTotal*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }  
        if (cudaMemcpy(cudaH, architecture.data(), numLayers*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        } 
        if (cudaMemcpy(cudaI, raw_fit_data[setIDX][sampleIDX].data(), fitSize*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        } 
    
        // Run function. Specify grid size and block size
        int threadsPerBlock = 16;
        int blocksPerGrid1;
        int blocksPerGrid2;

        // Number of threads are determined based on the number of connections between current layer and next layer (propagating forward), or previous layer and current layer (propagating backwards).
        if (forward)
        {
            blocksPerGrid1 = architecture[layerIDX] / threadsPerBlock + 1; 
            blocksPerGrid2 = architecture[layerIDX+1] / threadsPerBlock + 1;
        }
        else
        {
            blocksPerGrid1 = architecture[(numLayers - 1) - layerIDX] / threadsPerBlock + 1;
            blocksPerGrid2 = architecture[(num_layers - 1) - (layerIDX + 1)] / threadsPerBlock + 1;
        };

        dim3 blockDIM(threadsPerBlock, threadsPerBlock);
        dim3 gridDIM(blocksPerGrid1, blocksPerGrid2);

        int _activation_type = activation_type;
    
        updateNetwork<<< gridDIM, blockDIM >>>(cudaA, cudaB, cudaC, cudaD, cudaE, cudaF, cudaG, cudaH, cudaI, numLayers, layerIDX, _activation_type, forward);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Copy vectors from GPU
        if (cudaMemcpy(new_nodes.data(), cudaD, numNodesTotal*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }
        if (cudaMemcpy(new_weights.data(), cudaE, numConnectTotal*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }
        if (cudaMemcpy(new_biases.data(), cudaF, numNodesTotal*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
            cudaFree(cudaD);
            cudaFree(cudaE);
            cudaFree(cudaF);
            cudaFree(cudaG);
            cudaFree(cudaH);
            cudaFree(cudaI);
	        return;
        }

        // //Unallocate data on device
        cudaFree(cudaA);
	    cudaFree(cudaB);
        cudaFree(cudaC);
        cudaFree(cudaD);
        cudaFree(cudaE);
        cudaFree(cudaF);
        cudaFree(cudaG);
        cudaFree(cudaH);
        cudaFree(cudaI);    
    }
}