#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <csignal> 
#include "../Nets/Strides/Stride1.cuh"
#include "../Activations/Activation1.cuh"

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
            int IDX1 = internodal_data_list_indexer(nodeIDX,layer_IDX,connectionIDX,hh); // Index weights in previous(current) layer.
            int IDX2 = nodal_data_list_indexer(nodeIDX,layer_IDX,hh); // Index nodes in previous layer.

            int IDX3 = nodal_data_list_indexer(connectionIDX,layer_IDX+1,hh); // Index node in next layer.
            double w;
            if (layer_IDX == 0)
            {
                w = bb[IDX1]*aa[IDX2]; // Activation function is not applied to input nodes.
            }
            else
            { 
                double prev_a = activation_function(aa[IDX2]+cc[IDX2], activation_type, false); // Compute activation of previous layer.
                w = bb[IDX1]*prev_a; // Compute data for next node withouth appling the activation. 
            };

            atomicAdd(&dd[IDX3], w); // All threads add to existing node data after node has been cleared (due to code above). 
        };
    }
    else // Layer indexing is done backwards.
    {
        int connectionIDX = blockIdx.x*blockDim.x + threadIdx.x; // Index of connection leaving a node.
        int nodeIDX = blockIdx.y*blockDim.y + threadIdx.y; // Node's index.

        int IDX2 = nodal_data_list_indexer(connectionIDX,(numLayers-1)-layer_IDX,hh); // Node's index in next(current) layer.

        int IDX3 = internodal_data_list_indexer(nodeIDX,(numLayers-1)-(layer_IDX+1),connectionIDX,hh); // Index weights in previous layer.
        int IDX4 = nodal_data_list_indexer(nodeIDX,(numLayers-1)-(layer_IDX+1),hh); // Index node and biases in previous layer.

        int IDXX = nodal_data_list_indexer(nodeIDX,(numLayers-1)-layer_IDX,hh); // For indexing bias vector at last layer.

        if (nodeIDX < hh[(numLayers-1)-(layer_IDX+1)] && connectionIDX < hh[(numLayers-1)-layer_IDX])
        {
            if (layer_IDX == 0)
            {
                double w_val_last = activation_function(aa[IDX2] + cc[IDX2], activation_type, false); // Computes activation of last layer
                double delta_a_last = 2.*(w_val_last - ii[connectionIDX]); // compute dC/da of last layer.
                dd[IDX2] = delta_a_last;

                ff[IDXX] = delta_a_last*activation_function(aa[IDX4]*bb[IDX3], activation_type, true); // compute dC/df of last layer;
            };
            double w_val;
            double a_val;
            if (layer_IDX != (numLayers-1)) // Biases and activations should not be used in determining the values of nodes in first layer. (Indexing layers backwards)
            {
                w_val = activation_function(aa[IDX4] + cc[IDX4], activation_type, false); //Computes activation of node in previous layer. 
                a_val = w_val*bb[IDX3] + cc[IDX4];
            }
            else
            {
                w_val = aa[IDX4];
                a_val = w_val*bb[IDX3];
            };
            double delta_a = dd[IDX2]*activation_function(a_val, activation_type, true)*bb[IDX3]; // Derivative of cost function wrt previous node.
            double delta_b = dd[IDX2]*activation_function(a_val, activation_type, true)*w_val;
            double delta_c = dd[IDX2]*activation_function(a_val, activation_type, true);

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