#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include <csignal> 
#include "Clear.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void clear_layer_kernel(double *aA, double *bB, bool interconnection, int clrLayer, int length, int *cC)
{
    int node = blockIdx.x*blockDim.x + threadIdx.x;
    int IDX;

    if (interconnection)
    {
        int connection = blockIdx.y*blockDim.y + threadIdx.y;
        IDX = internodal_data_list_indexer(node, clrLayer, connection, cC);
    }
    else
    {
        IDX = nodal_data_list_indexer(node, clrLayer, cC);
    };

    if (IDX < length)
    {
        bB[IDX] = 0.0;
    };
}

namespace Clear
{
    vector<double> clear_layer(vector<double> A, int _layer, vector<int> architecture, int numb_layers, bool interconnection, bool forward)
    {
        double *cudaA;
        double *cudaB;
        int *cudaC;

        int layer = (numb_layers-1) - _layer;

        vector<double> vecA = A;

        int vecLength = vecA.size();

        vector<double> vecB = A;

        // Allocate memory to gpu.
        if (cudaMalloc((void**)&cudaA, vecLength*sizeof(double))!=cudaSuccess)
        {
            std::cout<<"Allocated memory did not accept given data!\n";
            return {};
        };
        if (cudaMalloc((void**)&cudaB, vecLength*sizeof(double))!=cudaSuccess)
        {
            std::cout<<"Allocated memory did not accept given data!\n";
            cudaFree(cudaA);
            return {};
        };
        if (cudaMalloc((void**)&cudaC, numb_layers*sizeof(int))!=cudaSuccess)
        {
            std::cout<<"Allocated memory did not accept given data!\n";
            cudaFree(cudaA);
            cudaFree(cudaB);
            return {};
        };

        // Supply allocated memory with data from host.
        if (cudaMemcpy(cudaA, vecA.data(), vecLength*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
        {
            std::cout<<"Allocated memory did not accept given data!\n";
            cudaFree(cudaA);
            cudaFree(cudaB);
            cudaFree(cudaC);
            return {};
        };
        if (cudaMemcpy(cudaB, vecB.data(), vecLength*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
        {
            std::cout<<"Allocated memory did not accept given data!\n";
            cudaFree(cudaA);
            cudaFree(cudaB);
            cudaFree(cudaC);
            return {};
        };
        if (cudaMemcpy(cudaC, architecture.data(), numb_layers*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess)
        {
            std::cout<<"Allocated memory did not accept given data!\n";
            cudaFree(cudaA);
            cudaFree(cudaB);
            cudaFree(cudaC);
            return {};
        };

        // Specify number of threads to run on GPU.
        int threadsPerBlock = 16;
        int blocksPerGrid1;
        int blocksPerGrid2;

        int length_of_vec = vecLength;
        int clrLayer;

        if (forward)
        {
            blocksPerGrid1 = architecture[_layer] / threadsPerBlock + 1;
            clrLayer = _layer + 1;
            if (interconnection)
            {
                blocksPerGrid2 = architecture[_layer+1] / threadsPerBlock + 1;
            }
            else
            {
                blocksPerGrid2 = 1;
            };
        }
        else
        {
            blocksPerGrid1 = architecture[layer] / threadsPerBlock + 1;
            clrLayer = layer - 1;
            if (interconnection)
            {
                blocksPerGrid2 = architecture[layer-1] / threadsPerBlock + 1;
            }
            else
            {
                blocksPerGrid2 = 1;
            };
        };

        dim3 blockDIM(threadsPerBlock, threadsPerBlock);
        dim3 gridDIM(blocksPerGrid1, blocksPerGrid2);

        // Run calculation on device kernel.
        clear_layer_kernel<<<gridDIM,blockDIM>>>(cudaA,cudaB,interconnection,clrLayer,length_of_vec,cudaC);

        // Catch possible gpu errors.
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Copy vectors from GPU to host.
        if (cudaMemcpy(vecB.data(), cudaB, vecLength*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
	        cout<<"Alloacted memory did not accept supplied data!\n";
	        cudaFree(cudaA);
	        cudaFree(cudaB);
            cudaFree(cudaC);
	        return {};
        }

        // Unallocate all pointers.
        cudaFree(cudaA);
        cudaFree(cudaB);
        cudaFree(cudaC);

        return vecB;
    }
}
