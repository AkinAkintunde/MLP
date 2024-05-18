#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include <csignal> 
#include "VecAdd.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void addVecKernel(double *aA, double *bB, double *cC, int length)
{
    int IDX = blockIdx.x*blockDim.x + threadIdx.x;
    if (IDX < length)
    {
        cC[IDX] = aA[IDX] + bB[IDX];
    };
}

namespace AddVec
{
    vector<double> addVecGPU(vector<double> A, vector<double> B)
    {
        double *cudaA;
        double *cudaB;
        double *cudaC;

        vector<double> vecA = A;
        vector<double> vecB = B;

        int vecLength = vecA.size();

        vector<double> vecC(vecLength, 0.);

        if (vecLength != vecB.size())
        {
            std::cout<<"Vectors are of unequal length!\n";
            return {};
        }

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
        if (cudaMalloc((void**)&cudaC, vecLength*sizeof(double))!=cudaSuccess)
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
        if (cudaMemcpy(cudaC, vecC.data(), vecLength*sizeof(double), cudaMemcpyHostToDevice)!=cudaSuccess)
        {
            std::cout<<"Allocated memory did not accept given data!\n";
            cudaFree(cudaA);
            cudaFree(cudaB);
            cudaFree(cudaC);
            return {};
        };

        // Specify number of threads to run on GPU.
        int threadsPerBlock = 256;
        int blocksPerGrid1;

        int length_of_vec = vecLength;

        blocksPerGrid1 = length_of_vec / threadsPerBlock + 1;

        dim3 blockDIM(threadsPerBlock, 1);
        dim3 gridDIM(blocksPerGrid1, 1);

        // Run calculation on device kernel.
        addVecKernel<<<gridDIM,blockDIM>>>(cudaA,cudaB,cudaC,length_of_vec);

        // Catch possible gpu errors.
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Copy vectors from GPU to host.
        if (cudaMemcpy(vecC.data(), cudaC, vecLength*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
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

        return vecC;
    }
}

