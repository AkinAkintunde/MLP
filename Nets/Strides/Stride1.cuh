#ifndef STRIDE1
#define STRIDE1

__device__ int internodal_data_list_indexer(int node_ID, int layer_ID, int connection_ID, int *architecture);
__device__ int nodal_data_list_indexer(int node_ID, int layer_ID, int *architecture);

#endif 