#ifndef PROPAGATOR1
#define PROPAGATOR1

__global__ void updateNetwork(double *aa, double *bb, double *cc, double *dd, double* ee, double *ff, int *gg, int *hh, double *ii, int numLayers, int layer_IDX, int activation_type, bool forward);

#endif