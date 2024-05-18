#ifndef NETDATA 
# define NETDATA

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector> 
#include <list>

using namespace std;

struct net_data 
{
    vector<double> nodes;
    vector<double> weights;
    vector<double> biases;
};

#endif