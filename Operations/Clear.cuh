#ifndef CLEAR
#define CLEAR 

#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

namespace Clear
{
    vector<double> clear_layer(vector<double> A, int _layer, vector<int> architecture, int numb_layers, bool interconnection, bool forward);
}

#endif