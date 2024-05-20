#include "Activation1.h"
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include <cmath> 
#include <csignal> 

using namespace std;

namespace HostActivation1
{
    double activation_function(double w_val, int type, bool derivative)
    {
        if (type == 0)
        {
            if (derivative)
            {
                double M = 1./(1. + exp(-w_val));
                return M*M*exp(-w_val);
            }
            else
            {
                return 1./(1. + exp(-w_val));
            };
        }
        else if (type == 1)
        {
            if (derivative)
            {
                if (w_val <= 0)
                {
                    return 0.0;
                } 
                else
                {
                    return 1.0;
                }; 
            }
            else
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
        };
        return 0.;
    }
}