#include <iostream>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

#include "Unpacker.h"

using namespace std;

namespace Unpacker
{
    int samp_checker(vector<vector<double>> vecAs, int vecSizes)
    {
        int iters = 0;
        int theTotals = 0;
    start:
        if (iters == vecSizes)
        {
            return theTotals;
        }
        else
        {
            vector<double> _vecAs = vecAs[iters];
            int totals = _vecAs.size();
            theTotals += totals;
            iters++;
            goto start;
        };
    };

    int vec_3d_size(vector<vector<vector<double>>> vecA, int vecSize)
    {
        int iter = 0;
        int theTotal = 0;
    start:
        if (iter == vecSize)
        {
            return theTotal;
        }
        else
        {
            vector<vector<double>> _vecA = vecA[iter];
            int total = samp_checker(_vecA, _vecA.size());
            theTotal+= total;
            iter++;
            goto start;
        };
    };
}