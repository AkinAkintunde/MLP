#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <stdio.h>
#include "TrainNet.h"
#include <pybind11/stl_bind.h>

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(MLP, m) 
{
    m.doc() = "A neural network that runs on GPU";
    m.attr("__version__") = "1.0.0";

    py::class_<TrainNet>(m, "TrainNet")
    .def(py::init<>())
    .def("set", &TrainNet::set_training_data);

}