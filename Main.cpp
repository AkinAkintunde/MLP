#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <stdio.h>
#include "TrainNet.cuh"
#include <pybind11/stl_bind.h>

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(MLP, m) 
{
    m.doc() = "A neural network that runs on GPU";
    m.attr("__version__") = "1.0.0";

    py::class_<TrainNet>(m, "TrainNet")
    .def(py::init<>())
    .def("set", &TrainNet::set_training_data)
    .def("run_gpu", &TrainNet::compute_gpu, "Train the neural network")
    .def("net_connections", &TrainNet::get_active_net_connections, "Return the state of connections between each node")
    .def("weights", &TrainNet::get_net_weights, "Return the current weight for a connection leaving a node.")
    .def("biases", &TrainNet::get_net_biases, "Return the current bias for a connection leaving a node.")
    .def("nodes", &TrainNet::get_net_nodes, "Return the current values for a given node")
    .def("cost", &TrainNet::get_net_cost, "Return the current cost of network.")
    .def("all_weights", &TrainNet::get_all_weights, "Access all weights in network.")
    .def("all_biases", &TrainNet::get_all_biases, "Access all biases in network.")
    .def("cost", &TrainNet::get_cost, "Returns the curent value of the cost function.")
    .def("manual_set", &TrainNet::setOfficialWeightsBiases, "Manually set the weights and biases.");

}