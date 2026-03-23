#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <variant>
#include <algorithm>
#include "parameter_handler.h"
#include "general_solver.h"
#include "dataset.h"
#include "configuration.h"
#include "statistics.h"
#include "tree.h"

namespace py = pybind11;
using namespace std;

void NumpyToConTreeData(const py::array_t<double, py::array::c_style>& _X,
    const py::array_t<int, py::array::c_style>& _y, Dataset& data, int& class_number) {
    
    auto X = _X.template unchecked<2>();
    auto y = _y.template unchecked<1>();
    const int num_instances = int(X.shape(0));
    const int num_features = int(X.shape(1));

    int data_point_index = 0;

    class_number = 1;
    for (py::size_t i = 0; i < num_instances; i++) {
        int label = y.size() == 0 ? 0 : y(i);
        class_number = std::max(class_number, label + 1);
        
        int feature_index = 0;
        for (py::size_t j = 0; j < num_features; j++) {
            data.add_feature_index_pair(feature_index++, data_point_index, float(X(i, j)), label);
        }
        data_point_index++;
    }
    data.compute_unique_value_indices();
}

void RecursiveClassify(const Tree& tree, const py::detail::unchecked_reference<double, 2>& X, const std::vector<int>& ids, std::vector<int>& predictions) {
    if (ids.size() == 0) return;
    if (tree.is_leaf()) {
        for (auto i : ids) {
            predictions[i] = tree.get_label();
        }
        return;
    }

    std::vector<int> left_ids, right_ids;
    left_ids.reserve(ids.size());
    right_ids.reserve(ids.size());
    for (int i : ids) {
        if (X(i, tree.get_split_feature()) <= tree.get_split_threshold()) {
            left_ids.push_back(i);
        } else {
            right_ids.push_back(i);
        }
    }
    RecursiveClassify(*(tree.get_left_tree()), X, left_ids, predictions);
    RecursiveClassify(*(tree.get_right_tree()), X, right_ids, predictions);
}

PYBIND11_MODULE(ccontree, m) {
    m.doc() = "This is documentation";

    /*************************************
       Configuration
    ************************************/
    py::class_<Configuration> config(m, "Config");

    config.def(py::init([]() {
        return new Configuration();
    }), py::keep_alive<0, 1>());
    config.def_readwrite("verbose", &Configuration::print_logs, "The verbosity of the solver.");
    config.def_readwrite("max_depth", &Configuration::max_depth, "The maximum depth of the tree.");
    config.def_readwrite("complexity_cost", &Configuration::complexity_cost, "The cost of adding a branching node.");
    config.def_readwrite("max_gap", &Configuration::max_gap, "The maximum permissable gap from optimal (number of misclassifications).");
    config.def_readwrite("max_gap_decay", &Configuration::max_gap_decay, "The decay in the permissable gap.");
    config.def_readwrite("use_upper_bound", &Configuration::use_upper_bound, "Enable/disable the use of upper bounds.");
    config.def_readwrite("sort_gini", &Configuration::sort_gini, "Sort the features by gini impurity.");
    config.def("is_within_time_limit", [](const Configuration& config) -> bool { return config.stopwatch.IsWithinTimeLimit();});

    /*************************************
           Decision Tree
     ************************************/
    py::class_<Tree, shared_ptr<Tree>> tree(m, "Tree");

    tree.def("is_leaf_node", &Tree::is_leaf, "Return true if this node is a leaf node.");
    tree.def("is_branching_node", &Tree::is_internal, "Return true if this node is a branching node.");
    tree.def("get_depth", &Tree::get_depth, "Return the depth of this tree.");
    tree.def("get_num_branching_nodes", &Tree::get_num_branching_nodes, "Get the number of branching nodes in this tree.");
    tree.def("get_num_leaf_nodes", &Tree::get_num_leaf_nodes, "Get the number of leaf nodes in this tree.");
    tree.def("__str__", [](const Tree& tree) { return tree.to_string(0);});
    tree.def("get_split_feature", &Tree::get_split_feature, "Get the split feature of this branching node.");
    tree.def("get_split_threshold", &Tree::get_split_threshold, "Get the split threshold of this branching node.");
    tree.def("get_label", &Tree::get_label, "Get the label of this leaf node.");
    tree.def("get_left", &Tree::get_left_tree, "Get the left child node.");
    tree.def("get_right", &Tree::get_right_tree, "Get the right child node.");
    tree.def("predict", [](const Tree& tree, const py::array_t<double, py::array::c_style>& _X) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));

        py::detail::unchecked_reference<double, 2> X = _X.template unchecked<2>();

        const int num_instances = int(X.shape(0));
        const int num_features = int(X.shape(1));

        std::vector<int> instances(num_instances);
        std::iota(instances.begin(), instances.end(), 0);
        vector<int> predictions(num_instances, 0);
        RecursiveClassify(tree, X, instances, predictions);
        
        return py::array_t<int, py::array::c_style>(predictions.size(), predictions.data());
    });

    /*************************************
           Solve
     ************************************/
    m.def("solve", [](const py::array_t<double, py::array::c_style>& _X,
        const py::array_t<int, py::array::c_style>& _y, Configuration& _config, double runtime_limit) {
        py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
        _config.stopwatch.Initialise(runtime_limit);
        
        Dataset unsorted_dataset{}; int class_number = -1;
        NumpyToConTreeData(_X, _y, unsorted_dataset, class_number);
        Dataset sorted_dataset = unsorted_dataset;
        
        auto config = _config;
        config.complexity_cost *= sorted_dataset.get_instance_number();
        config.stats = std::make_shared<statistics>();
        
        sorted_dataset.sort_feature_values();
        Dataview dataview(&sorted_dataset, &unsorted_dataset, class_number, config.sort_gini);

        auto optimal_decision_tree = std::make_shared<Tree>();

        int max_gap = config.max_gap;
        do {
            Cache::global_cache = Cache(config.max_depth, unsorted_dataset.get_instance_number());
            config.is_root = true;
            GeneralSolver::create_optimal_decision_tree(dataview, config, optimal_decision_tree, INT_MAX);

            max_gap = int(max_gap * config.max_gap_decay);
        } while (max_gap > 0);

        if (config.print_logs) {
            config.stats->print_statistics();
            if (!config.stopwatch.IsWithinTimeLimit()) {
                std::cout << std::endl << "The search was stopped because of a time-out. The tree is possibly not optimal." << std::endl << std::endl;
            }
        }

        return optimal_decision_tree;

    });
}