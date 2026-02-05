#include <chrono>
#include <climits>
#include <iomanip>
#include <iostream>

#include "cache.h"
#include "configuration.h"
#include "dataset.h"
#include "dataview.h"
#include "file_reader.h"
#include "general_solver.h"
#include "parameter_handler.h"
#include "statistics.h"
#include "tree.h"

void create_optimal_decision_tree(std::string file_name, int run_number, Configuration& config, double runtime_limit) {
    long long total_time = 0;
    int instance_number = -1;
    std::shared_ptr<Tree> optimal_decision_tree;

    Dataset unsorted_dataset{}; int class_number = -1; 
    file_reader::read_file(file_name, unsorted_dataset, class_number);
    config.complexity_cost *= unsorted_dataset.get_instance_number();

    for (int run = 0; run < run_number; run++) {
        
        Dataset sorted_dataset = unsorted_dataset;        

        auto start = std::chrono::high_resolution_clock::now();
        starting_time = start;
        config.stopwatch.Initialise(runtime_limit);

        sorted_dataset.sort_feature_values();

        Dataview dataview = Dataview(&sorted_dataset, &unsorted_dataset, class_number, config.sort_gini);

        optimal_decision_tree = std::make_shared<Tree>();
        int max_gap = config.max_gap;
        do {
            Cache::global_cache = Cache(config.max_depth, unsorted_dataset.get_instance_number());
            config.is_root = true;
            GeneralSolver::create_optimal_decision_tree(dataview, config, optimal_decision_tree, INT_MAX);

            max_gap = int(max_gap * config.max_gap_decay);
        } while (max_gap > 0);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        total_time += duration.count();
        instance_number = dataview.get_dataset_size();
        if (config.print_logs && !config.stopwatch.IsWithinTimeLimit()) {
            std::cout << std::endl << "The search was stopped because of a time-out. The tree is possibly not optimal." << std::endl << std::endl;
        }
    }

    double average_time = (double) total_time / run_number / 1000.0;
    int misclassification_score = int(std::round(optimal_decision_tree->objective - optimal_decision_tree->get_num_branching_nodes() * config.complexity_cost));
    std::cout << "Objective: " << optimal_decision_tree->objective << std::endl;
    std::cout << "Misclassification score: " << misclassification_score << std::endl;
    std::cout << "Accuracy: " << ((double) instance_number - misclassification_score) / (double) instance_number << std::endl;
    std::cout << "Number of branching nodes: " << optimal_decision_tree->get_num_branching_nodes() << std::endl;
    std::cout << "Average time taken to get the decision tree: " << std::fixed << std::setprecision(4) << average_time << " seconds" << std::endl;
    std::cout << "Optimal tree: " << std::fixed << std::setprecision(8) << *optimal_decision_tree << std::endl;

    if (config.print_logs) {
        statistics::print_statistics();
    }

}

int main(int argc, char *argv[]) {
    ParameterHandler parameters = ParameterHandler::DefineParameters();

    bool verbose = true;
    if (argc <= 1) {
        std::cout << "No parameters specified." << std::endl << std::endl;
        parameters.PrintHelpSummary();
        exit(1);
    }
    parameters.ParseCommandLineArguments(argc, argv);

    Configuration config;
    const std::string file = parameters.GetStringParameter("file");
    const int NUM_RUNS = int(parameters.GetIntegerParameter("run-number"));
    const double runtime_limit = parameters.GetFloatParameter("time");
    config.print_logs = parameters.GetBooleanParameter("print-logs");
    config.max_depth = int(parameters.GetIntegerParameter("max-depth"));
    config.use_upper_bound = parameters.GetBooleanParameter("use-upper-bound");
    config.max_gap = int(parameters.GetIntegerParameter("max-gap"));
    config.max_gap_decay = float(parameters.GetFloatParameter("max-gap-decay"));
    config.sort_gini = parameters.GetBooleanParameter("sort-features-gini-index");
    config.complexity_cost = float(parameters.GetFloatParameter("complexity-cost"));
    
    
    if (config.print_logs) {
        parameters.PrintParameterValues();
    }

    create_optimal_decision_tree(file, NUM_RUNS, config, runtime_limit);

    return 0;
}