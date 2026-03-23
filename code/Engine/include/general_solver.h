#ifndef GENERAL_SOLVER_H
#define GENERAL_SOLVER_H

#include <iostream>
#include <memory>
#include <queue>
#include <vector>

#include "cache.h"
#include "dataset.h"
#include "dataview.h"
#include "intervals_pruner.h"
#include "specialized_solver.h"
#include "statistics.h"
#include "tree.h"

/** 
 * A convenient container for all the split and interval info
 * 
 * Note the difference between mid and split_point
 * For example in the data 0 0 1 2 2 3, there are 3 possible split points, namely 0 0 | 1 | 2 2 | 3
 * Given interval [left, right] = [0, 2], we have mid = 1, the split point between value 1 and 2
 * Visually: 0 0 1 | 2 2 3, this corresponds to threshold = 1.5
 * and split point = 3 (the first sample index after the split)
 */
struct SplitInfo {
    int feature = -1;       // The feature that is used to split
    int left = -1;          // The split-index (inclusive) of the left of the interval
    int right = -1;         // the split-index (inclusive) of the right of the interval
    int mid = -1;           // the mid point in the interval, the split point we are splitting on now
    int split_point = -1;   // the sample index of the first sample right of the split
    float threshold = -1;   // the split threshold (in the continuous numbers, not the unique_value_index)
    int unique_value_threshold = -1; // the split threshold (in the unique value index, not the continuous value)
    std::shared_ptr<Tree> left_optimal_dt;
    std::shared_ptr<Tree> right_optimal_dt;
};

class GeneralSolver {
public:
    /**
     * Creates the optimal decision tree for the given dataset and solution configuration.
     * 
     * It uses the provided upper bound to prune the search space and reduce the number of possible solutions.
     * 
     * @param dataview The dataset to create the decision tree from.
     * @param config The configuration for the solution.
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     */
    static void create_optimal_decision_tree(const Dataview& dataview, Configuration& config, std::shared_ptr<Tree>& current_optimal_tree, float upper_bound);

private:
    /**
     * Creates the optimal decision tree for the given dataset, solution configuration and the first feature to split on.
     * 
     * It uses the provided upper bound to prune the search space and reduce the number of possible solutions.
     * 
     * @param dataview The dataset to create the decision tree from.
     * @param config The configuration for the solution.
     * @param feature_index The index of the feature to split on.
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     */
    static void create_optimal_decision_tree(const Dataview& dataview, Configuration& config, int feature_index, std::shared_ptr<Tree>& current_optimal_tree, float upper_bound);

    /**
     * Solves a split given a fixed feature and split point, either by a recursive call or by using the specialized depth-two solver
     * 
     * @param dataview The dataset to create the decision tree from.
     * @param config The configuration for the solution.
     * @param split The information on what feature and what threshold to split
     * @param current_optimal_tree The current optimal tree.
     * @param upper_bound The upper bound for the search space.
     */
    static void solve_split(const Dataview& dataview, Configuration& config, SplitInfo& split, std::shared_ptr<Tree>& current_optimal_tree, float upper_bound);
    
    /**
     * Calculates the misclassification score if the current node is a leaf node.
     * 
     * @param dataset The dataset to calculate the scores from.
     * @param feature_index The index of the feature to split on.
     * @param split_point The split point for the feature.
     * @param threshold The threshold value for the split.
     * @param left_optimal_tree The optimal tree for the left split.
     * @param right_optimal_tree The optimal tree for the right split.
     * @param upper_bound The upper bound for the scores.
     */
    static void calculate_leaf_node(int class_number, int instance_number, const std::vector<int>& label_frequency, std::shared_ptr<Tree>& current_optimal_decision_tree);

    static void reduce_node_budget(const Dataview& dataview, Configuration& config, float upper_bound);
};

#endif // GENERAL_SOLVER_H
