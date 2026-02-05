#ifndef SPECIALIZED_SOLVER_H
#define SPECIALIZED_SOLVER_H

#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>

#include "configuration.h"
#include "dataset.h"
#include "dataview.h"
#include "intervals_pruner.h"
#include "specialized_solver.h"
#include "statistics.h"
#include "tree.h"

class Depth1ScoreHelper;

class SpecializedSolver {
public:
    /**
     * Calculates the misclassification scores for both the left and right splits of the dataset using only one dataset traversal.
     * 
     * @param dataset The dataset to calculate the scores from.
     * @param feature_index The index of the feature to split on.
     * @param split_point The split point for the feature.
     * @param threshold The threshold value for the split.
     * @param left_optimal_tree The optimal tree for the left split.
     * @param right_optimal_tree The optimal tree for the right split.
     * @param upper_bound The upper bound for the scores.
     * @param complexity_cost The cost of adding a node
     */
    static void get_best_left_right_scores(const Dataview& dataset, int feature_index, int split_point, float threshold, 
        std::shared_ptr<Tree>& left_optimal_tree, std::shared_ptr<Tree>& right_optimal_tree, float upper_bound, float complexity_cost);

private:
    /**
    * Calculates the best depth one split for both the left and right splits for a feature to split on
    *
    * @param dataset The dataset to calculate the scores from.
    * @param feature_index The index of the feature (f1) to split on.
    * @param split_point The split point for the feature (f1).
    * @param current_feature_index The index of the second feature (f2) to split on
    * @param split_index The index of the split point (f1)
    * @param left_tree the depth one score helper for the left subtree
    * @param right_tree the depth one score helper for the right subtree
    * @param split_feature_split_indices A mapping from datapoint_indices to unique value indices
    * @param upper_bound The upper bound for the scores.
    * @param complexity_cost The cost of adding a node
    */
    template <bool is_same_feature>
    static void process_depth_one_feature(const Dataview& dataview,
        const int feature_index, const int split_point, const int current_feature_index, const int split_index,
        Depth1ScoreHelper& left_tree, Depth1ScoreHelper& right_tree,
        const std::vector<int>& split_feature_split_indices, float& upper_bound, const float complexity_cost);
};

#endif // SPECIALIZED_SOLVER_H
