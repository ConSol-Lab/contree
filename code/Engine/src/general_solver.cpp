#include "general_solver.h"

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, Configuration& config, std::shared_ptr<Tree>& current_optimal_decision_tree, float upper_bound) {
    RUNTIME_ASSERT(dataview.get_dataset_size() > 0, "Dataset cannot be empty.");
    RUNTIME_ASSERT(upper_bound >= 0, "Upper bound should always be zero or higher.");
    RUNTIME_ASSERT(current_optimal_decision_tree->objective >= 0, "Current objective should always be zero or higher.");

    // Check if we have the subproblem cached already
    if (Cache::global_cache.is_cached(dataview, config.max_depth)) {
        current_optimal_decision_tree = Cache::global_cache.retrieve(dataview, config.max_depth);
        statistics::total_number_cache_hits += 1;
        return;
    }

    // First calculate the leaf node solution
    calculate_leaf_node(dataview.get_class_number(), dataview.get_dataset_size(), dataview.get_label_frequency(), current_optimal_decision_tree);

    // If no remaining depth budget, return the leaf node solution
    if (config.max_depth == 0) return;

    // If the number of samples is not at least two times the complexity cost (ceiled), then no improving split can be found
    // This is true, because if the current leaf node has loss L, and if we split of n samples, by the sim.lb we have the LB L - n + cost-complexity
    // So the number of samples we split off (n) needs to be at least ceil(cost-complexity). So for a valid split, we need twice this amount (2 * ceil(cost-complexity))
    if (dataview.get_dataset_size() < 2 * static_cast<int>(std::ceil(config.complexity_cost))) return;

    // If the upperbound or misclassification score is lower than the complexity cost, then any split will exceed the upper bound, so return
    if (std::min(current_optimal_decision_tree->objective, upper_bound) <= config.complexity_cost + config.max_gap + EPSILON || dataview.get_dataset_size() == 1) {
        return;
    }

    // Go over all features (optionally in order of their gini index) and find an optimal split for that feature.
    for (int feature_nr = 0; feature_nr < dataview.get_feature_number(); feature_nr++) {
        // Get the feature index basesd on the gini ordering
        int feature_index = dataview.gini_values[feature_nr].second;
        // If the feature has no possible splits, skip it
        if (dataview.get_possible_split_indices(feature_index).size() == 0) continue;
        // Recursively solve the subproblem with this feature index as the branching feature
        create_optimal_decision_tree(dataview, config, feature_index, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->objective));
        // Stop if the incumbent solution cannot be improved
        if (current_optimal_decision_tree->objective <= config.complexity_cost + config.max_gap + EPSILON) return;
        // Stop if out of time
        if (!config.stopwatch.IsWithinTimeLimit()) return;
    }

    // only if the solution is better than the upper bound do we cache it
    if (current_optimal_decision_tree->objective <= upper_bound) {
        Cache::global_cache.store(dataview, config.max_depth, current_optimal_decision_tree);
    }
}

SplitInfo initialize_split(const std::vector<int>& possible_split_indices, const std::vector<Dataset::FeatureElement>& current_feature, int feature, int left, int right) {
    SplitInfo split;
    split.feature = feature;
    split.left = left;
    split.right = right;
    split.mid = (left + right) / 2;
    split.split_point = possible_split_indices[(left + right) / 2];
    split.threshold = split.mid > 0 ? (current_feature[possible_split_indices[split.mid - 1]].value + current_feature[split.split_point].value) / 2.0f
        : (current_feature[split.split_point].value + current_feature[0].value) / 2.0f;
    split.left_optimal_dt = std::make_shared<Tree>();
    split.right_optimal_dt = std::make_shared<Tree>();
    return split;
}

void GeneralSolver::create_optimal_decision_tree(const Dataview& dataview, Configuration& config, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, float upper_bound) {    
    // Get the data sorted by the split feature, and get all possible split indices
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);
    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    
    // Check if we can lower the depth and node budget
    reduce_node_budget(dataview, config, upper_bound);
    
    // Initialize the interval pruner. We use this datastructure to keep track of all solutions at previously investigated splits
    IntervalsPruner interval_pruner(possible_split_indices, (config.max_gap + 1) / 2, config.complexity_cost);

    // Initialize the queue of unsearched intervals with a single interval, the  interval that includes all  possible split indices
    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    int initial_left  = int(std::lower_bound(possible_split_indices.begin(), possible_split_indices.end(), static_cast<int>(std::ceil(config.complexity_cost))) - possible_split_indices.begin());
    int initial_right = int(std::upper_bound(possible_split_indices.begin(), possible_split_indices.end(), static_cast<int>(dataview.get_dataset_size() - std::ceil(config.complexity_cost))) - possible_split_indices.begin());
    unsearched_intervals.push({ initial_left, initial_right, -1, -1});

    // While there are unsearched split inervals left...
    while(!unsearched_intervals.empty()) {
        if (!config.stopwatch.IsWithinTimeLimit()) return;
        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        // Use subinterval pruning to check if we can skip this interval
        // Subinterval pruning checks if there is a split left and right of the interval such that 
        // the left subtree of the left split and the right subtree of the right split together are more than 
        // the upper bound. If so, adding samples in the middle cannot improve the solution. Therefore, skip this interval.
        if (interval_pruner.subinterval_pruning(current_interval, upper_bound)) continue;

        // Now update the interval using interval shrinking, which checks if an updated upper bound possibly
        // shrinks the interval relative to when it was first created.
        interval_pruner.interval_shrinking(current_interval, upper_bound);
        const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
        
        // If the interval is empty, skip it.
        if (left > right) continue;
         
        // Initialize the split by taking the mid point of the interval
        SplitInfo split = initialize_split(possible_split_indices, current_feature, feature_index, left, right);

        // Recursively solve this split either doing two recursive calls for left and right or 
        // by calling our special depth-two case
        solve_split(dataview, config, split, current_optimal_decision_tree, upper_bound);

        // compute the solution cost of the split. Check if both trees are properly initialized, and if the solution is better, update the incumbent
        const float current_best_score = split.left_optimal_dt->objective + split.right_optimal_dt->objective + config.complexity_cost;
        if (split.left_optimal_dt->is_initialized() && split.right_optimal_dt->is_initialized() && current_best_score < current_optimal_decision_tree->objective) {
            current_optimal_decision_tree->update_split(feature_index, split.threshold, split.left_optimal_dt, split.right_optimal_dt, config.complexity_cost);
            upper_bound = std::min(upper_bound, current_best_score);
            if (current_best_score <= config.complexity_cost + EPSILON) return;
            if (PRINT_INTERMEDIARY_TIME_SOLUTIONS && config.is_root)  {
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
            }
        }
        // Add the result to the interval pruner
        interval_pruner.add_result(split.mid, split.left_optimal_dt->objective, split.right_optimal_dt->objective);

        // If the interval is just a single split point, then the interval cannot be further split up
        if (left == right) continue;
        
        // Split the interval into two halves at the split point and
        // use the score difference between this solution and the upper bound to prune both halves
        // the neighborhood pruning returns the right (left) bound of the left (right) interval
        // and adds them to the queue if they are not empty
        const float score_difference = current_best_score - upper_bound;
        RUNTIME_ASSERT(score_difference >= 0, "score difference cannot be negative.");
        const auto [new_bound_left, new_bound_right] = interval_pruner.neighbourhood_pruning(score_difference, left, right, split.mid);

        if (new_bound_left <= right) {
            unsearched_intervals.push({new_bound_left, right, split.mid, current_right_bound});
        }

        if (left <= new_bound_right) {
            unsearched_intervals.push({left, new_bound_right, current_left_bound, split.mid});
        }
    }
}

void GeneralSolver::solve_split(const Dataview& dataview, const Configuration& config, SplitInfo& split, std::shared_ptr<Tree>& current_optimal_tree, float upper_bound) {
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(split.feature);
    const auto& possible_split_indices = dataview.get_possible_split_indices(split.feature);
    
    if (config.max_depth == 2) {
        // If the maximum remaining depth is two, we use our special depth two solver
        statistics::total_number_of_specialized_solver_calls += 1;
        SpecializedSolver::get_best_left_right_scores(dataview, split.feature, split.split_point, split.threshold, split.left_optimal_dt, split.right_optimal_dt, upper_bound, config.complexity_cost);
        RUNTIME_ASSERT(split.left_optimal_dt->objective >= 0, "D2 - Left tree should have non-negative misclassification score.");
        RUNTIME_ASSERT(split.right_optimal_dt->objective >= 0, "D2 - Right tree should have non-negative misclassification score.");
    } else {
        // We compute the distance from the split point to the sample indicies of the left and right interval split points
        const int interval_half_distance = std::max(split.split_point - possible_split_indices[split.left], possible_split_indices[split.right] - split.split_point);
        // The unique-value index of the split point. We use this to split the data (rather than the continuous threshold, which may give numerical instability)
        const int split_unique_value_index = current_feature[split.split_point].unique_value_index;
        
        // Split the dataset in two based on the split point
        Dataview left_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        Dataview right_dataview = Dataview(dataview.get_class_number(), dataview.should_sort_by_gini_index());
        Dataview::split_data_points(dataview, split.feature, split.split_point, split_unique_value_index, left_dataview, right_dataview, config.max_depth);

        // We first compute the subtree for the bigger dataset since it might make computing the smaller dataset obsolete
        auto& smaller_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? left_dataview : right_dataview;
        auto& larger_data = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? right_dataview : left_dataview;

        auto& smaller_optimal_dt = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? split.left_optimal_dt : split.right_optimal_dt;
        auto& larger_optimal_dt = (left_dataview.get_dataset_size() < right_dataview.get_dataset_size()) ? split.right_optimal_dt : split.left_optimal_dt;

        // The upper bound of the subtree is the current upper bound minus the cost of branching
        float larger_ub = config.use_upper_bound
            ? std::min(upper_bound, current_optimal_tree->objective) - config.complexity_cost
            : current_optimal_tree->objective;

        // Recursively solve the subtree
        statistics::total_number_of_general_solver_calls += 1;
        Configuration left_solution_configuration = config.GetLeftSubtreeConfig();
        GeneralSolver::create_optimal_decision_tree(larger_data, left_solution_configuration, larger_optimal_dt, larger_ub);
        RUNTIME_ASSERT(larger_optimal_dt->objective >= 0, "Right tree should have non-negative misclassification score.");
        RUNTIME_ASSERT(larger_optimal_dt->is_initialized(), "Right tree should be initialized.");

        // The upper bound for the second subtree is the current upper bound minus the cost of branching and minus the cost of the first subtree.
        // However, we add the distance from the mid point to the end of the interval, to increase the probability that we can prune the whole interval.
        float smaller_obj_ub = std::min(current_optimal_tree->objective, upper_bound) - larger_optimal_dt->objective - config.complexity_cost;
        float smaller_ub = config.use_upper_bound ? smaller_obj_ub +  float(interval_half_distance) : current_optimal_tree->objective;

        // We compute the second subtree only if we have a positive upper bound, larger than zero.
        // If the upper bound is precisely zero, but not because of the interval_half distance, we also want to compute the subproblem
        if (smaller_ub > 0 || (std::abs(smaller_ub) <= EPSILON && std::abs(smaller_obj_ub) <= EPSILON)) {
            statistics::total_number_of_general_solver_calls += 1;
            Configuration right_solution_configuration = config.GetRightSubtreeConfig(left_solution_configuration.max_gap);
            GeneralSolver::create_optimal_decision_tree(smaller_data, right_solution_configuration, smaller_optimal_dt, smaller_ub);
            RUNTIME_ASSERT(smaller_optimal_dt->objective >= 0, "Left tree should have non-negative misclassification score.");
            RUNTIME_ASSERT(smaller_optimal_dt->is_initialized(), "Left tree should be initialized.");
        } else {
            // Because the upper bound is negative, we are not going to search the second subproblem
            // So we set its objective to the trivial lower bound: zero
            smaller_optimal_dt->objective = 0;
        }
    }
}

void GeneralSolver::calculate_leaf_node(int class_number, int instance_number, const std::vector<int>& label_frequency, std::shared_ptr<Tree>& current_optimal_decision_tree) {
    int best_classification_score = -1;
    int best_classification_label = -1;

    for (int label = 0; label < class_number; label++) {
        if (label_frequency[label] > best_classification_score) {
            best_classification_score = label_frequency[label];
            best_classification_label = label;
        }
    }

    const int best_misclassification_score = instance_number - best_classification_score;

    if (best_misclassification_score < current_optimal_decision_tree->objective) {
        RUNTIME_ASSERT(best_classification_label != -1, "Cannot assign negative leaf label.");
        current_optimal_decision_tree->make_leaf(best_classification_label, best_misclassification_score);
    }
}

void GeneralSolver::reduce_node_budget(const Dataview& dataview, Configuration& config, float upper_bound) {
    // Compute the number of branching nodes based on the maximum depth (2^d - 1)
    int num_nodes = (1 << config.max_depth) - 1;

    // The number of samples is a maximum on the number of leaf nodes, so - 1 for the number of branching nodes
    int nodes = std::min(num_nodes, std::max(dataview.get_dataset_size() - 1, 0));

    // If there is a complexity cost set, then the number of nodes is bounded by the upper-bound divided by the complexity cost
    if (config.complexity_cost > 0) {
        int nodes = int(std::max(0.0f, std::min(float(num_nodes), (upper_bound + 1e-6f) / config.complexity_cost)));
    }

    // If the number of nodes is lower than the depth budget indicates, we can check if we can lower the depth limit
    if (nodes < num_nodes) {
        // If there can be at most n nodes, then the depth is also limited by n
        int new_max_depth = std::min(config.max_depth, nodes);
        if (new_max_depth < config.max_depth) {
            config.max_depth = new_max_depth;
        }
    }
}