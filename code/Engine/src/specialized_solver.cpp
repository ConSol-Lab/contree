#include "specialized_solver.h"

class Depth1ScoreHelper {
public:
    Depth1ScoreHelper(const int size, const int CLASS_NUMBER)
        : label_frequency(std::vector<int>(CLASS_NUMBER, 0)), current_label_frequency(std::vector<int>(CLASS_NUMBER, 0)), size(size) {

    }

    void reset_label_frequency() {
        std::fill(current_label_frequency.begin(), current_label_frequency.end(), 0);
        previous_value = 0.0f;
        previous_unique_value_index = -1;
        is_zero = false;
        can_skip = 0;
        current_element_count = 0;
    }

    int classification_score = -1;
    int best_feature_index = -1;
    float best_threshold = -1.0f;
    int best_left_label = -1;
    int best_right_label = -1;

    float previous_value = 0.0f;
    int previous_unique_value_index = -1;
    bool is_zero = false;

    int can_skip = 0;
    int current_element_count = 0;

    const int size;
    int max_label_frequency{ 0 };
    int max_label{ 0 };

    std::vector<int> label_frequency;
    std::vector<int> current_label_frequency;
};

void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, std::shared_ptr<Tree>& current_optimal_decision_tree, int upper_bound) {
    for (int feature_index = 0; feature_index < dataview.get_feature_number(); feature_index++) {
        create_optimal_decision_tree(dataview, solution_configuration, feature_index, current_optimal_decision_tree, std::min(upper_bound, current_optimal_decision_tree->misclassification_score));

        if (current_optimal_decision_tree->misclassification_score <= solution_configuration.max_gap) {
            return;
        }
    }
}

void SpecializedSolver::get_best_left_right_scores(const Dataview& dataview, int feature_index, int split_point, float threshold, std::shared_ptr<Tree> &left_optimal_dt, std::shared_ptr<Tree> &right_optimal_dt, int upper_bound) {
    const auto& split_feature = dataview.get_sorted_dataset_feature(feature_index);
    const auto& unsorted_split_feature = dataview.get_unsorted_dataset_feature(feature_index);
    std::vector<int> split_feature_split_indices(unsorted_split_feature.size());
    int split_index = -1;
    for (const auto& split_feature_data : split_feature) {
        split_feature_split_indices[split_feature_data.data_point_index] = split_feature_data.unique_value_index;
        if (split_index == -1 && split_feature_data.value >= threshold) {
            split_index = split_feature_data.unique_value_index;
        }
    }
    RUNTIME_ASSERT(split_index != -1, "Split index not found.");

    const int dataset_size = dataview.get_dataset_size();
    const int class_number = dataview.get_class_number();

    RUNTIME_ASSERT(split_point > 0 && split_point < dataset_size, "left and right subtree need to be non-empty.");
    Depth1ScoreHelper left_tree(split_point, class_number);
    Depth1ScoreHelper right_tree(dataset_size - split_point, class_number);
    

    left_tree.classification_score = std::max(0, left_tree.size - upper_bound);
    right_tree.classification_score = std::max(0, right_tree.size - upper_bound);

    Dataview::initialize_split_parameters(split_feature, class_number, dataview.get_label_frequency(), split_point, left_tree.label_frequency, right_tree.label_frequency);

    left_tree.max_label_frequency = 0;
    right_tree.max_label_frequency = 0;
    for (int label = 0; label < class_number; label++) {
        if (left_tree.label_frequency[label] > left_tree.max_label_frequency) {
            left_tree.max_label_frequency = left_tree.label_frequency[label];
            left_tree.max_label = label;
        }
        if (right_tree.label_frequency[label] > right_tree.max_label_frequency) {
            right_tree.max_label_frequency = right_tree.label_frequency[label];
            right_tree.max_label = label;
        }
    }
    left_tree.classification_score = std::max(left_tree.classification_score, left_tree.max_label_frequency);
    right_tree.classification_score = std::max(right_tree.classification_score, right_tree.max_label_frequency);

    for (int current_feature_index = 0 ; current_feature_index < dataview.get_feature_number(); current_feature_index++) {

        if (current_feature_index == feature_index) {
            process_depth_one_feature<true>(dataview, feature_index, split_point, current_feature_index, split_index,
                left_tree, right_tree, split_feature_split_indices, upper_bound);
        } else {
            process_depth_one_feature<false>(dataview, feature_index, split_point, current_feature_index, split_index,
                left_tree, right_tree, split_feature_split_indices, upper_bound);
        }


        if (left_tree.classification_score + right_tree.classification_score == dataset_size) {
            break;
        }
    }

    if (left_tree.classification_score == left_tree.max_label_frequency) {
        left_optimal_dt->make_leaf(left_tree.max_label, left_tree.size - left_tree.classification_score);
    } else {
        left_optimal_dt->update_split(left_tree.best_feature_index, left_tree.best_threshold, std::make_shared<Tree>(left_tree.best_left_label, -1), std::make_shared<Tree>(left_tree.best_right_label, -1));
        RUNTIME_ASSERT(left_tree.best_left_label != -1, "Left tree left label should be initialized.");
        RUNTIME_ASSERT(left_tree.best_right_label != -1, "Left tree right label should be initialized.");
    }
    left_optimal_dt->misclassification_score = left_tree.size - left_tree.classification_score;
    RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "LR - Left tree misclassification score should be non-negative.");

    if (right_tree.classification_score == right_tree.max_label_frequency) {
        right_optimal_dt->make_leaf(right_tree.max_label, right_tree.size - right_tree.classification_score);
    } else {
        right_optimal_dt->update_split(right_tree.best_feature_index, right_tree.best_threshold, std::make_shared<Tree>(right_tree.best_left_label, -1), std::make_shared<Tree>(right_tree.best_right_label, -1));
        RUNTIME_ASSERT(right_tree.best_left_label != -1, "Right tree left label should be initialized.");
        RUNTIME_ASSERT(right_tree.best_right_label != -1, "Right tree right label should be initialized.");
    }
    right_optimal_dt->misclassification_score = right_tree.size - right_tree.classification_score;
    RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "LR - Right tree misclassification score should be non-negative.");
}

template <bool is_same_feature>
void SpecializedSolver::process_depth_one_feature(const Dataview& dataview,
    const int feature_index, const int split_point, const int current_feature_index, const int split_index,
    Depth1ScoreHelper& left_tree, Depth1ScoreHelper& right_tree,
    const std::vector<int>& split_feature_split_indices, int& upper_bound) {
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(current_feature_index);
    const int class_number = dataview.get_class_number();
    const int dataset_size = dataview.get_dataset_size();
    int global_skip = 0;

    left_tree.reset_label_frequency();
    right_tree.reset_label_frequency();

    Depth1ScoreHelper* tree_p = &left_tree;
    int index = 0;
    for (const auto& current_feature_data : current_feature) {
        if constexpr (!is_same_feature) {
            int cur_split_index = split_feature_split_indices[current_feature_data.data_point_index];
            bool is_left_tree = (cur_split_index < split_index);
            tree_p = is_left_tree ? &left_tree : &right_tree;
        } else {
            if (index++ == split_point) {
                tree_p = &right_tree;
            };
        }
        auto& tree = *tree_p;

        if (tree.is_zero) {
            continue;
        }

        global_skip--;
        tree.can_skip--;

        if (current_feature_data.unique_value_index == tree.previous_unique_value_index || tree.can_skip > 0 || global_skip > 0) {
            tree.current_element_count++;
            tree.current_label_frequency[current_feature_data.label]++;
            tree.previous_value = current_feature_data.value;
            tree.previous_unique_value_index = current_feature_data.unique_value_index;
            continue;
        }

        int left_classification_score = -1;
        int right_classification_score = -1;

        int left_label = -1;
        int right_label = -1;

        for (int label_value = 0; label_value < class_number; label_value++) {
            if (tree.current_label_frequency[label_value] > left_classification_score) {
                left_classification_score = tree.current_label_frequency[label_value];
                left_label = label_value;
            }
            if (tree.label_frequency[label_value] - tree.current_label_frequency[label_value] > right_classification_score) {
                right_classification_score = tree.label_frequency[label_value] - tree.current_label_frequency[label_value];
                right_label = label_value;
            }
        }

        if (left_classification_score + right_classification_score > tree.classification_score) {
            RUNTIME_ASSERT(tree.classification_score <= tree.size, "LR - Classification score cannot exceed the number of instances.");
            tree.classification_score = left_classification_score + right_classification_score;
            tree.best_feature_index = current_feature_index;
            tree.best_threshold = (current_feature_data.value + tree.previous_value) / 2.0f;

            upper_bound = std::min(dataset_size - (left_tree.classification_score + right_tree.classification_score), upper_bound);

            tree.best_left_label = left_label;
            tree.best_right_label = right_label;
        } else {
            tree.can_skip = tree.classification_score - (left_classification_score + right_classification_score);
            global_skip = std::max(global_skip, upper_bound - (dataset_size - left_tree.classification_score - right_tree.classification_score));
        }

        int remaining_size = tree.size - tree.current_element_count;
        tree.is_zero |= (right_classification_score == remaining_size) || (tree.can_skip >= remaining_size);

        if (left_tree.is_zero && right_tree.is_zero) {
            break;
        }

        if (left_tree.can_skip >= left_tree.size - left_tree.current_element_count && right_tree.can_skip >= right_tree.size - right_tree.current_element_count) {
            break;
        }

        tree.current_element_count++;
        tree.current_label_frequency[current_feature_data.label]++;
        tree.previous_value = current_feature_data.value;
        tree.previous_unique_value_index = current_feature_data.unique_value_index;
    }
}

void SpecializedSolver::create_optimal_decision_tree(const Dataview& dataview, const Configuration& solution_configuration, int feature_index, std::shared_ptr<Tree> &current_optimal_decision_tree, int upper_bound) {
    const std::vector<Dataset::FeatureElement>& current_feature = dataview.get_sorted_dataset_feature(feature_index);

    const auto& possible_split_indices = dataview.get_possible_split_indices(feature_index);
    IntervalsPruner interval_pruner(possible_split_indices, solution_configuration.max_gap);

    std::queue<IntervalsPruner::Bound> unsearched_intervals;
    unsearched_intervals.push({0, (int)possible_split_indices.size() - 1, -1, -1});

    while(!unsearched_intervals.empty()) {
        auto current_interval = unsearched_intervals.front(); unsearched_intervals.pop();

        if (interval_pruner.subinterval_pruning(current_interval, current_optimal_decision_tree->misclassification_score)) {
            continue;
        }

        interval_pruner.interval_shrinking(current_interval, current_optimal_decision_tree->misclassification_score);
        const auto& [left, right, current_left_bound, current_right_bound] = current_interval;
        if(left > right) {
            continue;
        }

        const int mid = (left + right) / 2;
        const int split_point = possible_split_indices[mid];

        const float threshold = mid > 0 ? (current_feature[possible_split_indices[mid - 1]].value + current_feature[split_point].value) / 2.0f 
                                  : (current_feature[split_point].value + current_feature[0].value) / 2.0f;  

        std::shared_ptr<Tree> left_optimal_dt = std::make_shared<Tree>();
        std::shared_ptr<Tree> right_optimal_dt = std::make_shared<Tree>();

        statistics::total_number_of_specialized_solver_calls += 1;
        get_best_left_right_scores(dataview, feature_index, split_point, threshold, left_optimal_dt, right_optimal_dt, current_optimal_decision_tree->misclassification_score);
        RUNTIME_ASSERT(left_optimal_dt->misclassification_score >= 0, "D2 - Left tree should have non-negative misclassification score.");
        RUNTIME_ASSERT(right_optimal_dt->misclassification_score >= 0, "D2 - Right tree should have non-negative misclassification score.");

        const int current_best_score = left_optimal_dt->misclassification_score + right_optimal_dt->misclassification_score;

        if (current_best_score < current_optimal_decision_tree->misclassification_score) {

            current_optimal_decision_tree->misclassification_score = current_best_score;
            current_optimal_decision_tree->update_split(feature_index, threshold, left_optimal_dt, right_optimal_dt);

            upper_bound = std::min(upper_bound, current_best_score);

            if (current_best_score == 0) {
                return;
            }

            if (PRINT_INTERMEDIARY_TIME_SOLUTIONS && solution_configuration.is_root) {
                const auto stop = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time);
                std::cout << "Time taken to get the misclassification score " << current_best_score << ": " << duration.count() / 1000.0 << " seconds" << std::endl;
            }

        }

        interval_pruner.add_result(mid, left_optimal_dt->misclassification_score, right_optimal_dt->misclassification_score);

        if (left == right) {
            continue;
        }

        const int score_difference = current_best_score - current_optimal_decision_tree->misclassification_score;
        const auto [new_bound_left, new_bound_right] = interval_pruner.neighbourhood_pruning(score_difference, left, right, mid);

        if (new_bound_left <= right) {
            unsearched_intervals.push({new_bound_left, right, mid, current_right_bound});
        }

        if (left <= new_bound_right) {
            unsearched_intervals.push({left, new_bound_right, current_left_bound, mid});
        }
    }
}