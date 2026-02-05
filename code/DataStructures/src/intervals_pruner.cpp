#include "intervals_pruner.h"
#include "configuration.h"

IntervalsPruner::IntervalsPruner(const std::vector<int>& possible_split_indexes_ref, int max_gap, float complexity_cost)
    : possible_split_indexes(possible_split_indexes_ref), possible_split_size(int(possible_split_indexes.size())), max_gap(max_gap), complexity_cost(complexity_cost) {
        evaluated_indices_record.reserve(possible_split_indexes.size() / 8);
        rightmost_zero_index = possible_split_size;
        leftmost_zero_index = -1;
    }

bool IntervalsPruner::subinterval_pruning(const IntervalsPruner::Bound& current_bounds, float current_best_score) {
    // Given an interval, we are going to check the closest index to the left of left_bound
    // and to the right of right_bound to see what their left and right misclassification scores were
    // we add those, and if this exceeds the current best, we know that adding anything in between will not improve the score

    // We start with setting the misclassification scores to zero for the left and right subtree
    int left_bound_score_left = 0;
    int right_bound_score_right = 0;

    // If we have a solution for a split index left to the current interval, 
    // we take its left subtree misclassification score
    if (current_bounds.last_split_left_index != -1) {
        left_bound_score_left = evaluated_indices_record[current_bounds.last_split_left_index].first;
    }

    // If we have a solution for a split index right to the current interval, 
    // we take its right subtree misclassification score
    if (current_bounds.last_split_right_index != -1) {
        right_bound_score_right = evaluated_indices_record[current_bounds.last_split_right_index].second;
    }

    // If summed, these two scores are more than the current best score, we return true, to indicate that this interval must be pruned
    return left_bound_score_left + right_bound_score_right + max_gap + complexity_cost + EPSILON >= current_best_score;
}

void IntervalsPruner::interval_shrinking(IntervalsPruner::Bound& current_bounds, float current_best_score) {
    // Check if an interval can be shrinked based on updated results since we added the interval to the queue.
    // For this, we are going to check the solutions at the closest split indices to the left and right of the interval
    // indicated by last_split_left_index and last_split_right_index.

    // First, if the leftmost_zero_index or rightmost_zero_index have been updated since the interval
    // was added, we update the left and right bounds of the interval
    current_bounds.left_bound = std::max(current_bounds.left_bound, leftmost_zero_index + 1);
    current_bounds.right_bound = std::min(current_bounds.right_bound, rightmost_zero_index - 1);

    // If the current interval has an investigated split to the left of it
    if (current_bounds.last_split_left_index != -1) {
        // Compute the score difference of this split to the left relative to the current best solution
        float updated_score_difference = (evaluated_indices_record[current_bounds.last_split_left_index].first + evaluated_indices_record[current_bounds.last_split_left_index].second + complexity_cost) - current_best_score + max_gap;
        // If the split to the left is not optimal, we can prune around it.
        if (updated_score_difference > 0) {
            // We compute the extended_left_bound by taking the left split index and add the score difference + 1.
            // Plus one, because we want to improve on the current best solution.
            int extended_left_bound = possible_split_indexes[current_bounds.last_split_left_index] + static_cast<int>(std::floor(updated_score_difference)) + 1;

            // If the extended_left_bound is within the range of this interval, we seek for the smallest split-index
            // within the interval that is larger than it.
            if (extended_left_bound <= possible_split_indexes[current_bounds.right_bound]) {
                int new_left_bound = int(std::lower_bound(possible_split_indexes.begin() + current_bounds.left_bound, possible_split_indexes.begin() + current_bounds.right_bound, extended_left_bound) - possible_split_indexes.begin());
                current_bounds.left_bound = std::max(current_bounds.left_bound, new_left_bound);
            } else {
                // otherwise, we prune the interval
                current_bounds.left_bound = 1;
                current_bounds.right_bound = 0;
                current_bounds.last_split_left_index = -1;
                current_bounds.last_split_right_index = -1;
                return;
            }
        }
    }

    // We do the same procedure for the split on the right.
    if (current_bounds.last_split_right_index != -1) {
        float updated_score_difference = (evaluated_indices_record[current_bounds.last_split_right_index].first + evaluated_indices_record[current_bounds.last_split_right_index].second + complexity_cost) - current_best_score + max_gap;
        if (updated_score_difference > 0) {
            int extended_right_bound = possible_split_indexes[current_bounds.last_split_right_index] - (static_cast<int>(std::floor(updated_score_difference)) + 1);

            if (extended_right_bound >= possible_split_indexes[current_bounds.left_bound]) {
                int new_right_bound = int(std::upper_bound(possible_split_indexes.begin() + current_bounds.left_bound, possible_split_indexes.begin() + current_bounds.right_bound, extended_right_bound) - possible_split_indexes.begin());
                current_bounds.right_bound = std::min(current_bounds.right_bound, new_right_bound);
            } else {
                current_bounds.left_bound = 1;
                current_bounds.right_bound = 0;
                current_bounds.last_split_left_index = -1;
                current_bounds.last_split_right_index = -1;
                return;
            }
        }
    }
}

std::pair<int, int> IntervalsPruner::neighbourhood_pruning(float score_difference, int left, int right, int split_index) {
    // Currently we have the interval [left, right], with split_index = (left + right) / 2. All these are threshold indices
    // we now want to find two new intervals [a, b] and [c, d] 
    // with [a, b] subseteq of [left, split_index - 1]
    // and  [c, d] subseteq of [split_index + 1, right]
    // Since a = left and d = right, we simply return b and c.
    // Here, b = new_bound_right, c = new_bound_left
    
    // Based on score difference, we can prune split points that are less samples than 'score-difference' away from split-index.
    // If score difference is zero (the current solution is optimal), then we prune nothing and directly return by only pruning
    // the split-index.   
    score_difference += max_gap;
    if (score_difference <= EPSILON) {
        return {split_index + 1, split_index - 1};
    }

    // If we have already found a zero solution at leftmost_zero_index, then, seeking anything left of it, is redundant.
    // Therefore we set new_bound_left to the max of split_index + 1 and leftmost_zero_index + 1, and idem for right
    int new_bound_left = std::max(split_index + 1, leftmost_zero_index + 1);
    int new_bound_right = std::min(split_index - 1, rightmost_zero_index - 1);

    // We now seek for the sample index which may yield and improving solution. Therefore, we add +1, since the score
    // difference only brings us to a place where we can equal the current best solution
    int minimum_left_value  = possible_split_indexes[split_index] - (static_cast<int>(std::floor(score_difference)) + 1);
    int minimum_right_value = possible_split_indexes[split_index] + (static_cast<int>(std::floor(score_difference)) + 1);

    if (minimum_right_value > possible_split_indexes[right]) {
        // We first check if this minimum required value is even within the range of the current interval.
        // If not, we return new_bound_left = right + 1, which effectively drops this interval since [right + 1, right] is an empty set.
        new_bound_left = right + 1;
    } else {
        // Otherwise, we need to use binary search (since there can be multiple samples with the same unique_value_index)
        // to find the right split-index such that minimum_right_value is to the right of it (greater or equal).
        // For this we use std::lower_bound 
        new_bound_left = int(std::lower_bound(possible_split_indexes.begin() + new_bound_left, possible_split_indexes.begin() + right, minimum_right_value) - possible_split_indexes.begin());
    }

    // We do the same for the other interval
    if (minimum_left_value < possible_split_indexes[left]) {
        new_bound_right = left - 1;
    } else {
        // Now we need to find the split-index such that the minimum_left_value is to the left of it (less or equal)
        new_bound_right = int(std::upper_bound(possible_split_indexes.begin() + left, possible_split_indexes.begin() + new_bound_right, minimum_left_value) - possible_split_indexes.begin());
    }

    return {new_bound_left, new_bound_right};
}

void IntervalsPruner::add_result(int index, float left_score, float right_score) {
    if (left_score <= EPSILON) {
        leftmost_zero_index = std::max(leftmost_zero_index, index);
    }

    if (right_score <= EPSILON) {
        rightmost_zero_index = std::min(rightmost_zero_index, index);
    }

    if (left_score == -1) {
        left_score = 0;
    }

    if (right_score == -1) {
        right_score = 0;
    }

    evaluated_indices_record[index] = {left_score, right_score};
}
