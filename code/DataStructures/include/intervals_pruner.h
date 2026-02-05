#ifndef INTERVALS_PRUNER_H
#define INTERVALS_PRUNER_H

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cmath>

class IntervalsPruner {
public:
    /**
     * Constructor for IntervalsPruner.
     * 
     * Initializes the IntervalsPruner with a reference to a vector of possible split indexes
     * and a maximum allowable gap, which is used to control the permissibility of suboptimal solutions.
     * 
     * @param possible_split_indexes_ref Reference to a vector containing possible split indices.
     * @param max_gap The maximum allowable gap for the solution to be considered valid (off by at most max_gap).
     * @param complexity_cost The cost of adding a branching node
     */
    IntervalsPruner(const std::vector<int>& possible_split_indexes_ref, int max_gap, float complexity_cost);

    /**
    * We store the intervals using four values
    */
    struct Bound {
        int left_bound;             // the left split-index bound (inclusive) of the interval
        int right_bound;            // the right split-index bound (inclusive) of the interval
        int last_split_left_index;  // the last split index tested to the left of the left_bound (-1 if not set)
        int last_split_right_index; // the last split index tested to the right of the right_bound (-1 if not set)
    };

    /**
     * Performs neighborhood pruning by evaluating the interval around a given split index.
     * 
     * This method reduces the interval size based on the difference between the score at the current split
     * and the provided score difference, removing the middle part of the interval if it cannot lead to a better
     * solution.
     * 
     * @param score_difference The difference in scores used to determine pruning.
     * @param left The left boundary of the interval.
     * @param right The right boundary of the interval.
     * @param split_index The index at which the split is evaluated.
     * @return A pair of integers representing the new pruned interval bounds.
     */
    std::pair<int, int> neighbourhood_pruning(float score_difference, int left, int right, int split_index);

    /**
     * Applies subinterval pruning to determine if a given interval can be entirely pruned.
     * 
     * This method checks if the entire subinterval defined by current_bounds can be discarded based
     * on the current best score.
     * 
     * @param current_bounds The current bounds of the interval being evaluated.
     * @param current_best_score The best score obtained so far, used as a reference for pruning.
     * @return True if the subinterval can be pruned, otherwise false.
     */
    bool subinterval_pruning(const Bound& current_bounds, float current_best_score);

    /**
     * Performs interval shrinking by narrowing the bounds of the interval based on the current best score.
     * 
     * This method adjusts the interval size by eliminating portions of the interval that cannot lead to a
     * better solution, based on the difference between the current score and the best known score known when
     * the bound was created.
     * 
     * @param current_bounds The current bounds of the interval to be updated by shrinking
     * @param current_best_score The best score to compare against during the shrinking process.
     */
    void interval_shrinking(Bound& current_bounds, float current_best_score);

    /**
     * Records the result of a split, storing the index and associated scores.
     * 
     * This method stores the result of a split in the evaluated indices record, including the index
     * and the scores for both the left and right subintervals, used for later reference during pruning.
     * 
     * @param index The index at which the split was performed.
     * @param left_score The score associated with the left subinterval.
     * @param right_score The score associated with the right subinterval.
     */
    void add_result(int index, float left_score, float right_score);

private:
    const std::vector<int>& possible_split_indexes; 
    int possible_split_size;                    
    int rightmost_zero_index;                      
    int leftmost_zero_index;                       
    int max_gap;
    float complexity_cost;
    std::unordered_map<int, std::pair<float, float>> evaluated_indices_record; 
};

#endif // INTERVALS_PRUNER_H
