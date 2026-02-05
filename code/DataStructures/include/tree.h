#ifndef TREE_H
#define TREE_H

#include <iostream>
#include <climits>
#include <memory>
#include <string> 
#include "dataset.h"

struct Tree {
    Tree();
    Tree(int label, float objective);
    Tree(int split_feature, float split_threshold, const std::shared_ptr<Tree>& left, const std::shared_ptr<Tree>& right, float complexity_cost);

    std::shared_ptr<Tree> left{ nullptr }, right{ nullptr };
    int split_feature = -1, label = -1;
    float split_threshold = 0.0;

    float objective = INT_MAX;

    bool is_leaf() const;
    bool is_internal() const;
    bool is_initialized() const;

    int get_depth() const;
    int get_num_branching_nodes() const;
    inline int get_num_leaf_nodes() const { return get_num_branching_nodes() + 1; }

    inline int get_split_feature() const { return split_feature; }
    inline float get_split_threshold() const { return split_threshold; }
    inline int get_label() const { return label; }
    inline std::shared_ptr<Tree> get_left_tree() const { return left; }
    inline std::shared_ptr<Tree> get_right_tree() const { return right; }

    void make_leaf(int label, int misclassifications);
    void update_split(int split_feature, float split_threshold, const std::shared_ptr<Tree>& left, const std::shared_ptr<Tree>& right, float complexity_cost);

    std::string to_string(int indent = 0) const;
};

std::ostream& operator<<(std::ostream& os, const Tree& t);

#endif //TREE_H
