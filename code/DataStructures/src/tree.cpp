#include "tree.h"

Tree::Tree() = default;

Tree::Tree(int label, float objective) 
    : label(label), objective(objective) {}

Tree::Tree(int split_feature, float split_threshold, const std::shared_ptr<Tree>& left, const std::shared_ptr<Tree>& right, float complexity_cost) 
    : split_feature(split_feature), split_threshold(split_threshold), 
      left(left), right(right), objective(left->objective + right->objective + complexity_cost) {}

bool Tree::is_leaf() const { 
    return split_feature == -1; 
}

bool Tree::is_internal() const { 
    return !is_leaf(); 
}

bool Tree::is_initialized() const { 
    return split_feature != -1 || label != -1; 
}

int Tree::get_depth() const {
    if (is_leaf()) return 0;
    return 1 + std::max(left->get_depth(), right->get_depth());
}

int Tree::get_num_branching_nodes() const {
    if (is_leaf()) return 0;
    return 1 + left->get_num_branching_nodes() + right->get_num_branching_nodes();
}

void Tree::make_leaf(int label, int misclassifications) {
    RUNTIME_ASSERT(misclassifications >= 0, "Leaf should have non-negative misclassifications.");
    RUNTIME_ASSERT(label >= 0, "Leaf label should be non-negative.");
    this->label = label;
    this->objective = misclassifications;
    left = nullptr;
    right = nullptr;
    split_feature = -1;
    split_threshold = 0.0;
}

void Tree::update_split(int split_feature, float split_threshold, const std::shared_ptr<Tree>& left, const std::shared_ptr<Tree>& right, float complexity_cost) {
    this->label = -1;
    this->split_feature = split_feature;
    this->split_threshold = split_threshold;
    this->left = left;
    this->right = right;
    this->objective = left->objective + right->objective + complexity_cost;
}

std::ostream& operator<<(std::ostream& os, const Tree& t) {
    if (t.is_leaf()) {
        os << t.label;
    } else {
        os << "[" << t.split_feature << "," << t.split_threshold << ","
           << *t.left << "," << *t.right << "]";
    }
    return os;
}

std::string Tree::to_string(int indent) const {
    std::string result = "";
    for (int i = 0; i < indent; i++) {
        result += "  ";
    }
    if (is_leaf()) {
        return result + "Label = " + std::to_string(label) + "\n";
    } else {
        result += "Split on f" + std::to_string(split_feature) + " <= " + std::to_string(split_threshold) + "\n";
        result += left->to_string(indent + 1);
        result += right->to_string(indent + 1);
        return result;
    }
}