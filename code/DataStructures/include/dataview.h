#ifndef DATAVIEW_H   
#define DATAVIEW_H

#include <vector>

#include "dataset.h"
#include "dynamic_bitset.h"

class Dataview;

struct DataviewBitset {
    dynamic_bitset bitset;
    size_t size{ 0 };
    long long hash{ -1 };

    DataviewBitset() = default;
    DataviewBitset(const Dataview& data);
    inline bool operator==(const DataviewBitset& other) const { return size == other.size && bitset == other.bitset; }
    inline bool operator!=(const DataviewBitset& other) const { return !((*this) == other); }
    inline bool is_hash_set() const { return hash != -1; }
    inline bool is_bitset_set() const { return size > 0; }
    inline void set_hash(long long _hash) { hash = _hash; }
    inline long long get_hash() const { return hash; }
};

template <>
struct std::hash<DataviewBitset> {

    size_t operator()(const DataviewBitset& view) const {
        if (view.is_hash_set()) return view.get_hash();
        return hash<dynamic_bitset>()(view.bitset);
    }

};

class Dataview {
public:
    Dataview(int class_number, const bool sort_by_gini_index) : label_frequency(class_number, 0), class_number(class_number), sort_by_gini_index(sort_by_gini_index) {};
    Dataview(Dataset* sorted_dataset, Dataset* unsorted_dataset, int class_number, const bool sort_by_gini_index);

    int get_dataset_size() const;
    int get_feature_number() const;

    const std::vector<Dataset::FeatureElement>& get_sorted_dataset_feature(int feature_index) const;
    const std::vector<Dataset::FeatureElement>& get_unsorted_dataset_feature(int feature_index) const;

    const std::vector<int>& get_possible_split_indices(int feature_index) const;

    int get_class_number() const;
    const std::vector<int>& get_label_frequency() const;

    static void split_data_points(const Dataview& current_dataview, int feature_index, int split_point, int split_unique_value_index, Dataview& left_data, Dataview& right_data, int current_max_depth);
    static void initialize_split_parameters(const std::vector<Dataset::FeatureElement>& current_feature, int class_number, const std::vector<int>& current_label_frequency, int split_point, std::vector<int> &left_label_frequency, std::vector<int> &right_label_frequency);

    DataviewBitset& get_bitset() const {
        if (!bitset.is_bitset_set()) bitset = DataviewBitset(*this);
        return bitset;
    }

    inline long long get_hash() const { return bitset.get_hash(); }
    inline bool is_hash_set() const { return bitset.is_hash_set(); }
    inline void set_hash(long long hash) const { bitset.set_hash(hash); }

    std::vector<std::pair<float, int>> gini_values;

    const bool should_sort_by_gini_index() const { return sort_by_gini_index; }

private:
    std::vector<std::vector<Dataset::FeatureElement>> feature_data;
    std::vector<std::vector<int>> possible_split_indices;
    int class_number;
    std::vector<int> label_frequency;

    Dataset* unsorted_dataset; 

    mutable DataviewBitset bitset;

    const bool sort_by_gini_index;
};



#endif // DATAVIEW_H