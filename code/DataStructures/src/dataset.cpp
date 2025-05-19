#include "dataset.h"
#include <numeric>

int Dataset::get_instance_number() const {
    return (int) feature_data[0].size();
}

const std::vector<std::vector<Dataset::FeatureElement>>& Dataset::get_features_data() const {
    return feature_data;
}

void Dataset::add_feature_index_pair(int feature_index, int data_point_index, float value, int label) {
    if(data_point_index == 0) {
        feature_data.emplace_back();
    }

    feature_data[feature_index].push_back({value, -1, data_point_index, label});
}

const std::vector<Dataset::FeatureElement>& Dataset::get_feature(int index) const {
    return feature_data[index];
}

int Dataset::get_features_size() const {
    return (int) feature_data.size();
}

void Dataset::sort_feature_values() {
    for (auto &it : feature_data) {
        std::sort(it.begin(), it.end(),
                  [](const FeatureElement& first, const FeatureElement& second) {
                      return first.value < second.value;
                  }
        );
    }
}

void Dataset::compute_unique_value_indices() {
    std::vector<size_t> idx(get_instance_number());
    for (auto& cur_feature_data : feature_data) {
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&cur_feature_data](size_t i1, size_t i2) {return cur_feature_data[i1].value < cur_feature_data[i2].value;});
        double prev = -1.0f;
        bool first = true;
        int cur_unique_value_index = -1;
        for (size_t ix : idx) {
            auto& cur_feature_element = cur_feature_data[ix];
            if (first || cur_feature_element.value - prev >= EPSILON) cur_unique_value_index++;
            cur_feature_element.unique_value_index = cur_unique_value_index;
            prev = cur_feature_element.value;
            first = false;
        }

    }
}