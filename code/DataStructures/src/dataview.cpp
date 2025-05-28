#include "dataview.h"

Dataview::Dataview(Dataset* sorted_dataset, Dataset* unsorted_dataset, int class_number, const bool sort_by_gini_index) 
    : unsorted_dataset(unsorted_dataset), label_frequency(class_number, 0), class_number(class_number), sort_by_gini_index(sort_by_gini_index) {

    feature_data.resize(sorted_dataset->get_features_size());
    possible_split_indices.resize(sorted_dataset->get_features_size());

    const auto& first_feature = sorted_dataset->get_feature(0);
    feature_data[0].resize(first_feature.size());

    gini_values.resize(sorted_dataset->get_features_size());

    int last_unique_index = -1;
    for (int feature_element_idx = 0; feature_element_idx < first_feature.size(); feature_element_idx++) {
        feature_data[0][feature_element_idx] = std::move(first_feature[feature_element_idx]);
        label_frequency[first_feature[feature_element_idx].label]++;

        if (first_feature[feature_element_idx].unique_value_index != last_unique_index && last_unique_index != -1) {
            possible_split_indices[0].push_back(feature_element_idx);
        }

        last_unique_index = first_feature[feature_element_idx].unique_value_index;
    }

    float best_gini = 1.0f;

    if (sort_by_gini_index) {
        std::vector<int> left_label_frequency(class_number, 0);
        std::vector<int> right_label_frequency(label_frequency);

        for (int i = 0; i < first_feature.size() - 1; i++) {
            right_label_frequency[first_feature[i].label]--;
            left_label_frequency[first_feature[i].label]++;

            float left_gini = 1.0f; float right_gini = 1.0f;
            int left_count = i + 1; int right_count = int(first_feature.size()) - left_count;


            for (int label = 0; label < class_number; label++) {
                float left_probability = static_cast<float>(left_label_frequency[label]) / static_cast<float>(left_count);
                left_gini -= left_probability * left_probability;

                float right_probability = static_cast<float>(right_label_frequency[label]) / static_cast<float>(right_count);
                right_gini -= right_probability * right_probability;
            }

            float gini_index = (left_gini * left_count + right_gini * right_count) / first_feature.size();

            if (gini_index < best_gini) {
                best_gini = gini_index;
            }
        }
    }

    gini_values[0] = std::make_pair(best_gini, 0);

    for (int feature_idx = 1; feature_idx < sorted_dataset->get_features_size(); feature_idx++) {
        const auto& current_feature = sorted_dataset->get_feature(feature_idx);
        feature_data[feature_idx].resize(current_feature.size());
        last_unique_index = -1;

        
        std::vector<int> left_label_frequency(class_number, 0);
        std::vector<int> right_label_frequency(label_frequency);
        best_gini = 1.0f;

        for (int feature_element_idx = 0; feature_element_idx < current_feature.size(); feature_element_idx++) {
            feature_data[feature_idx][feature_element_idx] = std::move(current_feature[feature_element_idx]);

            if (current_feature[feature_element_idx].unique_value_index != last_unique_index && last_unique_index != -1) {
                possible_split_indices[feature_idx].push_back(feature_element_idx);
            }

            last_unique_index = current_feature[feature_element_idx].unique_value_index;

            if (sort_by_gini_index) {
                right_label_frequency[current_feature[feature_element_idx].label]--;
                left_label_frequency[current_feature[feature_element_idx].label]++;

                float left_gini = 1.0f; float right_gini = 1.0f;
                int left_count = feature_element_idx + 1; int right_count = int(current_feature.size()) - left_count;


                for (int label = 0; label < class_number; label++) {
                    if (left_count > 0) {
                        float left_probability = static_cast<float>(left_label_frequency[label]) / static_cast<float>(left_count);
                        left_gini -= left_probability * left_probability;
                    }

                    if (right_count > 0) {
                        float right_probability = static_cast<float>(right_label_frequency[label]) / static_cast<float>(right_count);
                        right_gini -= right_probability * right_probability;
                    }
                }

                float gini_index = (left_gini * left_count + right_gini * right_count) / current_feature.size();

                if (gini_index < best_gini) {
                    best_gini = gini_index;
                }

                
            }
        }

        gini_values[feature_idx] = std::make_pair(best_gini, feature_idx);
    }

    if (sort_by_gini_index) {
        std::sort(gini_values.begin(), gini_values.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        });
    }
}

int Dataview::get_dataset_size() const {
    return int(feature_data[0].size());
}

int Dataview::get_feature_number() const {
    return int(feature_data.size());
}

const std::vector<Dataset::FeatureElement>& Dataview::get_sorted_dataset_feature(int feature_index) const {
    return feature_data[feature_index];
}

int Dataview::get_class_number() const {
    return class_number;
}

const std::vector<int>& Dataview::get_label_frequency() const {
    return label_frequency;
}

const std::vector<Dataset::FeatureElement>& Dataview::get_unsorted_dataset_feature(int feature_index) const {
    return unsorted_dataset->feature_data[feature_index];
}

const std::vector<int>& Dataview::get_possible_split_indices(int feature_index) const {
    return possible_split_indices[feature_index];
}

void Dataview::split_data_points(const Dataview& current_dataview, int feature_index, int split_point, int split_unique_value_index, Dataview& left_dataview, Dataview& right_dataview, int current_max_depth) {
    left_dataview.feature_data.resize(current_dataview.get_feature_number());
    right_dataview.feature_data.resize(current_dataview.get_feature_number());

    left_dataview.possible_split_indices.resize(current_dataview.get_feature_number());
    right_dataview.possible_split_indices.resize(current_dataview.get_feature_number());

    left_dataview.gini_values.resize(current_dataview.get_feature_number());
    right_dataview.gini_values.resize(current_dataview.get_feature_number());

    const int class_number = current_dataview.get_class_number();

    const auto& current_feature = current_dataview.get_sorted_dataset_feature(feature_index);
    const auto& unsorted_split_feature = current_dataview.unsorted_dataset->feature_data[feature_index];

    Dataview::initialize_split_parameters(current_feature, class_number, current_dataview.label_frequency, split_point, left_dataview.label_frequency, right_dataview.label_frequency);

    int left_size = split_point;
    int right_size = int(current_feature.size()) - split_point;
  
    int feature_no = 0;

    for (const auto& it : current_dataview.feature_data) {
        auto& left_split_feature_data = left_dataview.feature_data[feature_no];
        left_split_feature_data.resize(left_size);

        auto& right_split_feature_data = right_dataview.feature_data[feature_no];
        right_split_feature_data.resize(right_size);
        
        std::vector<int> left_value_change_indices; left_value_change_indices.reserve(it.size());
        std::vector<int> right_value_change_indices; right_value_change_indices.reserve(it.size());

        int left_last_unique_index = -1;
        int rigth_last_unique_index = -1;

        int left_counter = 0;
        int right_counter = 0;

        float left_best_gini = 1.0f;
        float right_best_gini = 1.0f;

        std::vector<int> left_tree_left_label_frequency(class_number, 0);
        std::vector<int> left_tree_right_label_frequency(left_dataview.label_frequency);

        std::vector<int> right_tree_left_label_frequency(class_number, 0);
        std::vector<int> right_tree_right_label_frequency(right_dataview.label_frequency);

        for (const auto& feature_data : it) {
            if (unsorted_split_feature[feature_data.data_point_index].unique_value_index >= split_unique_value_index) {
                right_split_feature_data[right_counter] = feature_data;

                if (feature_data.unique_value_index != rigth_last_unique_index && rigth_last_unique_index != -1) {
                    right_value_change_indices.emplace_back(right_counter);
                }

                rigth_last_unique_index = feature_data.unique_value_index;
                right_counter++;

                if (current_dataview.sort_by_gini_index) {
                    right_tree_right_label_frequency[feature_data.label]--;
                    right_tree_left_label_frequency[feature_data.label]++;

                    float left_gini = 1.0f; float right_gini = 1.0f;
                    int left_count = right_counter; int right_count = int(current_feature.size()) - left_count;

                    for (int label = 0; label < class_number; label++) {
                        if (left_count > 0) {
                            float left_probability = static_cast<float>(right_tree_left_label_frequency[label]) / static_cast<float>(left_count);
                            left_gini -= left_probability * left_probability;
                        }
                        if (right_count > 0) {
                            float right_probability = static_cast<float>(right_tree_right_label_frequency[label]) / static_cast<float>(right_count);
                            right_gini -= right_probability * right_probability;
                        }
                    }

                    float gini_index = (left_gini * left_count + right_gini * right_count) / current_feature.size();

                    if (gini_index < right_best_gini) {
                        right_best_gini = gini_index;
                    }
                }
            } else {
                left_split_feature_data[left_counter] = feature_data;

                if (feature_data.unique_value_index != left_last_unique_index && left_last_unique_index != -1) {
                    left_value_change_indices.emplace_back(left_counter);
                }

                left_last_unique_index = feature_data.unique_value_index;
                left_counter++;

                if (current_dataview.sort_by_gini_index) {
                    left_tree_right_label_frequency[feature_data.label]--;
                    left_tree_left_label_frequency[feature_data.label]++;

                    float left_gini = 1.0f; float right_gini = 1.0f;
                    int left_count = left_counter; int right_count = int(current_feature.size()) - left_count;

                    for (int label = 0; label < class_number; label++) {
                        if (left_count > 0) {
                            float left_probability = static_cast<float>(left_tree_left_label_frequency[label]) / static_cast<float>(left_count);
                            left_gini -= left_probability * left_probability;
                        }

                        if (right_count > 0) {
                            float right_probability = static_cast<float>(left_tree_right_label_frequency[label]) / static_cast<float>(right_count);
                            right_gini -= right_probability * right_probability;
                        }
                    }

                    float gini_index = (left_gini * left_count + right_gini * right_count) / current_feature.size();

                    if (gini_index < left_best_gini) {
                        left_best_gini = gini_index;
                    }
                }
            }
        }

        left_dataview.possible_split_indices[feature_no] = std::move(left_value_change_indices);
        right_dataview.possible_split_indices[feature_no] = std::move(right_value_change_indices);

        left_dataview.gini_values[feature_no] = {left_best_gini, feature_no};
        right_dataview.gini_values[feature_no] = {right_best_gini, feature_no};

        feature_no++;
    }

    left_dataview.unsorted_dataset = current_dataview.unsorted_dataset;
    right_dataview.unsorted_dataset = current_dataview.unsorted_dataset;

    if (current_dataview.sort_by_gini_index && current_max_depth > 3) {
        std::sort(left_dataview.gini_values.begin(), left_dataview.gini_values.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        });

        std::sort(right_dataview.gini_values.begin(), right_dataview.gini_values.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        });
    }
}

void Dataview::initialize_split_parameters(const std::vector<Dataset::FeatureElement>& current_feature, int class_number, const std::vector<int> &current_label_frequency, int split_point, std::vector<int> &left_label_frequency, std::vector<int> &right_label_frequency) {
    if (split_point < current_feature.size() - split_point) {
        for (int left_counter = 0; left_counter < split_point; left_counter++) {
            left_label_frequency[current_feature[left_counter].label]++;
        }

        for (int label_instance = 0; label_instance < class_number; label_instance++) {
            right_label_frequency[label_instance] = current_label_frequency[label_instance] - left_label_frequency[label_instance];
        }
    } else {
        for (int right_counter = split_point; right_counter < current_feature.size(); right_counter++) {
            right_label_frequency[current_feature[right_counter].label]++;
        }

        for (int label_instance = 0; label_instance < class_number; label_instance++) {
            left_label_frequency[label_instance] = current_label_frequency[label_instance] - right_label_frequency[label_instance];
        }
    }
}

DataviewBitset::DataviewBitset(const Dataview& dataview) 
    : size(dataview.get_dataset_size()), 
      bitset(dataview.get_unsorted_dataset_feature(0).size()) {
    
    const auto& instances = dataview.get_sorted_dataset_feature(0);
    for (const auto& instance : instances) {
        bitset.set_bit(instance.data_point_index);
    }
}