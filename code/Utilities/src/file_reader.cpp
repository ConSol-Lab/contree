#include "file_reader.h"

void file_reader::read_file(const std::string &filename, Dataset& data, int& class_number) {
    std::istream* input_stream;
    std::ifstream file_stream;

    if (filename.empty()) {
        input_stream = &std::cin;
    } else {
        file_stream.open(filename);
        if (!file_stream) {
            std::cerr << "Failed to open the file: " << filename << std::endl;
            exit(1);
        }
        input_stream = &file_stream;
    }

    int data_point_index = 0;
    std::string line;

    class_number = 1;
    int feature_number = 0;
    while (std::getline(*input_stream, line) && !line.empty()) {
        std::istringstream iss(line);
        float tmp_value;

        int label;
        iss >> label;
        class_number = std::max(class_number, label + 1);

        int feature_index = 0;
        while (iss >> tmp_value) {
            data.add_feature_index_pair(feature_index, data_point_index, tmp_value, label);
            feature_index++;
            feature_number = std::max(feature_index, feature_number);
        }
        RUNTIME_ASSERT(feature_number == feature_index, "Each row should have the same number of features.");

        data_point_index++;
    }
    data.compute_unique_value_indices();
}

