#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "stopwatch.h"
#include "statistics.h"
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <memory>

#define EPSILON 0.0000001f

#define PRINT_INTERMEDIARY_TIME_SOLUTIONS 0

#ifdef NDEBUG
#define RUNTIME_ASSERT(cond, msg) ((void)0)
#else
#define RUNTIME_ASSERT(cond, msg)                                             \
        do {                                                                      \
            if (!(cond)) {                                                        \
                std::cerr << "Assertion failed: " << #cond                        \
                          << "\nMessage: " << msg                                 \
                          << "\nFile: " << __FILE__                               \
                          << "\nLine: " << __LINE__ << std::endl;                 \
                std::terminate();                                                 \
            }                                                                     \
        } while (0)
#endif

struct Configuration {

    Configuration() : stats(std::make_shared<statistics>()) {}

    int max_depth{ 3 };
    int max_gap{ 0 };
    float max_gap_decay{ 0.0 };
    bool print_logs{ false };
    bool use_upper_bound{ true };
    bool is_root{ false };
    bool sort_gini{ false };
    float complexity_cost{ 0.0 };
    Stopwatch stopwatch;
    std::shared_ptr<statistics> stats{ nullptr };
    Configuration GetLeftSubtreeConfig() const;
    Configuration GetRightSubtreeConfig(int left_gap) const;
};

#endif // CONFIGURATION_H
