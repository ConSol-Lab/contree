#ifndef STATISTICS_H
#define STATISTICS_H

class statistics {
public:
    unsigned long long total_number_of_specialized_solver_calls{ 0 };
    unsigned long long total_number_of_general_solver_calls{ 0 };
    unsigned long long total_number_cache_hits{ 0 };

    bool should_print{ false };

    void print_statistics();
};

#endif // STATISTICS_H


