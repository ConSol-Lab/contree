/**
From Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/
#ifndef STOPWATCH_H   
#define STOPWATCH_H
#include <chrono>

class Stopwatch {
public:
	Stopwatch() : time_limit_in_seconds(0) {}

	void Initialise(double time_limit_in_seconds) {
		starting_time = std::chrono::high_resolution_clock::now();
		this->time_limit_in_seconds = time_limit_in_seconds;
	}

	double TimeElapsedInSeconds() const {
		auto stop = std::chrono::high_resolution_clock::now();
		return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - starting_time).count()) / 1000.0;
	}

	double TimeLeftInSeconds() const {
		return time_limit_in_seconds - TimeElapsedInSeconds();
	}

	bool IsWithinTimeLimit() const {
		return TimeElapsedInSeconds() < time_limit_in_seconds;
	}

private:
	std::chrono::high_resolution_clock::time_point starting_time;
	double time_limit_in_seconds;
};

#endif // STOPWATCH_H
