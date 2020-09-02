#ifndef PDCI_UTILS_H_
#define PDCI_UTILS_H_

#include <random>
#include <algorithm>
#include <utility>
#include <vector>
#include <chrono>
#include <math.h>
using std::vector;
using std::pair;

namespace pDCI {

using value_type = float;
typedef vector<value_type> Point;           // data point
typedef vector<Point> Dataset;              // dataset
typedef vector<value_type> Index;           // index
typedef pair<value_type, int> Titem;
typedef vector<Titem> Bst;  // binary search tree

static Index generate_index(int d) {
    // generate a random unit vector (as index)
    // https://stackoverflow.com/questions/21516575/fill-a-vector-with-random-numbers-c
    // We use static in order to instantiate the random engine
    // and the distribution once only.
    // It may provoke some thread-safety issues.
    // set the upper and lower bound of distribution to 1 and 2 to avoid
    // underflow
    static std::uniform_real_distribution<value_type> distribution(1, 2);
    static std::default_random_engine generator(
        std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<value_type> data(d);
    std::generate(data.begin(), data.end(),
                  []() { return distribution(generator); });

    // normalize
    value_type sum = 0;
    for (value_type &e : data) sum += e * e;
    sum = sqrt(sum);
    for (value_type &e : data) e /= sum;

    return data;
}

value_type project(const Point &pi, const Index &u_jl);

int binary_search(const Bst &T_jl, value_type proj);

value_type euclidean(const Point &p1, const Point &p2);

}

#endif  // PDCI_UTILS_H_
