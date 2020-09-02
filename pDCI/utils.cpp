#include "utils.h"
#include <math.h>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>
using std::pair;
using std::vector;

namespace pDCI {
    
value_type project(const Point &pi, const Index &u_jl) {
    value_type proj = 0;
    for (int d_ = 0; d_ < pi.size(); ++d_) {
        proj += pi[d_] * u_jl[d_];
    }
    // since projections are on a line, only one dim is needed
    // return proj * u_jl[0];
    return proj;
}

int binary_search(const Bst &T_jl, value_type proj) {
    int left = 0, right = T_jl.size() - 1;
    int mid;
    while (left <= right) {
        mid = (left + right) / 2;
        if (T_jl[mid].first == proj)
            return mid;
        else if (T_jl[mid].first < proj)
            left = mid + 1;
        else
            right = mid - 1;
    }
    // T_jl[left] is the first point greater than proj
    if (left == 0) return left;
    if (left == T_jl.size()) return right;
    return ((T_jl[left].first - proj) < (proj - T_jl[right].first)) ? left : right;
}

value_type euclidean(const Point &p1, const Point &p2) {
    value_type sum = 0;
    for (int i = 0; i < p1.size(); ++i) {
        double tmp = p1[i] - p2[i];
        sum += tmp * tmp;
    }
    sum = sqrt(sum);
    return sum;
}

}  // namespace pDCI
