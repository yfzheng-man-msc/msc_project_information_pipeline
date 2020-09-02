#include "pDCI.h"
#include "utils.h"
#include <queue>
#include <vector>
#include <functional>
#include <utility>
#include <algorithm>
#include <stdlib.h>

// #define DEBUG
#ifdef DEBUG
#include <iostream>
using std::cout;
using std::endl;
#endif

namespace pDCI {

pDCI::pDCI(const Dataset &dataset, int m, int L) : D(dataset), m(m), L(L) {
    n = D.size();
    d = D[0].size();
    u = new Index[m * L];
    T = new Bst[m * L];
    // randomly initialize u
    for (int i = 0, total = m * L; i < total; ++i) {
        u[i] = generate_index(d);
    }
}

pDCI::~pDCI() {
    delete[] u;
    delete[] T;
}

void pDCI::construct() {
    // todo: for --> matrix multiplication
    for (int j = 0; j < m; ++j) {
        for (int l = 0; l < L; ++l) {
            Bst &T_jl = T[j * L + l];
            for (int i = 0; i < n; ++i) {
                // project p^i to u_jl, then insert it to binary search tree
                T_jl.push_back(
                    std::make_pair(project(D[i], u[j * L + l]), i));
            }
            // sort T_jl to support binary search
            std::sort(T_jl.begin(), T_jl.end(),
                      [](const pair<value_type, int> &p1,
                         const pair<value_type, int> &p2) {return p1.first < p2.first; });
        }
    }
}

std::vector<int> pDCI::query(Point query, int k, int k0, int k1, int nc) {
    // k: the number of points to retrieve
    // k0: the number of points to retrieve in each composite index
    // k1: the number of points to visit in each composite index
    // nc: the maximum number of candidates. -1 means INF
    // return: the set of k closet points

    if (k0 == -1) k0 = k;
    if (k1 == -1) k1 = k * n;

    std::priority_queue<_priority_queue_item, vector<_priority_queue_item>, std::greater<_priority_queue_item>> P[L];
    // locate the closest point (not pushed into the queue yet) on each simple
    // index. the closest point of query should be either left[j * L + l] or
    // right[j * L + l]
    std::vector<int> left(m * L);
    std::vector<int> right(m * L);
    // record the projection of the query on each simple index
    std::vector<value_type> projs(m * L);
    for (int l = 0; l < L; ++l) {
        // (distance_to_query, point_id, j)
        auto &P_l = P[l];
        for (int j = 0; j < m; ++j) {
            Bst &T_jl = T[j * L + l];
            value_type proj = project(query, u[j * L + l]);
            int position = binary_search(T_jl, proj);
            // points closet to query goes first. totally mL candidates
            P_l.push(std::make_pair(abs(T_jl[position].first - proj), std::make_pair(T_jl[position].second, j)));
            projs[j * L + l] = proj;
            left[j * L + l] = position - 1;
            right[j * L + l] = position + 1;
        }
    }

    std::vector<int> S;
    // count how many candicates has been selected from each composite index
    std::vector<int> count(L, 0);
    // count the selected times of each point
    std::vector<vector<int>> C(L, vector<int>(n, 0));

    // switch the nest order of k1 and L
    for (int l = 0; l < L; ++l) {

        auto &P_l = P[l];
        
        int selected;
        _priority_queue_item p;
        for (int i = 0; i < k1; ++i) {
            if (P_l.empty() || count[l] >= k0) continue;
            p = P_l.top();
            P_l.pop();
            selected = p.second.first;
            C[l][selected] += 1;
            if (C[l][selected] == m) {
                S.push_back(selected);
                count[l] += 1;

                if (S.size() == nc) goto BYE;
            }
            
            // push the next closest point in T_jl into the queue
            int j = p.second.second;
            int &left_pos = left[j * L + l], &right_pos = right[j * L + l];
            value_type proj = projs[j * L + l];
            Bst &T_jl = T[j * L + l];
            if (left_pos < 0 && right_pos >= n) {
                continue;
            } else if (left_pos < 0 ||
                       ((right_pos < n) && ((proj - T_jl[left_pos].first) >
                                            (T_jl[right_pos].first - proj)))) {
                P_l.push(std::make_pair(T_jl[right_pos].first - proj,
                         std::make_pair(T_jl[right_pos].second, j)));
                ++right_pos;
            } else {
                P_l.push(std::make_pair(proj - T_jl[left_pos].first,
                         std::make_pair(T_jl[left_pos].second, j)));
                --left_pos;
            }
        }
    }

BYE:
    // return k points closet to query in S
    std::vector<std::pair<value_type, int>> order_list;
    for (int index: S) {
        order_list.push_back(std::make_pair(euclidean(query, D[index]), index));
    }
    std::sort(order_list.begin(), order_list.end(),
              [](const std::pair<value_type, int> &p1,
                 const std::pair<value_type, int> &p2) { return p1.first < p2.first; });
    vector<int> rtn;
    for (int i = 0, end = order_list.size(); i < end; ++i) {
        if (rtn.size() == k) break;
        if (std::find(rtn.begin(), rtn.end(), order_list[i].second) == rtn.end()) {
            rtn.push_back(order_list[i].second);
        }
    }
    return rtn;
}

}