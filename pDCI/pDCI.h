#ifndef PDCI_PDCI_H_
#define PDCI_PDCI_H_

#include "utils.h"

namespace pDCI {

class pDCI {
   public:
    pDCI(const Dataset&, int, int);
    ~pDCI();
    void construct();
    std::vector<int> query(Point, int, int = -1, int = -1, int = -1);

   private:
    int m;  // the number of simple indices
    int L;  // the number of composite indices
    int d;  // dimension of vector space
    int n;  // the number of sample points
    Index *u;  // mL random unit vectors (as indices)
    Bst *T;  // mL empty binary search trees
    const Dataset &D;     // dataset

    // for priority queue
    typedef std::pair<value_type, std::pair<int, int>> _priority_queue_item;
};

}
#endif  // PDCI_PDCI_H_