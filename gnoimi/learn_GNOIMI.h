#ifndef LEARN_GNOIMI_H_
#define LEARN_GNOIMI_H_

#include <string>
#include <vector>
#include <memory>
#include <stdio.h>
#include <string.h>

using std::string;
using std::vector;

namespace learn_gnoimi {


struct IndexGNOIMI {
    IndexGNOIMI(int D, int K): args_D(D), args_K(K), learned(false) {}

    ~IndexGNOIMI() {
        clean();
    }

    void train(int64_t n, float *train_vecs, int l, int learnIterationsCount, int trainThreadChunksize, bool normalize_before_train,
               std::string outputFilesPrefix, int thread_num);
    
    void load(std::string inputFilesPrefix);

    std::vector<std::vector<int64_t>> query(int64_t query_num, float *query_vecs, int coarse_L, int fine_L, int topK);

    void add(int64_t base_num, float *base_vecs, int index_L, int thread_num);

    // void computeResiduals();

    std::vector<std::vector<std::pair<int, int>>>
    queryTopCells(int query_num, float*query_vecs, int coarse_L, int fine_L);


    vector<vector<int64_t>> IVF_ID;
    vector<int64_t> IVF_strip;

    void initializeVocabs();
    void initIntermediateVariables();
    void trainAlpha();
    void computeOptimalAssignsSubset(int threadId);
    void computeOptimalAlphaSubset(int threadId);
    void computeOptimalFineVocabSubset(int threadId);
    void computeOptimalCoarseVocabSubset(int threadId);
    void updatePrecompute();
    void updateAssigns();

    void clean() {
        if (learned) {
            free(coarseAssigns);
            free(fineAssigns);
            free(alphaNum);
            free(alphaDen);
            free(alpha);
            free(coarseVocab);
            free(coarseVocabNum);
            free(coarseVocabDen);
            free(fineVocab);
            free(fineVocabNum);
            free(fineVocabDen);
            free(coarseNorms);
            free(fineNorms);
            free(coarseFineProducts);
            free(errors);
        }
    }


    int64_t args_K;
    int args_D;
    int64_t args_N;
    int args_L;
    int args_threadsCount;
    int args_learnIterationsCount;
    int args_trainThreadChunkSize;
    bool args_normalize_before_train;
    string args_outputFilesPrefix;

    float *train_vecs;
    int *coarseAssigns;
    int *fineAssigns;
    int64_t *assigns;
    float *alphaNum, *alphaDen, *alpha;

    float *coarseVocab, *fineVocab, *coarseVocabNum, *fineVocabNum, *coarseVocabDen, *fineVocabDen;

    vector<float*> alphaNumerators, alphaDenominators, fineVocabNumerators, fineVocabDenominators, coarseVocabNumerators, coarseVocabDenominators;

    float *coarseNorms, *fineNorms, *coarseFineProducts;

    float *errors;
    string alphaFilename, fineVocabFilename, coarseVocabFilename;

    int pointsCount, chunksCount;
    bool learned;
};


}  // namespace learn_gnoimi

#endif  // LEARN_GNOIMI_H_