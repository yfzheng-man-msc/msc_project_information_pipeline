#include <chrono>
#include <memory>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "learn_GNOIMI.h"
extern "C" {
#include <cblas.h>

float kmeans (int d, int n, int k, int niter, 
	      const float * v, int flags, long seed, int redo, 
	      float * centroids, float * dis, 
	      int * assign, int * nassign);

int fvecs_write (const char *fname, int d, int n, const float *vf);

int fvecs_read (const char *fname, int d, int n, float *v);

void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k,
                   const char *transp,
                   float *result);
}

namespace learn_gnoimi {

float euclideanDistance(float *p1, float *p2, int d){
	float sum = 0, tmp;
	for (int i = 0; i < d; ++i){
		tmp = p1[i] - p2[i];
		sum += tmp * tmp;
	}
	return pow(sum, 0.5);
}


void IndexGNOIMI::train(int64_t n, float *train_vecs_in, int l, int learnIterationsCount, int trainThreadChunkSize,
	bool normalize_before_train, std::string outputFilesPrefix, int thread_num){
	clean();
	learned = true;
	args_N = n, args_L = l;
	args_learnIterationsCount = learnIterationsCount;
	args_trainThreadChunkSize = trainThreadChunkSize;
	args_normalize_before_train = normalize_before_train;
	args_outputFilesPrefix = outputFilesPrefix;
	args_threadsCount = thread_num;
	initIntermediateVariables();

	train_vecs = train_vecs_in;

	initializeVocabs();
	int64_t N_t = args_N;
	args_N = args_N / args_threadsCount / args_trainThreadChunkSize
			 * args_trainThreadChunkSize * args_threadsCount;
	std::cout << "args_N(args.n) change from " << N_t << " to " << args_N << std::endl;
	trainAlpha();
}

void IndexGNOIMI::initIntermediateVariables(){
	coarseAssigns = (int*)malloc(args_N * sizeof(int));
	fineAssigns = (int*)malloc(args_N * sizeof(int));
	assigns = (int64_t*)malloc(args_N * sizeof(int64_t));
	alphaNum = (float*)malloc(args_K * args_K * sizeof(float));
	alphaDen = (float*)malloc(args_K * args_K * sizeof(float));
	alpha = (float*)malloc(args_K * args_K * sizeof(float));

	coarseVocab = (float*)malloc(args_D * args_K * sizeof(float));
	fineVocab = (float*)malloc(args_D * args_K * sizeof(float));
	fineVocabNum = (float*)malloc(args_D * args_K * sizeof(float));
	fineVocabDen = (float*)malloc(args_K * sizeof(float));
	coarseVocabNum = (float*)malloc(args_D * args_K * sizeof(float));
	coarseVocabDen = (float*)malloc(args_K * sizeof(float));
	coarseNorms = (float*)malloc(args_K * sizeof(float));
	fineNorms = (float*)malloc(args_K * sizeof(float));
	coarseFineProducts = (float*)malloc(args_K * args_K * sizeof(float));

	errors = (float*)malloc(args_threadsCount * sizeof(float));
	alphaNumerators.resize(args_threadsCount);
	alphaDenominators.resize(args_threadsCount);
	fineVocabNumerators.resize(args_threadsCount);
	fineVocabDenominators.resize(args_threadsCount);
	coarseVocabNumerators.resize(args_threadsCount);
	coarseVocabDenominators.resize(args_threadsCount);
	for(int threadId = 0; threadId < args_threadsCount; ++threadId){
		alphaNumerators[threadId] = (float*)malloc(args_K * args_K * sizeof(float));
		alphaDenominators[threadId] = (float*)malloc(args_K * args_K * sizeof(float));
		fineVocabNumerators[threadId] = (float*)malloc(args_K * args_D * sizeof(float));
		fineVocabDenominators[threadId] = (float*)malloc(args_K * sizeof(float));
		coarseVocabNumerators[threadId] = (float*)malloc(args_K * args_D * sizeof(float));
		coarseVocabDenominators[threadId] = (float*)malloc(args_K * sizeof(float));
	}

	alphaFilename = args_outputFilesPrefix + "alpha.fvecs";
	fineVocabFilename = args_outputFilesPrefix + "fine.fvecs";
	coarseVocabFilename = args_outputFilesPrefix + "coarse.fvecs";

	pointsCount = args_N / args_threadsCount;
	chunksCount = pointsCount / args_trainThreadChunkSize;
}

void IndexGNOIMI::initializeVocabs(){
	int64_t N = args_N;
	if (N > 10000000){
		N = 10000000;
		std::cout << "use only" << N << "training points to initialize vocabs" << std::endl;
	}


	std::unique_ptr<int64_t> closest_id(new int64_t[N]);
	std::unique_ptr<int> closest_id_int(new int[N]);
	std::unique_ptr<float> closest_dis(new float[N]);
	std::unique_ptr<int> nassign(new int[args_K]);

	int64_t *closest_id_ptr = closest_id.get();

	std::cout << "initializing GNOIMI coarse & fine vocabs..." << std::endl;
	float error = 0;

	error = kmeans(args_D, N, args_K, 15, train_vecs, 0x10000 | args_threadsCount, 1234, 1, 
		coarseVocab, closest_dis.get(), closest_id_int.get(), nassign.get());

	for (int64_t i = 0, end = N * args_D; i < end; i += args_D) {
		float *current_vec = train_vecs + i;
		float minDis = 0x7fffffff;
		int64_t minIdx = -1;
		for (int64_t j = 0, end = args_K * args_D; j < end; j += args_D) {
			float *current_center = coarseVocab + j;
			float current_dis = euclideanDistance(current_vec, current_center, args_D);
			if (current_dis < minDis) {
				minDis = current_dis;
				minIdx = j / args_D;
			}
		}
		closest_id_ptr[i/args_D] = minIdx;
	}

	std::vector<float> residuals(args_D * N);
	for (int64_t i = 0; i < N; ++i) {
		for (int64_t j = 0; j < args_D; ++j) {
			int64_t tmp1 = i*args_D+j;
			residuals[tmp1] = train_vecs[tmp1] - coarseVocab[closest_id_ptr[i] * args_D + j];
		}
	}
	
	error = kmeans(args_D, N, args_K, 15, residuals.data(), 0X10000 | args_threadsCount, 1234, 1,
	fineVocab, closest_dis.get(), closest_id_int.get(), nassign.get());
}

void IndexGNOIMI::trainAlpha(){
	for(int i = 0; i < args_K * args_K; ++i){
		alpha[i] = 1.0;
	}
	std::cout << "start learning iterarions..." << std::endl;
	for(int it = 0; it < args_learnIterationsCount; ++it){
		updatePrecompute();
		updateAssigns();
		int threadId, i;

		memset(alphaNum, 0, args_K * args_K * sizeof(float));
		memset(alphaDen, 0, args_K * args_K * sizeof(float));
		#pragma omp parallel for private(threadId) num_threads(args_threadsCount)
		for(threadId = 0; threadId < args_threadsCount; ++threadId){
			computeOptimalAlphaSubset(threadId);
		}
		for(int threadId = 0; threadId < args_threadsCount; ++threadId){
			cblas_saxpy(args_K * args_K, 1, alphaNumerators[threadId], 1, alphaNum, 1);
			cblas_saxpy(args_K * args_K, 1, alphaDenominators[threadId], 1, alphaDen, 1);
		}

		for(i = 0; i < args_K * args_K; ++i){
			alpha[i] = (alphaDen[i] == 0) ? 1.0 : alphaNum[i] / alphaDen[i];
		}

		memset(fineVocabNum, 0, args_K * args_D * sizeof(float));
		memset(fineVocabDen, 0, args_K * sizeof(float));
		#pragma omp parallel for private(threadId) num_threads(args_threadsCount)
		for(threadId = 0; threadId < args_threadsCount; ++threadId){
			computeOptimalFineVocabSubset(threadId);
		}
		for(int threadId = 0; threadId < args_threadsCount; ++threadId){
			cblas_saxpy(args_K * args_D, 1, fineVocabNumerators[threadId], 1, fineVocabNum, 1);
			cblas_saxpy(args_K, 1, fineVocabDenominators[threadId], 1, fineVocabDen, 1);
		}

		for(i = 0; i < args_K * args_D; ++i){
			fineVocab[i] = (fineVocabDen[i / args_D] == 0) ? 0 : fineVocabNum[i] / fineVocabDen[i / args_D];
		}

		memset(coarseVocabNum, 0, args_K * args_D * sizeof(float));
		memset(coarseVocabDen, 0, args_K * sizeof(float));
		#pragma omp parallel for private(threadId) num_threads(args_threadsCount)
		for(threadId = 0; threadId < args_threadsCount; ++threadId){
			computeOptimalCoarseVocabSubset(threadId);
		}
		for(int threadId = 0; threadId < args_threadsCount; ++threadId){
			cblas_saxpy(args_K * args_D, 1, coarseVocabNumerators[threadId], 1, coarseVocabNum, 1);
			cblas_saxpy(args_K, 1, coarseVocabDenominators[threadId], 1, coarseVocabDen, 1);
		}

		for(i = 0; i < args_K * args_D; ++i){
			coarseVocab[i] = (coarseVocabDen[i / args_D] == 0) ? 0 : coarseVocabNum[i] / coarseVocabDen[i / args_D];
		}
	}

	fvecs_write(alphaFilename.c_str(), args_K, args_K, alpha);
	fvecs_write(fineVocabFilename.c_str(), args_D, args_K, fineVocab);
	fvecs_write(coarseVocabFilename.c_str(), args_D, args_K, coarseVocab);
	std::cout << "finish train alpha and write them to file" << std::endl;
}

void IndexGNOIMI::updatePrecompute(){
	for(int k = 0; k < args_K; ++k){
		coarseNorms[k] = cblas_sdot(args_D, coarseVocab + k * args_D, 1, coarseVocab + k * args_D, 1) / 2;
		fineNorms[k] = cblas_sdot(args_D, fineVocab + k * args_D, 1, fineVocab + k * args_D, 1) / 2; 
	}

	fmat_mul_full(fineVocab, coarseVocab, args_K, args_K, args_D, "TN", coarseFineProducts);
}

void IndexGNOIMI::updateAssigns(){
	memset(errors, 0, args_threadsCount * sizeof(float));
	int threadId;
	#pragma omp parallel for private(threadId) num_threads(args_threadsCount)
	for (threadId = 0; threadId < args_threadsCount; ++threadId) {
		computeOptimalAssignsSubset(threadId);
	}
	float totalError = 0.0;
	for(threadId = 0; threadId < args_threadsCount; ++threadId){
		totalError += errors[threadId];
	}
}

void IndexGNOIMI::computeOptimalAssignsSubset(int threadId) {
	int64_t startId = (args_N / args_threadsCount) * threadId;
	float *pointsCoarseTerms = (float*)malloc(args_trainThreadChunkSize * args_K * sizeof(float));
	float *pointsFineTerms = (float*)malloc(args_trainThreadChunkSize * args_K * sizeof(float));
	errors[threadId] = 0.0;

	float* chunkPoints = train_vecs + startId * args_D;
	std::vector<std::pair<float, int>>coarseScores(args_K);
	for(int chunkId = 0; chunkId < chunksCount; ++chunkId, chunkPoints += args_trainThreadChunkSize * args_D){
		fmat_mul_full(coarseVocab, chunkPoints, args_K, args_trainThreadChunkSize, args_D, "TN", pointsCoarseTerms);
		fmat_mul_full(fineVocab, chunkPoints, args_K, args_trainThreadChunkSize, args_D, "TN", pointsFineTerms);

		for(int pointId = 0; pointId < args_trainThreadChunkSize; ++pointId){

			cblas_saxpy(args_K, -1.0, coarseNorms, 1, pointsCoarseTerms + pointId * args_K, 1);
			for(int k = 0; k < args_K; ++k){
				coarseScores[k].first = (-1.0) * pointsCoarseTerms[pointId * args_K + k];
				coarseScores[k].second = k;
			}
			std::sort(coarseScores.begin(), coarseScores.end());
			float currentMinScore = 999999999.0;
			int currentMinCoarseId = -1;
			int currentMinFineId = -1;
			for(int l = 0; l < args_L; ++l){
				int currentCoarseId = coarseScores[l].second;
				float currentCoarseTerm = coarseScores[l].first;

				for(int currentFineId = 0; currentFineId < args_K; ++currentFineId){
					float alphaFactor = alpha[currentCoarseId * args_K + currentFineId];
					float score = currentCoarseTerm + alphaFactor * coarseFineProducts[currentCoarseId * args_K + currentFineId]
					+ (-1.0) * alphaFactor * pointsFineTerms[pointId * args_K +currentFineId]
					+ alphaFactor * alphaFactor * fineNorms[currentFineId];
					if(score < currentMinScore){
						currentMinScore = score;
						currentMinCoarseId = currentCoarseId;
						currentMinFineId = currentFineId;
					}
				}
			}

			coarseAssigns[startId + chunkId * args_trainThreadChunkSize + pointId] = currentMinCoarseId;
			fineAssigns[startId + chunkId * args_trainThreadChunkSize + pointId] = currentMinFineId;
			assigns[startId + chunkId * args_trainThreadChunkSize + pointId] = currentMinCoarseId * args_K + currentMinFineId;
			errors[threadId] += currentMinScore * 2 +1.0;
		}
	}

	free(pointsCoarseTerms);
	free(pointsFineTerms);

}

void IndexGNOIMI::computeOptimalAlphaSubset(int threadId){
	memset(alphaNumerators[threadId], 0, args_K * args_K * sizeof(float));
	memset(alphaDenominators[threadId], 0, args_K * args_K * sizeof(float));
	int64_t startId = (args_N / args_threadsCount) * threadId;
	float* residual = (float*)malloc(args_D * sizeof(float));
	float* chunkPoints = train_vecs + startId * args_D;
	for(int chunkId = 0; chunkId < chunksCount; ++chunkId, chunkPoints += args_trainThreadChunkSize * args_D){
		for(int pointId = 0; pointId < args_trainThreadChunkSize; ++pointId){
			int coarseAssign = coarseAssigns[startId + chunkId * args_trainThreadChunkSize + pointId];
			int fineAssign = fineAssigns[startId + chunkId * args_trainThreadChunkSize + pointId];
			memcpy(residual, chunkPoints + pointId * args_D, args_D * sizeof(float));
			cblas_saxpy(args_D, -1.0, coarseVocab + coarseAssign * args_D, 1, residual, 1);
			alphaNumerators[threadId][coarseAssign * args_K + fineAssign] += 
			cblas_sdot(args_D, residual, 1, fineVocab + fineAssign * args_D, 1);
			alphaDenominators[threadId][coarseAssign * args_K + fineAssign] += fineNorms[fineAssign] *2;
		}
	}
	free(residual);
}

void IndexGNOIMI::computeOptimalFineVocabSubset(int threadId){
	memset(fineVocabNumerators[threadId], 0, args_K * args_D * sizeof(float));
	memset(fineVocabDenominators[threadId], 0, args_K * sizeof(float));
	int64_t startId = (args_N / args_threadsCount) * threadId;
	float* residual = (float*)malloc(args_D * sizeof(float));
	float* chunkPoints = train_vecs + startId * args_D;
	for(int chunkId = 0; chunkId < chunksCount; ++chunkId, chunkPoints += args_trainThreadChunkSize * args_D){
		for(int pointId = 0; pointId < args_trainThreadChunkSize; ++pointId){
			int coarseAssign = coarseAssigns[startId + chunkId * args_trainThreadChunkSize + pointId];
			int fineAssign = fineAssigns[startId + chunkId * args_trainThreadChunkSize + pointId];
			float alphaFactor = alpha[coarseAssign * args_K + fineAssign];
			memcpy(residual, chunkPoints + pointId * args_D, args_D * sizeof(float));
			cblas_saxpy(args_D, -1.0, coarseVocab + coarseAssign * args_D, 1, residual, 1);
			cblas_saxpy(args_D, alphaFactor, residual, 1, fineVocabNumerators[threadId] + fineAssign * args_D, 1);
			fineVocabDenominators[threadId][fineAssign] += alphaFactor * alphaFactor;
		}
	}
	free(residual);
} 

void IndexGNOIMI::computeOptimalCoarseVocabSubset(int threadId){
	memset(coarseVocabNumerators[threadId], 0, args_K * args_D * sizeof(float));
	memset(coarseVocabDenominators[threadId], 0, args_K * sizeof(float));
	int64_t startId = (args_N / args_threadsCount) * threadId;
	float* residual = (float*)malloc(args_D * sizeof(float));
	float* chunkPoints = train_vecs + startId * args_D;
	for(int chunkId = 0; chunkId < chunksCount; ++chunkId, chunkPoints += args_trainThreadChunkSize * args_D){
		for(int pointId = 0; pointId < args_trainThreadChunkSize; ++pointId){
			int coarseAssign = coarseAssigns[startId + chunkId * args_trainThreadChunkSize + pointId];
			int fineAssign = fineAssigns[startId + chunkId * args_trainThreadChunkSize + pointId];
			float alphaFactor = alpha[coarseAssign * args_K + fineAssign];
			memcpy(residual, chunkPoints + pointId * args_D, args_D * sizeof(float));
			cblas_saxpy(args_D, -1.0 * alphaFactor, fineVocab + fineAssign * args_D, 1, residual, 1);
			cblas_saxpy(args_D, 1, residual, 1, coarseVocabNumerators[threadId] + coarseAssign * args_D, 1);
			coarseVocabDenominators[threadId][coarseAssign] += 1.0;
		}
	}
	free(residual);
}


std::vector<std::vector<int64_t>>
IndexGNOIMI::query(int64_t query_num, float *query_vecs, int coarse_L, int fine_L, int topK){
	if (IVF_ID.size() <= 0) {
		std::cout << "cannot querying before indexing" << std::endl;
		exit(-1);
	}
	std::cout << "querying..." << std::endl;

	std::vector<std::vector<int64_t>> result(query_num, std::vector<int64_t>(topK));

	int batch_size = 10000;
	int batch_num = (query_num + batch_size - 1) / batch_size;
	int64_t start = 0;
	for(int b = 0; b < batch_num; ++b, start += batch_size){
		int current_batch_size = (b == batch_num - 1) ? (query_num - start) : batch_size;

		std::vector<std::vector<std::pair<int, int>>> topCells = queryTopCells(current_batch_size, query_vecs + start * args_D, coarse_L, fine_L);

		for(int pointId = 0; pointId < current_batch_size; ++pointId){
			std::vector<std::pair<float, int64_t>> distances;
			for (int i = 0; i < fine_L; ++i) {
				int64_t cellId = topCells[pointId][i].first * args_K + topCells[pointId][i].second;
				for (int j = 0; j < IVF_ID[cellId].size(); ++j) {
					distances.push_back({euclideanDistance(query_vecs + args_D * (start + pointId), train_vecs + args_D * IVF_ID[cellId][j], args_D),
										 IVF_ID[cellId][j]});
				}
			}
			
			int end = topK < distances.size() ? topK : distances.size();
			std::partial_sort(distances.begin(), distances.begin() + end, distances.end());
			for(int i = 0; i < end; ++i){
				result[start + pointId][i] = distances[i].second;
			}
		}
	}
	return result;
}

void IndexGNOIMI::load(std::string inputFilesPrefix){
	clean();
	learned = true;
	args_N = 1, args_threadsCount = 1, args_trainThreadChunkSize = 1;
	args_outputFilesPrefix = inputFilesPrefix;
	initIntermediateVariables();

	std::cout << "loading GNOIMI: " << alphaFilename << "," << coarseVocabFilename << "," << fineVocabFilename << std::endl;
	if (fvecs_read(alphaFilename.c_str(), args_K, args_K, alpha) != args_K) {
		std::cout << "error when reading " << alphaFilename << std::endl;
		exit(-1);
	}
	if (fvecs_read(fineVocabFilename.c_str(), args_D, args_K, fineVocab) != args_K) {
		std::cout << "error when reading " << fineVocabFilename << std::endl;
		exit(-1);
	}
	if (fvecs_read(coarseVocabFilename.c_str(), args_D, args_K, coarseVocab) != args_K) {
		std::cout << "error when reading " << coarseVocabFilename << std::endl;
		exit(-1);
	}
}

void IndexGNOIMI::add(int64_t base_num, float *base_vecs, int index_L, int thread_num){
	std::cout << "GNOIMI start indexing..." << std::endl;
	args_N = base_num;
	train_vecs = base_vecs;
	free(assigns);
	assigns = (int64_t*)malloc(args_N * sizeof(int64_t));

	updatePrecompute();

	int batch_size = 10000;
	int64_t batch_num = (base_num + batch_size - 1) / batch_size;
	int64_t i;
	# pragma omp parallel for private(i) num_threads(thread_num)
	for(i = 0; i < batch_num; ++i){
		int64_t start = i * batch_size;
		int current_batch_size = (i == batch_num -1) ? (base_num - start) : batch_size;

		std::vector<std::vector<std::pair<int, int>>> topCells = queryTopCells(current_batch_size, train_vecs + start * args_D, index_L, 1);

		for(int pointId = 0; pointId < current_batch_size; ++pointId){
			assigns[pointId + start] = topCells[pointId][0].first * args_K + topCells[pointId][0].second;
		}
	}

	IVF_ID = std::vector<std::vector<int64_t>>(args_K * args_K, std::vector<int64_t>());
	IVF_strip = std::vector<int64_t>(args_K * args_K, 0);

	for(int64_t i = 0; i < args_N; ++i){
		IVF_ID[assigns[i]].push_back(i);
	}

	for (int64_t i = 1; i < IVF_strip.size(); ++i){
		IVF_strip[i] = IVF_strip[i-1] + IVF_ID[i-1].size();
	}
}

std::vector<std::vector<std::pair<int, int>>>
IndexGNOIMI::queryTopCells(int query_num, float *query_vecs, int coarse_L, int fine_L) {
	std::vector<std::vector<std::pair<int, int>>> result(query_num, std::vector<std::pair<int, int>>(fine_L));

	float* pointsCoarseTerms = (float*)malloc(query_num * args_K * sizeof(float));
	float* pointsFineTerms = (float*)malloc(query_num * args_K * sizeof(float));
	fmat_mul_full(coarseVocab, query_vecs, args_K, query_num, args_D, "TN", pointsCoarseTerms);
	fmat_mul_full(fineVocab, query_vecs, args_K, query_num, args_D, "TN", pointsFineTerms);

	std::vector<std::pair<float, int>> coarseScores(args_K);
	for(int pointId = 0; pointId < query_num; ++pointId){
		for(int k = 0; k < args_K; ++k){
			coarseScores[k].first = coarseNorms[k] - pointsCoarseTerms[pointId * args_K + k];
			coarseScores[k].second = k;
		}
		std::partial_sort(coarseScores.begin(), coarseScores.begin() + coarse_L, coarseScores.end());

		std::vector<std::pair<float, std::pair<int, int>>> fineScores;

		float currentMinScore = 999999999.0;
		int currentMinCoarseId = -1;
		int currentMinFineId = -1;

		for(int l = 0; l < coarse_L; ++l){
			int currentCoarseId = coarseScores[l].second;
			float currentCoarseTerm = coarseScores[l].first;
			for(int currentFineId = 0; currentFineId < args_K; ++currentFineId){
				int64_t cellId = currentCoarseId * args_K + currentFineId;
				float alphaFactor = alpha[cellId];
				float score = currentCoarseTerm + alphaFactor * coarseFineProducts[cellId]
				+ (-1.0) * alphaFactor * pointsFineTerms[pointId * args_K +currentFineId]
				+ alphaFactor * alphaFactor * fineNorms[currentFineId];

				if(fine_L == 1){
					if(score < currentMinScore){
						currentMinScore = score;
						currentMinCoarseId = currentCoarseId;
						currentMinFineId = currentFineId;
					}
				}
				else{
					fineScores.push_back({score, {currentCoarseId, currentFineId}});
				}
			}
		}

		if (fine_L == 1){
			result[pointId][0].first = currentMinCoarseId;
			result[pointId][0].second = currentMinFineId;
		}
		else{
			std::partial_sort(fineScores.begin(), fineScores.begin() +fine_L, fineScores.end());
			for (int i = 0; i < fine_L; ++i){
				result[pointId][i].first = fineScores[i].second.first;
				result[pointId][i].second = fineScores[i].second.second;
			}
		}
	}

	free(pointsCoarseTerms);
	free(pointsFineTerms);

	return result;
}

}  // namespace learn_gnoimi

std::vector<float> read_data(std::string filename) {
	std::ifstream f(filename);
	if (!f.is_open()) {
		std::cout << "fail to open file " << filename << std::endl;
		exit(-1);
	}
	std::vector<float> data;
	while (!f.eof()) {
		float v;
		f >> v;
		data.push_back(v);
	}
	return data;
}

int main(int argc, char **argv) {
	int dimension = 768, query_size = 10000, base_size = 54617;
	int K = atoi(argv[1]);
	int L = 8, learnIterationsCount = 20, trainThreadChunksize = 100, thread_num = 8;
	bool normalize_before_train = true;
	int index_L = atoi(argv[2]), coarse_L = atoi(argv[2]), fine_L = atoi(argv[3]), topK = 25;

	std::cout << "parameters: " << K << " " << index_L << " " << fine_L << std::endl;
	std::ofstream record("record", std::fstream::app);
	record << K << " " << index_L << " " << fine_L << " ";

	learn_gnoimi::IndexGNOIMI index(dimension, K);
	std::vector<float> base(base_size * dimension);
	std::vector<float> query(query_size * dimension);

	fvecs_read("../data/bert_base_binary", dimension, base_size, base.data());
	fvecs_read("../data/bert_base_query_small_binary", dimension, query_size, query.data());

	std::cout << "load data: base size = " << (base.size()/dimension) << ", query size = " << (query.size()/dimension) << std::endl;

	std::chrono::steady_clock::time_point begin, end;
	int64_t cost_time;
	begin = std::chrono::steady_clock::now();
	index.train(base_size, base.data(), L, learnIterationsCount, trainThreadChunksize, normalize_before_train, "../dumps/", thread_num);
	index.add(base_size, base.data(), index_L, thread_num);
	end = std::chrono::steady_clock::now();
	cost_time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
	std::cout << "indexing time = " << ((float) cost_time / 1000000) << "s" << std::endl;
	record << ((float) cost_time / 1000000) << "s ";

	begin = std::chrono::steady_clock::now();
	std::vector<std::vector<int64_t>> results = index.query(query_size, query.data(), coarse_L, fine_L, topK);
	end = std::chrono::steady_clock::now();
	cost_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	std::cout << "query time = " << ((float)cost_time / 1000000) << "s" << std::endl;
	record << ((float)cost_time / 1000000) << "s ";

	float AR = 0.0;
	std::vector<float> true_radiuses = read_data("../data/ground_truth_K25");
	for (int i = 0; i < query_size; ++i) {
		if (results[i].size() < topK) {
			std::cout << "warning: query " << i << " has less than " << topK << " candidates\n";
			continue;
		}
		int64_t radius_idx = results[i].back();
		float approximate_radius = learn_gnoimi::euclideanDistance(base.data() + radius_idx * dimension, query.data() + i * dimension, dimension);
		AR += approximate_radius / true_radiuses[i];
	}
	AR /= query_size;
	std::cout << "AR score = " << AR << std::endl;
	record << AR << std::endl;
}
















