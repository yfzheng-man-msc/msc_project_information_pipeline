#include <math.h>

#include <algorithm>
#include <iostream>
#include <utility>
#include <chrono>
#include <fstream>
using namespace std::chrono; 

#include "pDCI.h"
#include "utils.h"
using namespace std;

extern "C" {
  int fvecs_read(const char *fname, int d, int n, float *a);
}

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

pair<double, double> test_2t(int d, int n, int m, int L, int k, int nc,
                             int query_num, int k0 = -1, int k1 = -1, int b = -1) {
  std::cout << "parameters: " << d << " " << n << " " << m << " " << L << " " << k << " " << nc << " " << query_num << " " << k0 << " " << k1 << " " << std::endl;

  std::vector<float> bases(n * d);
  std::vector<float> queries(query_num * d);

  fvecs_read("../data/bert_base_binary", d, n, bases.data());
  fvecs_read("../data/bert_base_query_small_binary", d, query_num, queries.data());

  std::ofstream f("pdci_record", std::fstream::app);
  f << m << " " << L << " " << nc << " ";

  pDCI::Dataset dataset;
  for (int i = 0; i < n; ++i) {
    dataset.push_back(std::vector<float>(bases.begin() + i * d, bases.begin() + (i + 1) * d));
  }

  cout << "start constructing" << endl;
  // measure execution time
  auto start = high_resolution_clock::now(); 

  pDCI::pDCI obj(dataset, m, L);
  obj.construct();
  auto stop = high_resolution_clock::now();
  double construct_time = duration_cast<milliseconds>(stop - start).count();
  cout << "construction time: " << construct_time/1000 << 's' << endl;
  f << construct_time/1000 << "s ";
  double query_time = 0;
  
  double metric = 0;
  std::vector<float> true_radiuses = read_data("../data/ground_truth_K25");
  for (int q = 0; q < query_num; ++q) {
    pDCI::Point query = std::vector<float>(queries.begin() + q * d, queries.begin() + (q + 1) * d);
    auto start = high_resolution_clock::now();
    auto result1 = obj.query(query, k, k0, k1, nc);
    auto stop = high_resolution_clock::now();
    query_time += duration_cast<milliseconds>(stop - start).count();

    if (result1.size() < k) {
      std::cout << "warning: query " << q << " has less than " << k << " candidates\n";
      continue;
    }
    float result_radius = pDCI::euclidean(query, dataset[result1[k-1]]);
    float true_radius = true_radiuses[q];
    metric += result_radius / true_radius;
  }
  metric /= query_num;
  cout << "approximate quality: " << metric << endl;
  cout << "query time: " << query_time/1000 << 's' << endl;
  f << query_time/1000 << "s " << metric << std::endl;
  return std::make_pair(metric, construct_time+query_time);
}


int main(int argc, char* argv[]) {
    int repeat_times = 1;
    int d = 768, k = 25, m = atoi(argv[1]), L = atoi(argv[2]), n = 54617, query_num = 10000;
    int nc = atoi(argv[3]);

    // calculate k0 and k1
    double tmp1 = log(n / k);
    double tmp2 = pow(n / k, 1 - m * log2(d));
    int k0 = (k * (tmp1 > tmp2 ? tmp1 : tmp2) + 1) * 2;
    tmp2 = pow(n / k, 1 - log2(d));
    int k1 = (m * k * (tmp1 > tmp2 ? tmp1 : tmp2) + 1) * 10;
    k0 = 250, k1 = 150000;
    cout << "parameter k0: " << k0 << " k1: " << k1 << endl;

    int b = 1;           // number of bins. for LSH

    double metric = 0;
    double spent_time = 0;
    for (int t = 0; t < repeat_times; ++t) {
        auto result = test_2t(d, n, m, L, k, nc, query_num, k0, k1, b);
        metric += result.first;
        spent_time += result.second;
    }
    metric /= repeat_times;
    double avg_time = spent_time / repeat_times / 1000;
    cout << "final quality: " << metric << endl;
    cout << "average time: " << avg_time << 's' << endl;
}
