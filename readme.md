## An NLP Information Extraction Pipeline

This repository is the implementation and experiment codes of my master project *An NLP Information Extraction Pipeline*. Mainly there are two parts in this repository:

- The Embedding experiments. This part includes fine-tuning BERT models, predicting masked words with BERT models (MLM task), and embedding sentences. The implementation of this part is at file `sy.ipynb`.
- The kNN/ANN experiments. This part includes comparing different kNN/ANN algorithms. The four algorithms compared in this project is the KD-Tree, pDCI, OPQ and GNO-IMI, which correspond to files/directories `kdtree.py`, `pDCI/`, `opq.py`, and `gnoimi/`.

Apart from these files/directories, other files/directories in this repository are:

- `data/`: the directory containing the datasets used in this project.
- `lib/`: third-party libraries used by the implementation.

To initialize datasets and compile C++ programs, run `init.sh`:

```bash
user:path$ ./init.sh
```

The compiled binary files are at `./bin`.

The command line arguments for GNO-IMI are as follows:

```bash
user:path$ ./GNOIMI K index_L fine_L
```

where `K index_L fine_L` are integers. `K` is the number of first-order and second-order centroids, `index_L` is the `L` parameter for the indexing step, `fine_L` is the number of second-order centroids considered at the query step, which are all mentioned in the original paper. Some recommended parameter settings for GNO-IMI are as follows:

```bash
user:path$ ./GNOIMI 16 4 1
user:path$ ./GNOIMI 16 2 2
user:path$ ./GNOIMI 8 2 16
```

The command line arguments for pDCI are as follows:

```bash
user:path$ ./pDCI m L nc
```

where `m L nc` are integers. `m` is the number of simple indices for each composite index. `L` is the number of composite indices. `nc` is the number of candidates for the query step. Some recommended parameter settings for GNO-IMI are as follows:

```bash
user:path$ ./pDCI 5 20 100
user:path$ ./pDCI 5 20 3125
user:path$ ./pDCI 5 20 12500
```

Note that the codes are all running on the specific datasets, which are mentioned in the dissertation. These datasets locate at the `data/` directory.