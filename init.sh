rm -f data/bert_base
rm -f data/bert_base_binary
cat data/bert_base_split_* > data/bert_base
cat data/bert_base_binary_split_* > data/bert_base_binary

mkdir -p dumps
mkdir -p bin
g++ gnoimi/learn_GNOIMI.cc lib/libyael.a -lblas -llapack -fopenmp -o bin/GNOIMI -g
g++ pDCI/main.cpp pDCI/pDCI.cpp pDCI/utils.cpp lib/libyael.a -lblas -llapack -fopenmp -o bin/pDCI -g
