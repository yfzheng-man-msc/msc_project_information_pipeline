import faiss
import time
import pickle
import numpy as np

model = "bert_base"

with open('data/{}'.format(model), 'rb') as f:
    base = np.array(pickle.load(f), dtype=np.float32)

with open('data/{}_query_small'.format(model), 'rb') as f:
    query = np.array(pickle.load(f), dtype=np.float32)

d = len(query[0])
query_size = len(query)
print("base_size =", len(base), "query_size =", query_size, "dimension =", d)

m = 1
training = True
if training:
    index = faiss.index_factory(d, "OPQ{},PQ{}x8".format(m, m))
    start = time.time()
    index.train(base)
    index.add(base)
    print("model = {} indexing time:".format(
        model), time.time() - start, "seconds")
    faiss.write_index(index, "dumps/opq{}".format(m))
else:
    index = faiss.read_index("dumps/opq{}".format(m))

# approximate ratio
k = 25
true_radiuss = []
with open('ground_truth_dis', 'r') as f:
    count = 0
    for line in f:
        true_radiuss.append(float(line.strip().split()[k-1]))
        count += 1
        if count == query_size:
            break

start = time.time()
D, I = index.search(query, k)
print("k = {}, model = {} query time:".format(
    k, model), time.time() - start, "seconds")
approximate_ratio = 0.0
for i, index in enumerate(I):
    result_radius = np.linalg.norm(query[i] - base[index[-1]])
    true_radius = true_radiuss[i]
    approximate_ratio += result_radius / true_radius
approximate_ratio /= len(query)
print("k =", k, "approximate ratio:", approximate_ratio)
