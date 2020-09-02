import time
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors

model = "bert_base"
query_size = 10000


with open('data/{}'.format(model), 'rb') as f:
    base = np.array(pickle.load(f))

with open('data/{}_query_small'.format(model), 'rb') as f:
    query = np.array(pickle.load(f))

query = query[:query_size]
print("n =", len(query), "dimension =", len(query[0]))

# AR
k = 25
start = time.time()
nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(np.array(base))
print("k = {}, model = {} indexing time:".format(k, model), time.time() - start, "seconds")
start = time.time()
distances, indices = nbrs.kneighbors(np.array(query))
print("k = {}, model = {} query time:".format(k, model), time.time() - start, "seconds")
