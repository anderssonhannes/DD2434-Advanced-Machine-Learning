import numpy  as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from math import dist
import matplotlib.pyplot as plt

N = 1000     
D = 20
K = 2

# rotate and uncenter the data
X = (2 * np.random.rand(D, D)) @ np.random.randn(D, N) + np.arange(D)[:, None]

# Generate data and then shift its center to [0, 1, 2, 3]
uncentered_data = np.random.randn(4, N) + np.arange(4)[:, None]

# build the magical 1/Ns matrix.
averaging_ones = (np.ones((N, N)))/N

# just show the first few rows
print("originally shifted by ",  np.arange(4))
print("averages found\n", (uncentered_data @ averaging_ones)[:, :6])

centering_matrix = lambda n: np.identity(n) -(np.ones((n, 1)) @ np.ones((1, n)))/n
centered_X = X @ centering_matrix(N)

# Do singular value decomposition
u, s, vh = np.linalg.svd(centered_X)
# Take the top K eigenvalues (np.linalg.svd orders eigenvalues)
pc_scores_from_X = np.diag(s[:K]) @ vh[:K]

distance_matrix = euclidean_distances(X.T)**2

distance_matrix_from_centered = euclidean_distances(centered_X.T)**2
assert np.allclose(distance_matrix, distance_matrix_from_centered)

actual_gram = centered_X.T @ centered_X

uncentered_gram = X.T @ X

centered_gram = centering_matrix(N).T @ uncentered_gram @ centering_matrix(N)

assert np.allclose(actual_gram, centered_gram)

w, v = np.linalg.eig(centered_gram)
pc_score_from_gram_matrix = np.diag(np.sqrt(w[:K])) @ v.T[:K]

gram_from_dist = -(centering_matrix(N) @ distance_matrix @ centering_matrix(N))/2

assert np.allclose(gram_from_dist, centered_gram)




def MDS(distance_matrix, K):
    # Double checking that the matrix is the right size
    N = distance_matrix.shape[0]
    assert distance_matrix.shape[0] == distance_matrix.shape[1], 'dist should be a square matrix, but it\'s {}x{}'.format(dist.shape)
    
    # Compute the Gram matrix from the distance_matrix
    gram_from_dist = -(centering_matrix(N) @ distance_matrix @ centering_matrix(N))/2

    # Compute the PC scores from Gram matrix
    w, v = np.linalg.eig(gram_from_dist)
    # Double check the eigenvalues are positive. While they will be for 
    # actual distance matrices, this becomes a problem if we extend this
    # to other types of symmetric matrices that may not be positive semi-definite.
    assert np.all(w[:K] > 0)
    proj = np.diag(np.sqrt(w[:K])) @ v.T[:K]
    
    return proj

pc_scores_from_distance_matrix = MDS(distance_matrix, K)

dataset_name = "usca312"

import requests

def get_data_from_fsu(url):
    response = requests.get(url)
    response_lines = response.text.split('\n')
    for i, line in enumerate(response_lines):
        if not line.startswith('#'):
            break
    return response_lines[i:]

url = "https://people.sc.fsu.edu/~jburkardt/datasets/cities/"

raw_city_labels_data = get_data_from_fsu("{}{}_name.txt".format(url, dataset_name))
raw_city_dists_data = get_data_from_fsu("{}{}_dist.txt".format(url, dataset_name))

# the last element is a blank character
assert raw_city_labels_data[-1] == ''
city_labels = raw_city_labels_data[:-1]

# The txt file splits long rows up, and it's a tiny bit easier to
# just make a big long list and reshape it
raw_city_dists_list = sum([
    list(map(int, line.split()))
    for line in raw_city_dists_data
], [])

city_dist_matrix = np.array(raw_city_dists_list).reshape(
    len(city_labels), 
    len(city_labels)
)

D = np.load('D.npy')

# locations = MDS(city_dist_matrix, K).T
locations = MDS(D, K).T


for i, cord in enumerate(locations):
    txt = city_labels[i][:2]
    color = 'red'
    plt.plot(cord[0],cord[1],marker='.',color=color)
    plt.text(cord[0],cord[1],txt)

plt.show()

print(locations)