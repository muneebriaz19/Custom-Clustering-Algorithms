# Foundations of Data Mining - Practical Task 1
# Version 2.1 (2024-10-27)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to related scikit-learn classes.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array


class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
        the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
        and/or encapsulate the necessary mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        # Input validation:
        X = check_array(X, accept_sparse='csr')

        np.random.seed(self.random_state)
        self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            new_centers = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.cluster_centers_, new_centers):
                break
            self.cluster_centers_ = new_centers

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
    
class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
        long as it does this, you may change the content of this method completely and/or encapsulate the necessary
        mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        X = check_array(X, accept_sparse='csr')

        X = X.astype(np.float32)
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, 0)
        cluster_label = 0

        for i in range(n_samples):
            if self.labels_[i] != 0:
                continue
            neighbors = self._find_neighbors_within_radius(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                cluster_label += 1
                self.labels_[i] = cluster_label
                self._expand_cluster(X, i, neighbors, cluster_label)

        return self
    
    def _find_neighbors_within_radius(self, X, point_idx):
        if self.metric == 'euclidean':
            distances = np.linalg.norm(X - X[point_idx], axis=1)
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(X - X[point_idx]), axis=1)
        elif self.metric == 'chebyshev':
            distances = np.max(np.abs(X - X[point_idx]), axis=1)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_label):
        seeds = list(neighbors)

        while seeds:
            current_object = seeds.pop(0)
            if self.labels_[current_object] == 0:
                self.labels_[current_object] = cluster_label
            current_neighbors = self._find_neighbors_within_radius(X, current_object)
            if len(current_neighbors) >= self.min_samples:
                for neighbor_index in current_neighbors:
                    if self.labels_[neighbor_index] in [-1, 0]:
                        seeds.append(neighbor_index)
                        self.labels_[neighbor_index] = cluster_label

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_


