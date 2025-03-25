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


