from .labelcorrection import LabelCorrectionModel
import pandas as pd
import math
import numpy as np
from sklearn.cluster import KMeans
import mlflow

class ClusterBasedCorrection(LabelCorrectionModel):
    """
    Cluster-based correction algorithm

    Reference:
    Nicholson, Bryce, et al. "Label noise correction methods." 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015.

    Attributes
    ----------
    n_iterations : int
        Number of iterations to run
    n_clusters : int
        Number of clusters to use
    """
    def __init__(self, n_iterations, n_clusters):
        super().__init__('CC')
        self.n_iterations = n_iterations
        self.n_clusters = n_clusters

    def calc_weights(self, cluster_labels:pd.Series, label_dist, n_labels):
        d = [cluster_labels.value_counts().loc[l]/len(cluster_labels) if l in cluster_labels.value_counts().index else 0 for l in range(n_labels)]
        u = 1/n_labels
        multiplier = min(math.log(len(cluster_labels), 10), 2)

        return [multiplier * ((d[l] - u)/label_dist[l]) for l in range(n_labels)]

    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        n_labels = len(y.unique())
        label_totals = [y.value_counts().loc[l]/len(y) for l in range(n_labels)]
        ins_weights = np.zeros((X.shape[0], n_labels))

        if len(X) < self.n_clusters:
            print('Number of samples is less than the number of clusters, using half of the samples as the number of clusters')
            self.n_clusters = int(len(X)/2)

        for i in range(1, self.n_iterations+1):
            k = int((i/self.n_iterations) * self.n_clusters) # on the original paper, the number of clusters varies from 2 to half of the number of samples

            if k == 0:
                k = 2

            C = KMeans(n_clusters=k, random_state=42).fit(X)

            clusters = pd.Series(C.labels_, index=X.index)
            cluster_weights = {c: self.calc_weights(y.loc[clusters == c], label_totals, n_labels) for c in range(k)}
            
            for idx in X.index:
                ins_weights[idx] += cluster_weights[C.labels_[idx]]

        y_corrected = [np.argmax(ins_weights[idx]) for idx in X.index]
        return pd.Series(y_corrected, index=original_index)

    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('n_iterations', self.n_iterations)
        mlflow.log_param('n_clusters', self.n_clusters)