from .labelcorrection import LabelCorrectionModel
import pandas as pd
import random
import numpy as np
from sklearn.cluster import KMeans
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class HybridLabelNoiseCorrection(LabelCorrectionModel):
    """ 
    Hybrid Label Noise Correction algorithm

    Reference:
    Xu, Jiwei, Yun Yang, and Po Yang. "Hybrid label noise correction algorithm for medical auxiliary diagnosis." 2020 IEEE 18th International Conference on Industrial Informatics (INDIN). Vol. 1. IEEE, 2020.

    Attributes
    ----------
    n_clusters : int
        Number of clusters to use in KMeans clustering
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        data = X.copy()
        data['y'] = y

        if len(data) < self.n_clusters:
            print('Number of samples is less than the number of clusters, using half of the samples as the number of clusters')
            self.n_clusters = int(len(data)/2)

        C = KMeans(n_clusters=self.n_clusters, random_state=0).fit(data)
        clusters = pd.Series(C.labels_, index=X.index)
        cluster_labels = [1 if x[-1] > 0.5 else 0 for x in C.cluster_centers_]    

        high_conf = []
        low_conf = []

        for i in data.index:
            if cluster_labels[clusters.loc[i]] == y.loc[i]:
                high_conf.append(i)
            else:
                low_conf.append(i)

        y_corrected = y.copy()

        while len(low_conf) > self.n_clusters:
            # SSK-Means
            seed_set = data.loc[high_conf]

            C = KMeans(n_clusters=self.n_clusters, random_state=0).fit(seed_set)
            cluster_labels = [1 if x[-1] > 0.5 else 0 for x in C.cluster_centers_] 
            centers = np.array([x[:-1] for x in C.cluster_centers_])

            C_ss = KMeans(n_clusters=self.n_clusters, random_state=0, init=centers, n_init=1).fit(X.loc[low_conf])
            y_pred_sskmeans = [cluster_labels[x] for x in C_ss.labels_]

            # Co-training
            # should the feature sets be randomized each iteration?
            features_dt = random.sample(list(X.columns), len(X.columns)//2)
            features_svm = [col for col in X.columns if col not in features_dt]


            if y_corrected.loc[high_conf].unique().shape[0] == 1:
                print('All high confidence labels are the same, co-training will return the same label')
                y_pred_dt = [y_corrected.loc[high_conf].unique()[0]] * len(low_conf)
                y_pred_svm = [y_corrected.loc[high_conf].unique()[0]] * len(low_conf)
            else:      
                dt = DecisionTreeClassifier(random_state=42).fit(X.loc[high_conf, features_dt], y_corrected.loc[high_conf])
                svm = SVC(random_state=42).fit(X.loc[high_conf, features_svm], y_corrected.loc[high_conf])

                y_pred_dt = dt.predict(X.loc[low_conf, features_dt])
                y_pred_svm = svm.predict(X.loc[low_conf, features_svm])

            # Correct labels if classifiers agree, else send back to low confidence data
            for i in range(len(low_conf)):
                if y_pred_dt[i] == y_pred_svm[i] and y_pred_dt[i] == y_pred_sskmeans[i]:
                    high_conf.append(low_conf[i])
                    y_corrected.loc[low_conf[i]] = y_pred_dt[i]
                    
            low_conf = [x for x in low_conf if x not in high_conf]

        return pd.Series(y_corrected.values, index=original_index)
    
    def log_params(self):
        mlflow.log_param('correction_alg', 'Hybrid Label Noise Correction')
        mlflow.log_param('n_clusters', self.n_clusters)