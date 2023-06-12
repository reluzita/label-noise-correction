from .labelcorrection import LabelCorrectionModel
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import math
import mlflow

class BayesianEntropy(LabelCorrectionModel):
    """
    Framework for identifying and correcting mislabeled instances

    Reference:
    Sun, Jiang-wen, et al. "Identifying and correcting mislabeled training instances." Future generation communication and networking (FGCN 2007). Vol. 1. IEEE, 2007.

    Attributes
    ----------
    alpha : float
        Noise rate
    n_folds : int
        Number of folds to use
    """
    def __init__(self, alpha, n_folds):
        super().__init__('BE')
        self.alpha = alpha
        self.n_folds = n_folds

    def evaluate(self, X:pd.DataFrame, y:pd.Series):
        kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
        entropy = pd.Series(index=y.index, dtype=float)
        y_pred = pd.Series(index=y.index, dtype=int)

        for train_index, test_index in kf.split(X, y):
            X_train = X.loc[train_index]
            X_test = X.loc[test_index]
            y_train = y.loc[train_index]

            gnb = GaussianNB().fit(X_train, y_train)
            
            y_pred.loc[test_index] = gnb.predict(X_test)

            probs = gnb.predict_proba(X_test)
            entropy.loc[test_index] = [- sum(x * math.log(x, 2) if x != 0 else 0 for x in probs[i]) for i in range(len(probs))]

        return entropy, y_pred
    
    def correct(self, X: pd.DataFrame, y: pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        entropy, y_pred = self.evaluate(X, y)
        threshold = entropy.sort_values(ascending=True, ignore_index=True)[int(self.alpha*len(entropy))]

        y_corrected = y.copy()

        last_changed = []
        it = 0
        while True:
            stop = True
            changed = []

            for i in y_corrected.index:
                if entropy.loc[i] < threshold and y_corrected.loc[i] != y_pred.loc[i]:
                    y_corrected.loc[i] = y_pred.loc[i]
                    stop = False
                    changed.append(i)

            changed.sort()
            if stop or changed == last_changed:
                return pd.Series(y_corrected.values, index=original_index)
            
            if len(changed) == 1 and it > 50:
                return pd.Series(y_corrected.values, index=original_index)

            last_changed = changed
            entropy, y_pred = self.evaluate(X, y_corrected)
            it += 1

        
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('n_folds', self.n_folds)
        mlflow.log_param('alpha', self.alpha)