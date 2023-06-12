from .labelcorrection import LabelCorrectionModel
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import mlflow

class PolishingLabels(LabelCorrectionModel):
    """ 
    Polishing Labels algorithm

    Reference:
    Nicholson, Bryce, et al. "Label noise correction methods." 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015.

    Attributes
    ----------
    classifier
        Type of classifier to use
    n_folds : int
        Number of folds to use
    """
    def __init__(self, classifier, n_folds):
        super().__init__('PL')
        self.classifier = classifier
        self.n_folds = n_folds

    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
        
        models = []
        for train_index, _ in kf.split(X, y):
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]

            models.append(self.classifier(random_state=42).fit(X_train.values, y_train.values))

        y_corrected = X.apply(
            lambda x: 0 if np.mean([models[i].predict([x.values])[0] for i in range(self.n_folds)]) < 0.5 else 1, 
            axis=1).to_list()
        
        return pd.Series(y_corrected, index=original_index)
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('correction_classifier', self.classifier.__name__)
        mlflow.log_param('n_folds', self.n_folds)