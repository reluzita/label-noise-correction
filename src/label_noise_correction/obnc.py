from .labelcorrection import LabelCorrectionModel
import pandas as pd
from sklearn.ensemble import BaggingClassifier
import mlflow

class OrderingBasedCorrection(LabelCorrectionModel):
    """
    Ordering-Based Correction algorithm

    Reference:
    Feng, Wei, and Samia Boukir. "Class noise removal and correction for image classification using ensemble margin." 2015 IEEE International Conference on Image Processing (ICIP). IEEE, 2015.

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    """
    def __init__(self, m):
        self.m = m

    def calculate_margins(self, X, y, bagging:BaggingClassifier):
        margins = pd.Series(dtype=float)
        for i in X.index:
            preds = [dt.predict(X.loc[i].values.reshape(1, -1))[0] for dt in bagging.estimators_]
            true_y = y.loc[i]

            v_1 = sum(preds)
            v_0 = len(preds) - v_1

            if true_y == 1:
                margins.loc[i] = (v_1 - v_0) / len(preds)
            else:
                margins.loc[i] = (v_0 - v_1) / len(preds)

        return margins

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X, y)
        y_pred = pd.Series(bagging.predict(X), index=y.index)

        margins = self.calculate_margins(X.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)
        index = margins.index[:int(self.m*len(margins))]
        y_corrected.loc[index] = y_pred.loc[index]

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', 'Ordering-Based Correction')
        mlflow.log_param('m', self.m)