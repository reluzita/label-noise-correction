from .obnc import OrderingBasedCorrection
import pandas as pd
from sklearn.ensemble import BaggingClassifier
import mlflow
import numpy as np

class FairOBNCRemoveSensitive(OrderingBasedCorrection):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC-rs)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    """
    def __init__(self, m, sensitive_attr):
        super().__init__('Fair-OBNC-rs', m)
        self.sensitive_attr = sensitive_attr

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()
        X_fair = X.drop(columns=self.sensitive_attr)

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X_fair, y)
        y_pred = pd.Series(bagging.predict(X_fair), index=y.index)

        margins = self.calculate_margins(X_fair.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)
        index = margins.index[:int(self.m*len(margins))]
        y_corrected.loc[index] = y_pred.loc[index]

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('m', self.m)
        mlflow.log_param('sensitive_attr', self.sensitive_attr)

class FairOBNCOptimizeDemographicParity(OrderingBasedCorrection):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC-dp)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    prob : float
        Probability of correcting a label that does not contribute to balancing label distribution across sensitive groups
    """
    def __init__(self, m:float, sensitive_attr:str):
        super().__init__('Fair-OBNC-dp', m)
        self.sensitive_attr = sensitive_attr

    def dem_par_diff(self, X, y, attr):
        p_y1_g1 = len(y.loc[(X[attr] == 1) & (y == 1)]) / len(y.loc[X[attr] == 1])
        p_y1_g0 = len(y.loc[(X[attr] == 0) & (y == 1)]) / len(y.loc[X[attr] == 0])

        return p_y1_g1 - p_y1_g0
    
    def fair_correction(self, X, y, y_pred, margins):
        y_corrected = y.copy()

        n = int(self.m*len(margins))
        corrected = 0

        for i in margins.index:
            dem_par = self.dem_par_diff(X, y, self.sensitive_attr)
            if X.loc[i, self.sensitive_attr] == 0 and y.loc[i] == int(dem_par < 0):
                y_corrected.loc[i] = y_pred.loc[i]
                corrected += 1
            elif X.loc[i, self.sensitive_attr] == 1 and y.loc[i] == int(dem_par > 0):
                y_corrected.loc[i] = y_pred.loc[i]
                corrected += 1
            elif dem_par == 0:
                y_corrected.loc[i] = y_pred.loc[i]
                corrected += 1
            
            if corrected == n:
                break

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X, y)
        y_pred = pd.Series(bagging.predict(X), index=y.index)

        margins = self.calculate_margins(X.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)

        y_corrected = self.fair_correction(X, y, y_pred, margins)

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('m', self.m)
        mlflow.log_param('sensitive_attr', self.sensitive_attr)

class FairOBNC(FairOBNCOptimizeDemographicParity):
    """
    Fair Ordering-Based Correction algorithm (Fair-OBNC)

    Attributes
    ----------
    m : float
        Proportion of labels to correct
    sensitive_attr : str
        Name of sensitive attribute
    prob : float
        Probability of correcting a label that does not contribute to balancing label distribution across sensitive groups
    """
    def __init__(self, m:float, sensitive_attr:str):
        super().__init__('Fair-OBNC', m, sensitive_attr)

    def correct(self, X:pd.DataFrame, y:pd.Series):
        X_fair = X.drop(columns=self.sensitive_attr)

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X_fair, y)
        y_pred = pd.Series(bagging.predict(X_fair), index=y.index)

        margins = self.calculate_margins(X_fair.loc[y != y_pred], y.loc[y != y_pred], bagging).apply(lambda x: abs(x)).sort_values(ascending=False)

        y_corrected = self.fair_correction(X, y, y_pred, margins)

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('m', self.m)
        mlflow.log_param('sensitive_attr', self.sensitive_attr)