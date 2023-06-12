from abc import ABC, abstractmethod
import pandas as pd

class LabelCorrectionModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def correct(self, X:pd.DataFrame, y:pd.Series) -> pd.Series:
        """
        Corrects the labels of the given dataset

        Parameters
        ----------
        X : pd.DataFrame
            Dataset features
        y : pd.Series
            Labels to correct

        Returns
        -------
        y_corrected: pd.Series
            Corrected labels
        """
        pass

    @abstractmethod
    def log_params(self):
        pass
