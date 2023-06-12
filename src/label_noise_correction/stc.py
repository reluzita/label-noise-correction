from .labelcorrection import LabelCorrectionModel
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import mlflow

class SelfTrainingCorrection(LabelCorrectionModel):
    """
    Self-Training Correction algorithm

    Reference:
    Nicholson, Bryce, et al. "Label noise correction methods." 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015.

    Attributes
    ----------
    classifier
        Type of classifier to use 
    n_folds : int
        Number of folds to use
    correction_rate : float
        Percentage of labels to correct
    """
    def __init__(self, classifier, n_folds, correction_rate):
        super().__init__('STC')
        self.classifier = classifier
        self.n_folds = n_folds
        self.correction_rate = correction_rate

    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
        noisy = set()

        # Split the current training data set using an n-fold cross-validation scheme
        for train_index, test_index in kf.split(X, y):
            X_train = X.loc[train_index]
            X_test = X.loc[test_index]
            y_train = y.loc[train_index]
            y_test = y.loc[test_index]

            # For each of these n parts, a learning algorithm is trained on the other n-1 parts, resulting in n different classifiers
            model = self.classifier(random_state=42).fit(X_train, y_train)

            # These n classifiers are used to tag each instance in the excluded part as either correct or mislabeled, by comparing the training label with that assigned by the classifier.
            y_pred = pd.Series(model.predict(X_test), index=test_index)
            
            # The misclassified examples from the previous step are added to the noisy data set.
            for i, value in y_pred.items():
                if value != y_test.loc[i]:
                    noisy.add(i)

        noisy = list(noisy)

        X_clean = X.drop(noisy)
        y_clean = y.drop(noisy)
        X_noisy = X.loc[noisy]

        if y_clean.unique().shape[0] == 1:
            print('All selected clean labels are the same, returning original labels')
            return pd.Series(y.values, index=original_index)
        
        if len(noisy) == 0:
            print('All labels are clean, returning original labels')
            return pd.Series(y.values, index=original_index)

        # Build a model from the clean set and uses that to calculate the confidence that each of the instances from the noisy set is mislabeled
        model = self.classifier(random_state=42).fit(X_clean, y_clean)
        y_pred = pd.Series(model.predict(X_noisy), index=noisy)
        y_prob = pd.DataFrame(model.predict_proba(X_noisy), index=noisy, columns=[0, 1])

        corrected = pd.DataFrame(columns=['y_pred', 'y_prob'])
        for i in list(noisy):
            if y_pred.loc[i] != y.loc[i]:
                corrected.loc[i] = [y_pred.loc[i], y_prob.loc[i, y_pred.loc[i]]]

        # The noisy instance with the highest calculated likelihood of belonging to some class that is not equal to its current class 
        # is relabeled to the class that the classifier determined is the instance’s most likely true class. 
        correction_n = int(self.correction_rate*len(corrected))
        y_corrected = y.values
        for i in corrected.sort_values('y_prob', ascending=False)[:correction_n].index:
            y_corrected[i] = corrected.loc[i, 'y_pred']

        return pd.Series(y_corrected, index=original_index)

    def log_params(self):
        mlflow.log_param('correction_alg', self.name)
        mlflow.log_param('correction_classifier', self.classifier.__name__)
        mlflow.log_param('n_folds', self.n_folds)
        mlflow.log_param('correction_rate', self.correction_rate)