import pandas as pd

from sklearn.base import TransformerMixin


class EqualsTransformer(TransformerMixin):
    def __init__(self, value):
        self.__value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # value is assumed to be a tuple
        result = [True] * len(X)

        for i in range(len(self.__value)):
            result = result & self.isNanEqual(X.iloc[:, i], self.__value[i])

        result = pd.DataFrame(result)
        result.columns = [str(self.__value)]

        return result

    @staticmethod
    def isNanEqual(a, b):
        result = ((a == b) | (pd.isnull(a) & pd.isnull(b)))
        return result


class WeightTransformer(TransformerMixin):
    # X is expected to be a one dimensional numpy array
    def fit(self, X, y=None):
        self.weights = self._balanced_weights(list(X))
        return self

    def transform(self, X):
        result = self._balanced_sample_weights(self.weights, list(X))
        return result

    # Taken from revscoring/scorer_models/util.py
    @staticmethod
    def _balanced_weights(labels):
        """Compute weights for classes.

        Generates a mapping of class weights that will re-weight a training set
        in a balanced way such that weight(label) = len(obs) / freq(label in obs).
        """
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1

        return {l: (len(labels) / counts[l]) for l in counts}

    # Taken from revscoring/scorer_models/util.py
    @staticmethod
    def _balanced_sample_weights(weights, labels):
        """Generate a vector of balancing weights for a vector of labels."""
        return [weights[label] for label in labels]
