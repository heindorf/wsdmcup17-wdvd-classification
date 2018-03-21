# -----------------------------------------------------------------------------
# WSDM Cup 2017 Classification and Evaluation
#
# Copyright (c) 2017 Stefan Heindorf, Martin Potthast, Gregor Engels, Benno Stein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

import logging
import itertools

import numpy as np
import pandas as pd
from numpy.core import getlimits
from sklearn.base import TransformerMixin
from sklearn.preprocessing.label import LabelBinarizer

from src.utils import collect_garbage

_logger = logging.getLogger()

########################################################################
# Transformers
#    - Input: Pandas DataFrame
#    - Output: Either Pandas DataFrame (or scipy.sparse.csr_matrix)

########################################################################
# Handling of NAs
########################################################################


class StringImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna("missing")


class ZeroImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.fillna(0)
        result = InfinityImputer().fit_transform(result)

        return result


class MinusOneImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = X.fillna(-1)
        result = InfinityImputer().fit_transform(result)

        return result


class MeanImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.__mean = X.mean()
        return self

    def transform(self, X):
        result = X.fillna(self.__mean)
        result = InfinityImputer().fit_transform(result)

        return result


class MedianImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.__median = X.median()
        return self

    def transform(self, X):
        result = X.fillna(self.__median)
        result = InfinityImputer().fit_transform(result)

        return result


class BooleanImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        collect_garbage()
        result = X.astype(np.float32)
        result = result.fillna(0.5)

        return pd.DataFrame(result)


class InfinityImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X

        for column in X.columns:
            datatype = result.loc[:, column].dtype.type
            limits = getlimits.finfo(datatype)

            result.loc[:, column].replace(np.inf, limits.max, inplace=True)
            result.loc[:, column].replace(-np.inf, limits.min, inplace=True)

        return result


class NaIndicator(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = X.isnull()
        result = result.to_frame()

        return result


class Float32Transformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = X.astype(np.float32)

        return result


########################################################################
# Categories
########################################################################
class CategoryTransformer(TransformerMixin):
    def __init__(self, cat_column_names):
        self.__cat_column_names = cat_column_names

    # Shortcut to gain some speed instead of fit, transform
    def fit_transform(self, X, y=None):
        for column in self.__cat_column_names:
            _logger.debug("Categorizing " + str(column))
            collect_garbage()
            X[column] = X[column].astype('category')

        return X

########################################################################
# Scaling
########################################################################


# computes the formula sign(X)*ceil(log2(|X|+1))
class LogTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X

        sign = result.apply(np.sign)

        result = result.apply(np.absolute)
        result = result + 1
        result = result.apply(np.log2)
        result = result.apply(np.ceil)
        result = sign * result

        result = result.fillna(0)
        return result


class SqrtTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X

        result = result + 1
        result = result.apply(np.sqrt)
        result = result.apply(np.ceil)
        return result


########################################################################
# Handling of Strings
########################################################################

class LengthTransformer(TransformerMixin):
    """Computes the length of a string (both in bytes as well as in words)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if len(X.columns) > 1:
            raise Exception("Only one column supported")
        else:
            X = X.iloc[:, 0]

        collect_garbage()
        X = X.astype(str)  # sometimes fails with MemoryError

        result = pd.DataFrame()
        result['byteLength'] = X.str.len()
        result['byteLength'].fillna(-1, inplace=True)

        rows = X.str.findall("\w+")

        result['wordLength'] = [(len(row) if row == row else -1) for row in rows]

        return result

########################################################################
# Combining features
########################################################################


# see https://github.com/pydata/pandas/issues/11635
def category_workaround(X):
    result = X.copy()
    for column in X.columns:
        if hasattr(X[column], 'cat'):
            result[column] = result[column].cat.codes
    return result


class AggTransformer(TransformerMixin):
    """Aggregates values in the last column of a data frame.

    Given a data frame with columns (c1, c2, ..., cn), computes for each tuple
    (c1, c2, ..., cn-1), the aggregation of the values in the last column cn.

    For example, computes for each user, the average number of bytes added per
    revision (aggregation function is mean).

    For example, computers for each user, the unique number of items
    edited (aggregation function is nunique).
    """

    def __init__(self, func):
        self.__func = func

    def fit(self, X, y=None):
        X = category_workaround(X)

        first_columns = list(X.columns[0:-1])
        last_column = X.columns[-1]

        grouped = X.groupby(by=first_columns)[last_column]

        self.__aggValues = self.__func(grouped)
        self.__aggValues.name = '_aggValues'

        return self

    def transform(self, X):
        X = category_workaround(X)

        first_columns = list(X.columns[0:-1])

        # join first columns with the number of aggregation values in the last
        # column
        result = X.join(self.__aggValues, on=first_columns, how='left')

        # about 10% of users are NA (because they are not in the training set)
        result = result['_aggValues'].fillna(0)
        result = result.to_frame()

        return result


########################################################################
# Handling of categorical values
########################################################################
class UniqueTransformer(TransformerMixin):
    """
    Transforms categorical features to a numeric value (unique frequency).

    Given a data frame with columns (c1, c2, ..., cn), computes for each tuple
    (c1, c2, ..., cn-1) the number of unique values in the last column cn.

    For example, computes for each user the number of uniques items edited.
    """

    def fit(self, X, y=None):
        agg_transformer = AggTransformer(
            pd.core.groupby.SeriesGroupBy.nunique)  # @UndefinedVariable
        agg_transformer.fit(X, y)
        return agg_transformer


class SumTransformer(TransformerMixin):
    def fit(self, X, y=None):
        agg_transformer = AggTransformer(
            pd.core.groupby.DataFrameGroupBy.sum)  # @UndefinedVariable
        agg_transformer.fit(X, y)
        return agg_transformer


class MeanTransformer(TransformerMixin):
    def fit(self, X, y=None):
        agg_transformer = AggTransformer(
            pd.core.groupby.DataFrameGroupBy.mean)  # @UndefinedVariable
        agg_transformer.fit(X, y)
        return agg_transformer


class FrequencyTransformer(TransformerMixin):
    """Transforms categorical features to a numeric value (frequency).

    Given a data frame with columns (C1, C2, ..., Cn), computes for each
    unique tuple (c1, c2, ..., cn), how often it appears in the data frame.

    For example, for every revision, it counts how many revisions were done by
    this user (one column C1='userName').
    """

    def fit(self, X, y=None):
        self.__frequencies = X.groupby(by=list(X.columns)).size()
        self.__frequencies.name = 'frequencies'
        return self

    def transform(self, X):
        result = X.join(self.__frequencies, on=list(X.columns), how='left')

        # all other frequencies are at least 1
        result = result['frequencies'].fillna(0)
        result = result.to_frame()

        return result


class LongTailImputer(TransformerMixin):
    def __init__(self, count=None):
        self.__count = count

    def fit_transform(self, X, y=None, **fit_params):
        top_values = X.groupby(by=X).size()[0:self.__count]

        top_values.name = 'topValues'

        result = pd.DataFrame(X).join(top_values, on=X.name, how='left')
        result = result['topValues'].fillna(0)

        return result


class CategoryBinarizer(TransformerMixin):
    def __init__(self):
        self.__encoder = LabelBinarizer(sparse_output=False)

    def fit(self, X, y=None):
        # X = X.astype(str)
        X = X.values
        self.__encoder.fit(X)
        return self

    def transform(self, X):
        X = X.values
        result = self.__encoder.transform(X)
        result = pd.DataFrame(result)
        result.columns = self.__encoder.classes_

        return result


########################################################################
# Logical Transformers
########################################################################
class LogicalOrTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = pd.Series([False] * len(X))

        for column in X.columns:
            result = result | X[column]

        result = result.to_frame()

        return result


########################################################################
# Cumulative Transformers (only considering information up
# until the current revision)
########################################################################
class CumFrequencyTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumption: X is ordered by revisionId
        grouped = X.groupby(by=list(X.columns))
        result = grouped.cumcount() + 1

        result = result.to_frame()

        return result


class CumAggTransformer(TransformerMixin):

    # func should be a cumulative function
    def __init__(self, func):
        self.__func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        first_columns = list(X.columns[0:-1])
        last_column = X.columns[-1]

        grouped = X.groupby(by=first_columns)[last_column]

        # Some functions pandas.core.groupby.DataFrameGroupBy.X are
        # realized as properties
        if isinstance(self.__func, property):
            result = self.__func.fget(grouped)()
        else:
            result = self.__func(grouped)

        result = result.fillna(0)

        if not isinstance(result, pd.DataFrame):
            result = result.to_frame()

        return result


class CumCountTransformer(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(
            pd.core.groupby.DataFrameGroupBy.cumcount)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer


# not used, because too slow (alternative implementation is used)
class CumSumTransformerPandas(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(
            pd.core.groupby.DataFrameGroupBy.cumsum)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer


class CumMinTransformer(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(
            pd.core.groupby.DataFrameGroupBy.cummin)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer


class CumMaxTransformer(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(
            pd.core.groupby.DataFrameGroupBy.cummax)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer


class CumMeanTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cum_sum_transformer = CumSumTransformer()
        cum_count_transformer = CumCountTransformer()

        sums = cum_sum_transformer.fit_transform(X)
        counts = cum_count_transformer.fit_transform(X)

        result = sums.iloc[:, 0] / (counts.iloc[:, 0] + 1)
        result = result.astype(np.float32)
        result = result.fillna(0)
        result = result.to_frame()

        return result


# This class is implemented in pure Python, because the Pandas implementation
# cumsum is too slow (in particular for many groups such as items)
class CumSumTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    # Assumption: X is ordered by revisionId
    # Assumption: X is indexed 0,1,2, ... , len(df)
    def transform(self, X):
        if X.index[len(X) - 1] != len(X) - 1:
            raise Exception("Expecting data frame to be indexed 0, 1, 2, ...")

        result = [np.nan] * len(X)

        # dictionary of numbers. The dicitonary has the first columns of the
        # dataframe as key and the current sum for those as value.
        dictionary = {}

        for row in X.itertuples(index=True):
            index = row[0]
            first_columns = row[1:-1]
            last_column = row[-1]

            # Skip first_columns containing NaN (dictionaries do not work well
            # with np.float(np.NaN) values)
            # Pandas also ignores groups containing NaN
            if any([x != x for x in first_columns]):
                continue

            if first_columns not in dictionary:
                dictionary[first_columns] = 0

            # Ignore NaN values in last column
            if last_column == last_column:
                dictionary[first_columns] += last_column

            result[index] = dictionary[first_columns]

        result = pd.DataFrame(result)

        result = result.fillna(0)  # necessary for first_columns containing NaN

        return result


class CumUniqueTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    # Assumption: df is ordered by revisionId
    # Assumption: df is indexed 0,1,2, ... , len(df)
    def transform(self, X):
        if X.index[len(X) - 1] != len(X) - 1:
            raise Exception("Expecting data frame to be indexed 0, 1, 2, ...")

        result = [np.nan] * len(X)

        # Dictionary of sets: The dictionary has the first columns of the
        # dataframe as key and the set contains all the values of the
        # last column
        dictionary = {}

        for row in X.itertuples(index=True):
            index = row[0]
            first_columns = row[1:-1]
            last_column = row[-1]

            # Skip first_columns containing NaN (dictionaries do not work well
            # with np.float(np.NaN) values)
            # Pandas also ignores groups containing NaN
            if any([x != x for x in first_columns]):
                continue

            if first_columns not in dictionary:
                dictionary[first_columns] = set()

            # Ignore NaN values in last column
            if last_column == last_column:
                dictionary[first_columns].add(last_column)

            result[index] = len(dictionary[first_columns])

        result = pd.DataFrame(result)
        return result


class LastTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = [np.nan] * len(X)

        # dictionary of numbers. The dicitonary has the first columns of the
        # dataframe as key and the current sum for those as value.
        dictionary = {}

        for row in X.itertuples(index=True):
            index = row[0]
            first_columns = row[1:-1]
            last_column = row[-1]

            # Skip first_columns containing NaN (dictionaries do not work well
            # with np.float(np.NaN) values)
            # Pandas also ignores groups containing NaN
            if any([x != x for x in first_columns]):
                continue

            if first_columns not in dictionary:
                dictionary[first_columns] = "NA"

            # this is the last value (e.g., the last property)
            result[index] = dictionary[first_columns]

            # Ignore NaN values in last column
            if last_column == last_column:
                dictionary[first_columns] = last_column

        result = pd.DataFrame(result)

        # necessary for first_columns containing NaN
        result = result.fillna("NA")

        return result


########################################################################
# Time Transformers
########################################################################
class TimeTransformer(TransformerMixin):
    def __init__(self, unit):
        self._unit = unit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        timestamp_series = X.iloc[:, 0]
        result = pd.DataFrame()

        if self._unit == 'hourOfDay':
            result['hourOfDay'] = pd.DataFrame(timestamp_series.dt.hour)
        elif self._unit == 'dayOfWeek':
            result['dayOfWeek'] = pd.DataFrame(timestamp_series.dt.weekday)
        elif self._unit == 'dayOfMonth':
            result['dayOfMonth'] = pd.DataFrame(timestamp_series.dt.day)
        else:
            raise Exception('undefined unit')

        return result


########################################################################
# Special Transformer
########################################################################
class TimeSinceLastRevisionTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    # assumption: first column is username, second column is timestamp
    def transform(self, X):
        X = X.copy()

        # convert to seconds since initial point in time
        X['timestamp'] = X['timestamp'].astype(np.int64)
        result = X.groupby('userName').diff(1)

        return result


########################################################################
# Tag Transformers for Revision Tags
########################################################################
class MostFrequentTagTransformer(TransformerMixin):
    """Transforms multiple tags to the single, most frequent tag.

    Returns the most frequent tag for each row in a single column.
    For example, ['def,abc','abc', np.nan] would be transformed to
    ['abc', 'abc', np.nan] (because abc is more frequent than def).
    """

    def fit(self, X, y=None):
        _logger.debug("Fitting revisionTag...")
        _logger.debug("Splitting...")
        tmp = X.iloc[:, 0].str.split(',')
        _logger.debug("Dropping...")
        tmp = tmp.dropna()
        _logger.debug("Summing up...")

        # this line is incredibly slow, hence we use the following alternative
        # tmp = tmp.sum()

        tmp = tmp.tolist()
        tmp = list(itertools.chain.from_iterable(tmp))

        _logger.debug("counting values...")
        self.__freq = pd.Series(tmp).value_counts()
        _logger.debug("Fitting revisionTag... done.")
        return self

    def transform(self, X):
        _logger.debug('Transforming revisionTag...')

        def get_freq(element):
            result = self.__freq.get(element)
            if result is None:
                result = 0

            return result

        def most_frequent(x):
            if x == x:
                result = sorted(x, key=get_freq, reverse=True)[0]
            else:
                result = x  # x is NaN
            return result

        mapping = {}
        for category in X.iloc[:, 0].cat.categories:
            mapping[category] = most_frequent(category.split(','))

        result = X.iloc[:, 0].map(mapping)

        result = result.astype('category')
        result = result.to_frame()

        _logger.debug('Transforming revisionTag... done.')

        return result


########################################################################
# Unused Transformers (because they are too slow)
########################################################################
class CumUniqueTransformerPandas(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        first_columns = list(X.columns[0:-1])

        _logger.debug("dropping duplicates...")
        without_duplicates = X.drop_duplicates()

        _logger.debug("grouping...")
        unique_count = (without_duplicates.groupby(first_columns)
                                          .cumcount()
                                          .astype(np.float32) + 1)
        unique_count.name = 'unique_count'

        _logger.debug("joining...")
        result = X.join(unique_count, how='left')

        _logger.debug("grouping...")
        grouped = result.groupby(first_columns)

        # very slow (because internall it does not use cython and creates
        # a lot of data frames)
        _logger.debug("filling...")
        result = grouped.ffill()

        _logger.debug("transforming to frame...")
        result = result[unique_count.name].to_frame()

        return result
