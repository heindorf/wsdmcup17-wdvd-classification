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
import copy
import sys

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Parallel, delayed

import config
from src import featurelist
from src import storage
from src import transformers
from src import utils
from src.dataset import DataSet
from src.utils import Timer


_logger = logging.getLogger()


def fit_transform(time_label, system_name, data, features, fit_slice):
    _logger.info("Preprocessing...")
    utils.collect_garbage()

    features = build_features(features)

    # First, iloc creates a (deep) copy of the data because multiple dtypes are involved
    # Second, iloc create a view of the data because only one dtype is involved
    # http://stackoverflow.com/a/23296545/6244640
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].copy().astype(np.float32)
    data.drop(data.columns, axis=1, inplace=True)  # free memory

    fitting_X = X[fit_slice]

    preprocessor = MultiFeatureTransformer(time_label, system_name, features)
    data = preprocessor.fit(fitting_X).transform(X)

    data = build_dataset(data, y)

    storage.dump_preprocessor(preprocessor, time_label, system_name)

    df = pd.DataFrame(data.get_X())
    df.columns = data.get_feature_names()
    df.describe(include='all').to_csv(
        "%s_%s_%s_feature_statistics_preprocessed.csv" %
        (config.OUTPUT_PREFIX, time_label, system_name))

    utils.collect_garbage()

    _logger.info("Preprocessing... done.")

    data.set_time_label(time_label)
    data.set_system_name(system_name)

    return data


def build_features(features):
    tmp_features = copy.deepcopy(features)
    for feature in tmp_features:
        feature.get_transformers().append(transformers.Float32Transformer())

    result = copy.deepcopy(featurelist.get_meta_list()) + tmp_features

    return result


def build_dataset(df, y):
    _logger.debug('building dataset...')

    _logger.debug('slicing...')

    _logger.debug('meta...')
    n_meta = len(featurelist.get_meta_list())
    new_meta = df.iloc[:, 0:n_meta]

    _logger.debug('X...')
    new_X = df.iloc[:, n_meta:].values  # takes a looong time
    features = df.columns[n_meta:]

    _logger.debug('y...')
    new_Y = y.values

    utils.collect_garbage()

    _logger.debug('dataset...')
    new_data = DataSet()

    _logger.debug('set_meta...')
    new_data.set_meta(new_meta)

    _logger.debug('set_X...')
    new_data.set_X(new_X)

    _logger.debug('set_Y...')
    new_data.set_Y(new_Y)

    _logger.debug('set_features...')
    new_data.set_features(features)

    _logger.debug('building dataset...done.')

    return new_data


class MultiFeatureTransformer(TransformerMixin):
    def __init__(self, time_label, system_name, features):
        self.time_label = time_label
        self.system_name = system_name

        # features contain the (fitted) transformers
        self.features = features

        self.feature_transformers = {}
        for feature in self.features:
            self.feature_transformers[feature.get_output_name()] = \
                FeatureTransformer(feature)

    def fit(self, X):
        arguments = self._build_arguments(X)

        # this is the central part of this function
        Parallel(n_jobs=1, backend='multiprocessing')(
            delayed(MultiFeatureTransformer._fit_feature)(*arguments[i])
            for i in range(len(arguments))
        )

        return self

    def transform(self, X):
        arguments = self._build_arguments(X)

        # this is the central part of this function
        df_list = Parallel(n_jobs=1, backend='multiprocessing')(
            delayed(MultiFeatureTransformer._transform_feature)(*arguments[i])
            for i in range(len(arguments))
        )

        result = MultiFeatureTransformer._combine_dfs(df_list)

        return result

    def _build_arguments(self, X):
        arguments = []

        for feature in self.features:
            feature_input_names = feature.get_input_names()
            # List is not empty?
            if feature_input_names:
                # All preconditions for feature fulfilled?
                if set(feature_input_names).issubset(X.columns):
                    cur_data = X.loc[:, feature_input_names]
                else:
                    MultiFeatureTransformer._log_missing_columns(feature_input_names)
                    break
            else:
                cur_data = pd.DataFrame()

            arguments.append(
                (self.feature_transformers[feature.get_output_name()], cur_data))

        return arguments

    # Log missing columns only one time
    missing_columns = set()

    @staticmethod
    def _log_missing_columns(feature_input_names):
        if not set(feature_input_names).issubset(MultiFeatureTransformer.missing_columns):
            _logger.warn("Input data missing: %s", feature_input_names)
            MultiFeatureTransformer.missing_columns = \
                MultiFeatureTransformer.missing_columns.union(set(feature_input_names))

    # This method is called by multiple processes
    @staticmethod
    def _fit_feature(feature_processor, data):
        utils.collect_garbage()

        _logger.debug(
            "Fitting feature %s (%s)...",
            str(feature_processor.feature.get_output_name()),
            str(feature_processor.feature.get_transformers())
        )

        with Timer() as t:
            result = feature_processor.fit(data)

        _logger.debug(
            "=> elapsed time for feature %s: %f s (current data size: %f MB)",
            feature_processor.feature.get_output_name(),
            t.secs,
            sys.getsizeof(result) / 1024 / 1024)

        utils.collect_garbage()

        return result

    # This method is called by multiple processes
    @staticmethod
    def _transform_feature(feature_processor, data):
        _logger.debug(
            "transforming feature %s (%s)...",
            str(feature_processor.feature.get_output_name()),
            str(feature_processor.feature.get_transformers())
        )

        with Timer() as t:
            result = feature_processor.transform(data)

        _logger.debug(
            "=> elapsed time for feature %s: %f s (current data size: %f MB)",
            feature_processor.feature.get_output_name(),
            t.secs,
            sys.getsizeof(result) / 1024 / 1024)

        utils.collect_garbage()

        return result

    # returns np.array
    @staticmethod
    def _combine_dfs(df_list):
        result = pd.DataFrame()

        for df in df_list:
            for column in df.columns:
                if column in result.columns:
                    raise Exception("duplicate column: " + str(column))

                result[column] = df[column]

        return result

    @staticmethod
    def _checkTransformation(training_X, feature_names):
        if len(feature_names) != training_X.shape[1]:
            raise Exception("There are " +
                            str(len(feature_names)) + " feature names but " +
                            str(training_X.shape[1]) + " features in the data")


class FeatureTransformer(TransformerMixin):
    def __init__(self, feature):
        self.feature = feature

    def fit(self, X):
        try:
            transformers = self.feature.get_transformers()
            if len(transformers) > 0:
                for transformer in transformers[:-1]:
                    X = transformer.fit(X).transform(X)

                transformer = transformers[-1]
                transformer.fit(X)
        except:  # noqa
            _logger.error(self.feature.get_output_name() + ": ")
            raise

        return self

    def transform(self, X):
        try:
            for transformer in self.feature.get_transformers():
                X = transformer.transform(X)

                if not isinstance(X, pd.DataFrame):
                    raise Exception(
                        "return type should be pandas.DataFrame but was " +
                        str(type(X)))

                if len(X.columns.values) == 1:
                    name = self.feature.get_output_name()
                    X.columns = self._generateColumn(self.feature, name)

                elif len(X.columns.values) > 1:
                    new_columns = []
                    for i in range(len(X.columns.values)):
                        name = self.feature.output_name + "_" + str(X.columns.values[i])
                        column = self._generateColumn(self.feature, name)
                        new_columns.append(column)
                    X.columns = new_columns
        except:  # noqa
            _logger.error(self.feature.get_output_name() + ": " + str(transformer))
            raise

        return X

    @staticmethod
    def _generateColumn(feature, name):
        if (feature.get_group() is not None) & (feature.get_subgroup() is not None):
            result = pd.MultiIndex.from_tuples(
                [(feature.get_group(), feature.get_subgroup(), name)],
                names=['group', 'subgroup', 'feature'])
        else:
            result = [name]

        return result


########################################################################
# Given a pandas data frame and a list of features,
# applies the features' transformers on the data frame and
# returns the whole transformed data set
def fit_on_slice_transform(self, X, fit_slice):
    _logger.info("Preprocessing...")
    utils.collect_garbage()

    data = self._applyTransformers(
        X, self.features, do_fit=True, do_transform=True, fit_slice=fit_slice,
        n_jobs=config.PREPROCESSING_N_JOBS)

    _logger.debug('checking transformation...')
    MultiFeatureTransformer._checkTransformation(data.get_X(), data.get_feature_names())

    utils.collect_garbage()

    _logger.info("Preprocessing... done.")

    data.set_time_label(self.time_label)
    data.set_system_name(self.system_name)

    return data
