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

import config
from src import constants
from src import featurelist
from src import queuelogger
from src import utils
from src.dataset import DataSet
from src.pipeline import classification
from src.pipeline import loading
from src.pipeline import featureranking
from src.pipeline import onlinelearning
from src.pipeline import optimization
from src.pipeline import preprocessing
from src.pipeline import statistics


########################################################################
# Remark: To execute this script the 64 bit version of Python, NumPy,
#         and SciPy is required (otherwise a MemoryError occurs)
########################################################################
_logger = logging.getLogger()

VALIDATION = 'VALIDATION'
TEST = 'TEST'

WDVD = 'WDVD'
ORES = 'ORES'
FILTER = 'FILTER'


def main(files):
    utils.print_system_info()
    utils.init_pandas()

    _logger.info("FILES=" + str(files))

    run_all(files)


def run_all(files):
    wdvd_features = featurelist.get_feature_list()
    filter_features = featurelist.get_filter_feature_list()
    ores_features = featurelist.get_ores_feature_list()

    if config.STATISTICS_ENABLED:
        compute_statistics(files)

    if config.USE_VALIDATION_SET:
        execute_pipeline(    files, VALIDATION, WDVD  , wdvd_features  , use_test_set=False, rank_features=config.FEATURE_RANKING_ENABLED, optimize=config.OPTIMIZATION_ENABLED)  # noqa
        if config.BASELINES_ENABLED:
            execute_pipeline(files, VALIDATION, FILTER, filter_features, use_test_set=False, rank_features=False                         , optimize=config.OPTIMIZATION_ENABLED)  # noqa
            execute_pipeline(files, VALIDATION, ORES  , ores_features  , use_test_set=False, rank_features=False                         , optimize=config.OPTIMIZATION_ENABLED)  # noqa

    if config.USE_TEST_SET:
        execute_pipeline(    files, TEST      , WDVD  , wdvd_features  , use_test_set=True , rank_features=config.FEATURE_RANKING_ENABLED, optimize=False)  # noqa
        if config.BASELINES_ENABLED:
            execute_pipeline(files, TEST      , FILTER, filter_features, use_test_set=True , rank_features=False                         , optimize=False)  # noqa
            execute_pipeline(files, TEST      , ORES  , ores_features  , use_test_set=True , rank_features=False                         , optimize=False)  # noqa

    if config.ONLINE_LEARNING_ENABLED:
        onlinelearning.learn_online(
            files, wdvd_features, filter_features, ores_features)


def omit_holdout_df(df):
    """Omit the holdout dataframe."""
    tail_set_start_index = \
        DataSet.get_index_for_revision_id_from_df(df, constants.TAIL_SET_START)
    df = df[:tail_set_start_index]
    return df


def get_splitting_indices(data, use_test_set):
    training_set_start = constants.TRAINING_SET_START

    if use_test_set:
        validation_set_start = constants.TEST_SET_START
        test_set_start = constants.TAIL_SET_START
    else:
        validation_set_start = constants.VALIDATION_SET_START,
        test_set_start = constants.TEST_SET_START

    # transform revision id to index in data set
    training_set_start = DataSet.get_index_for_revision_id_from_df(
        data, training_set_start)
    validation_set_start = DataSet.get_index_for_revision_id_from_df(
        data, validation_set_start)
    test_set_start = DataSet.get_index_for_revision_id_from_df(
        data, test_set_start)

    return training_set_start, validation_set_start, test_set_start


def compute_statistics(files):
    df = loading.load_df(files, None)  # compute statistics for all columns
    df = omit_holdout_df(df)  # omit the holdout dataset
    statistics.compute_statistics(df)
    df = None  # free memory


# one independent run (the data is loaded again)
def execute_pipeline(files, time_label, system_name, features,
                     use_test_set, rank_features, optimize):
    queuelogger.set_context(time_label, system_name)

    if (rank_features or optimize or config.CLASSIFICATION_ENABLED or
            config.CLASSIFICATION_GROUPS_ENABLED):
        data = loading.load_df(files, featurelist.get_columns(features))

        training_set_start, validation_set_start, test_set_start = \
            get_splitting_indices(data, use_test_set)

        # Starting the fitting at 0 yields the best results
        fit_slice = slice(0, validation_set_start)

        data = preprocessing.fit_transform(
            time_label, system_name, data, features, fit_slice)

        # Splitting the data set into training and validation sets
        training = data[training_set_start:validation_set_start]
        validation = data[validation_set_start:test_set_start]
        _logger.debug("Training size: " + str(len(training)))
        _logger.debug("Validation size: " + str(len(validation)))

    if rank_features:
        featureranking.rank_features(training, validation)

    if optimize:
        optimization.optimize(training, validation)

    if config.CLASSIFICATION_ENABLED:
        if training.get_system_name() == 'WDVD':
            classification.default_random_forest(training, validation)
            classification.optimized_random_forest(training, validation)
            classification.bagging_and_multiple_instance(training, validation)
        elif training.get_system_name() == 'FILTER':
            classification.default_random_forest(training, validation)
        elif training.get_system_name() == 'ORES':
            classification.optimized_random_forest(training, validation)

    if config.CLASSIFICATION_GROUPS_ENABLED:
        if training.get_system_name() == 'WDVD':
            classification.compute_metrics_for_classifiers_and_groups(
                training, validation)
