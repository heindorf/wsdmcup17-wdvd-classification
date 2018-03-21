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

import pandas as pd

from src import evaluationutils
from src import featurelist
from src import queuelogger
from src import utils
from src.dataset import DataSet
from src.pipeline import classification
from src.pipeline import loading
from src.pipeline import preprocessing

_logger = logging.getLogger()
_metrics = pd.DataFrame()

TRAINING_START_DATE = pd.Timestamp('2013-5-1T00:00:00Z')
VALIDATION_SIZE = pd.DateOffset(months=2)
STEP_SIZE = pd.DateOffset(months=2)
END_DATE = pd.Timestamp('2016-07-1T00:00:00Z')


def learn_online(input_file, wdvd_features, filter_features, ores_features):
    validation_start_date = TRAINING_START_DATE + VALIDATION_SIZE
    test_start_date = validation_start_date + VALIDATION_SIZE

    while test_start_date <= END_DATE:
        execute_online_pipeline(
            input_file, 'WDVD', wdvd_features,
            TRAINING_START_DATE, validation_start_date, test_start_date)
        execute_online_pipeline(
            input_file, 'FILTER', filter_features,
            TRAINING_START_DATE, validation_start_date, test_start_date)
        execute_online_pipeline(
            input_file, 'ORES', ores_features,
            TRAINING_START_DATE, validation_start_date, test_start_date)

        validation_start_date = validation_start_date + STEP_SIZE
        test_start_date = test_start_date + STEP_SIZE


def execute_online_pipeline(
        input_file, system_name, features,
        training_start_date, validation_start_date, test_start_date):
    """Execute one independent run (the data is loaded again)."""
    utils.collect_garbage()

    time_label = (
        str(validation_start_date.date()) + ' to ' +
        str(test_start_date.date())
    )

    queuelogger.set_context(time_label, system_name)

    data = loading.load_df(input_file, featurelist.get_columns(features))

    # The revision ids computed here slightly differ from the values in the
    # file constants.py. However, both computations result in exactly the same
    # training and validation set. The reason for different revision ids is
    # that the corpus does not contain bot revisions while the revision ids in
    # the constants file include bot revisions.
    training_start_index = DataSet.get_index_for_date_from_df(data, training_start_date)
    validation_start_index = DataSet.get_index_for_date_from_df(data, validation_start_date)
    test_start_index = DataSet.get_index_for_date_from_df(data, test_start_date)

#     _logger.debug('Training start revisionId: %s' % str(data.loc[training_start_index]['revisionId']))
#     _logger.debug(data.loc[training_start_index-5:training_start_index+5,['revisionId', 'timestamp']])
#     _logger.debug('Validation start revisionId: %s' % str(data.loc[validation_start_index]['revisionId']))
#     _logger.debug(data.loc[validation_start_index-5:validation_start_index+5,['revisionId', 'timestamp']])
#     _logger.debug('Test start revisionId: %s' % str(data.loc[test_start_index]['revisionId']))
#     _logger.debug(data.loc[test_start_index-5:test_start_index+5,['revisionId', 'timestamp']])

    data = data[0: test_start_index]  # preprocessing transformation does not have to be applied to the whole data set
    fit_slice = slice(0, validation_start_index)

    data = preprocessing.fit_transform(
        time_label, system_name, data, features, fit_slice)

    training = data[training_start_index:validation_start_index]
    validation = data[validation_start_index:test_start_index]

    if validation.get_system_name() == 'WDVD':
        metrics = classification.bagging_and_multiple_instance(
            training, validation, print_results=False)
    else:
        metrics = classification.default_random_forest(
            training, validation, print_results=False)

    metrics = metrics.reorder_levels(['Dataset', 'System', 'Classifier'])

    metrics[('ALL', 'VANDALISM_FRACTION')] = validation.get_vandalism_fraction()

    _print_metrics(metrics)


def _print_metrics(metrics):
    global _metrics

    _metrics = _metrics.append(metrics)

    local_metrics = _metrics.copy()

    local_metrics = local_metrics[evaluationutils.COLUMNS]
    evaluationutils.print_metrics(
        local_metrics, 'onlinelearning', append_global=False)
