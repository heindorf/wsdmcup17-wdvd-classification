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

import pandas as pd

import config
from src import evaluationutils
from src import storage
from src.classifiers import multipleinstance
from src.pipeline import optimization

_logger = logging.getLogger()
_metrics = {}  # dictionary of pd.DataFrames, the key is the time label

DFOREST = 'DFOREST'
OFOREST = 'OFOREST'
BAGGING = 'BAGGING'


def default_random_forest(
        training, validation, group=None, print_results=True):
    _logger.info("Default random forest...")
    clf = optimization.get_default_random_forest()
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS)

    index = _get_index(validation, 'Default random forest', group)

    _, _, metrics = evaluationutils.fit_predict_evaluate(
        index, clf, training, validation)

    if print_results:
        _logger.info("Writing model to file ...")
        clf_name = evaluationutils.index_to_str(index)
        storage.dump_clf(clf, clf_name)
        _print_metrics(metrics)
    _logger.info("Default random forest... done.")

    return metrics


def optimized_random_forest(
        training, validation, group=None, print_results=True, sample_weight=None):
    _logger.info("Optimized random forest...")
    clf = optimization.get_optimal_random_forest(validation.get_system_name())
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS)

    index = _get_index(validation, 'Optimized random forest', group)

    _, _, metrics = evaluationutils.fit_predict_evaluate(
        index, clf, training, validation, sample_weight=sample_weight)

    if print_results:
        _logger.info("Writing model to file ...")
        clf.set_params(n_jobs=1)
        clf_name = evaluationutils.index_to_str(index)
        storage.dump_clf(clf, clf_name)
        _print_metrics(metrics)
    _logger.info("Optimized random forest... done.")

    return metrics


def bagging_and_multiple_instance(
        training, validation, group=None, print_results=True):
    _logger.info("Bagging and multiple-instance...")

    result = pd.DataFrame()

    # Bagging
    clf = optimization.get_optimal_bagging_classifier(validation.get_system_name())
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS)
    index = _get_index(validation, 'Bagging', group)
    _, prob, metrics = evaluationutils.fit_predict_evaluate(
        index, clf, training, validation)
    result = result.append(metrics)
    if print_results:
        clf_name = evaluationutils.index_to_str(index)
        storage.dump_clf(clf, clf_name)
        _print_metrics(metrics)

    # Single-instance learning (SIL)
    clf = multipleinstance.SingleInstanceClassifier(
        base_estimator=None, agg_func='cummean', window=config.BACKPRESSURE_WINDOW)
    clf.set_proba(prob)  # shortcut to save some computational time
    index = _get_index(validation, 'SIL MI', group)
    sil_pred, sil_prob = evaluationutils.predict(clf, validation, index)
    metrics = evaluationutils.evaluate(index, sil_pred, sil_prob, validation)
    result = result.append(metrics)
    if print_results:
        _print_metrics(metrics)

    # Simple multiple-instance (SMI)
    clf = optimization.get_optimal_bagging_classifier(validation.get_system_name())
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS_SIMPLE_MI)
    clf = multipleinstance.SimpleMultipleInstanceClassifier(
        base_estimator=clf, trans_func='cummin_cummax', window=config.BACKPRESSURE_WINDOW)
    index = _get_index(validation, 'Simple MI', group)
    _, smi_prob, metrics = evaluationutils.fit_predict_evaluate(
        index, clf, training, validation)
    result = result.append(metrics)
    if print_results:
        _print_metrics(metrics)

    # Combination of SIL and SMI
    clf = multipleinstance.CombinedMultipleInstanceClassifier(
        base_estimator=None)
    # shortcut to save some computational time
    clf.set_proba(sil_prob, smi_prob)
    index = _get_index(validation, 'Combined MI', group)
    combined_pred, combined_prob = evaluationutils.predict(clf, validation, index)
    metrics = evaluationutils.evaluate(
        index, combined_pred, combined_prob, validation)

    result = result.append(metrics)
    if print_results:
        _print_metrics(metrics)

    _logger.info("Bagging and multiple-instance... done.")

    return result


def _get_index(dataset, clf_name, group_name=None):
    values = (dataset.get_time_label(), dataset.get_system_name(), clf_name)
    names = ['Dataset', 'System', 'Classifier']

    if group_name is not None:
        values = values + (group_name,)
        names.append('Group')

    result = pd.MultiIndex.from_tuples([values], names=names)

    return result


def compute_metrics_for_classifiers_and_groups(training, validation):
    local_metrics = pd.DataFrame()

    for clf_name in [DFOREST, OFOREST, BAGGING]:
        for group in ['ALL'] + validation.get_groups():
            training2 = training.select_group(group)
            validation2 = validation.select_group(group)

            result = _compute_metrics_for_classifier(
                training2, validation2, clf_name, group)

            local_metrics = local_metrics.append(result)
            local_metrics.to_csv(config.OUTPUT_PREFIX + '_' +
                                 validation.get_time_label() +
                                 '_classifiers_groups.csv')
            screen_output = evaluationutils.remove_plots(local_metrics)
            _logger.info("Metrics:\n" + str(screen_output))


def _compute_metrics_for_classifier(training, validation, clf_name, group):
    if clf_name == DFOREST:
        result = default_random_forest(
            training, validation, group=group, print_results=False)
    elif clf_name == OFOREST:
        result = optimized_random_forest(
            training, validation, group=group, print_results=False)
    elif clf_name == BAGGING:
        result = bagging_and_multiple_instance(
            training, validation, group=group, print_results=False)

    return result


def _print_metrics(metrics):
    """Print one metrics row and save it."""
    time_label = metrics.index.get_level_values('Dataset')[0]

    global _metrics
    if time_label not in _metrics:
        _metrics[time_label] = pd.DataFrame()

    _metrics[time_label] = _metrics[time_label].append(metrics)

    local_metrics = _metrics[time_label].copy()

    local_metrics = _reverse_order_within_system_groups(local_metrics)

    local_metrics = local_metrics[evaluationutils.COLUMNS]

    evaluationutils.print_metrics(
        local_metrics, time_label + '_results', append_global=False)


def _reverse_order_within_system_groups(metrics):
    # returns a list of tuples based on the data frame's multiIndex
    old_order = metrics.index.values
    new_order = []

    # group by first and second index column (Dataset and System)
    for _, group in itertools.groupby(old_order, lambda x: x[0:2]):
        new_order = new_order + list(group)[::-1]

    result = metrics.reindex(new_order)

    return result
