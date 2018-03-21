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

import collections
import csv
import itertools
import logging
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import scipy.interpolate
import sklearn.metrics

import config
from src import utils
from src.classifiers import multipleinstance
from src import streamtransformers
from src import storage
from src import dataset


CURVES = ['fprValues', 'tprValues', 'rocThresholds', 'precisionValues', 'recallValues']
PRED_METRICS = ['ACC', 'P', 'R', 'F']  # Metrics requiring predictions
PROB_METRICS = ['PR', 'ROC']
STATISTICS = ['RESULTS', 'VANDALISM']
COLUMNS = (
    list(itertools.product(['ALL'], STATISTICS + PRED_METRICS + PROB_METRICS + CURVES)) +
    list(itertools.product(['ITEM_HEAD', 'ITEM_BODY'], STATISTICS + PROB_METRICS)) +
    list(itertools.product(['REGISTERED', 'UNREGISTERED'], STATISTICS + PROB_METRICS))
)

_metrics = pd.DataFrame()
_logger = logging.getLogger()


#######################################################################
# Computing metrics of classifiers / features
#######################################################################
def fit(clf, dataset, index='', sample_weight=None):
    label = _get_label(index)

    _logger.debug("Fitting %s..." % label)
    if (isinstance(clf, multipleinstance.BaseMultipleInstanceClassifier)):
        g = dataset.get_group_ids()
        clf.fit(g, dataset.get_X(), dataset.get_Y())
    else:
        clf.fit(dataset.get_X(), dataset.get_Y(), sample_weight)

    _logger.debug("Fitting %s... done." % label)


def predict(clf, dataset, index=''):
    label = _get_label(index)

    _logger.debug("Predicting %s..." % label)
    if (isinstance(clf, multipleinstance.BaseMultipleInstanceClassifier)):
        g = dataset.get_group_ids()
        prob = clf.predict_proba(g, dataset.get_X())
    else:
        # second column denotes the probability for vandalism
        prob = clf.predict_proba(dataset.get_X())[:, 1]

    pred = get_pred_from_prob(prob)

    _logger.debug("Predicting %s... done." % label)
    return pred, prob


def get_pred_from_prob(prob):
    return np.asarray(prob >= 0.5)


def split_groups(dataset):
    r = dataset.get_revision_ids().values
    g = dataset.get_group_ids().values
    s = dataset.get_meta()['revisionAction'] == 'rollback'

    result = [np.nan] * len(g)
    transformer = streamtransformers.StreamGroupSplitTransformer()
    for i in range(len(g)):
        result[i] = transformer.partial_fit_transform(r[i], g[i], s[i])

    _logger.debug("Number of group splits: " + str(transformer.group_splits))

    return result


def _get_label(index):
    if hasattr(index, 'values'):
        label = "%s" % (str(index.values[0]))
    else:
        label = str(index)

    return label


def evaluate_print(name, pred, prob, dataset):
    metrics = evaluate(name, pred, prob, dataset)
    print_metrics(metrics)
    result = metrics.iloc[0].loc[('ALL', 'PR')]

    return result


def evaluate(index, pred, prob, ds, save_prob=True, fit_time=-1, prob_time=-1):
    label = _get_label(index)
    _logger.debug("Evaluating %s..." % label)

    # might perform some conversion internally (e.g., from float64 to float32)
    name = index_to_str(index)
    storage.dump_predictions(name, ds, prob, tmp=not save_prob)
    prob = storage.load_predictions(name, tmp=not save_prob)['VANDALISM_SCORE'].values

    local_metrics = compute_metrics(
        index, ds.get_metrics_meta(), ds.get_Y(), prob, pred)

    idx = local_metrics.columns.get_loc(('ALL', 'PR')) + 1

    local_metrics.insert(idx, ('ALL', 'TOTAL_TIME'), fit_time + prob_time)
    local_metrics.insert(idx, ('ALL', 'PROB_TIME'), prob_time)
    local_metrics.insert(idx, ('ALL', 'FIT_TIME'), fit_time)

    _logger.debug("Evaluating %s... done." % label)

    return local_metrics


def index_to_str(index):
    """Convert Pandas MultiIndex to String."""
    if isinstance(index, pd.core.index.MultiIndex):
        index_str_entries = map(str, index[0])  # convert index levels to string
        filename = '_'.join(index_str_entries).replace(' ', '_')
    else:  # multiindex
        filename = index
    return filename


def fit_predict_evaluate(
        index, clf, training, validation, save_prob=True, sample_weight=None):
    fit_start = time.time()
    fit(clf, training, index, sample_weight)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    prob_start = time.time()
    pred, prob = predict(clf, validation, index)

    prob_end = time.time()
    prob_time = prob_end - prob_start

    metrics = evaluate(index, pred, prob, validation,
                       save_prob, fit_time, prob_time)

    return pred, prob, metrics


def remove_plots(metrics):
    return remove_columns(metrics, CURVES)


def remove_columns(metrics, columns):
    labels_to_drop = list(itertools.product(
        metrics.columns.levels[0], columns))
    result = metrics.drop(labels_to_drop, axis=1, errors='ignore')

    return result


def _remove_duplicates(seq):
    result = []
    for e in seq:
        if e not in result:
            result.append(e)
    return result


def print_metrics(metrics, suffix='metrics', append_global=True):
    if (append_global):
        global _metrics
        _metrics = _metrics.append(metrics)
        _print_metrics(_metrics, suffix)
    else:
        _print_metrics(metrics, suffix)


def _print_metrics(metrics, suffix):
    metrics.to_csv(config.OUTPUT_PREFIX + '_' + suffix + '.csv')

    metrics = remove_columns(metrics, CURVES)

    _logger.info("Metrics for %s:\n" % suffix +
                 (metrics.to_string(float_format='{:.4f}'.format)))

    metrics = remove_columns(metrics, STATISTICS)

    print_metrics_to_latex(metrics,
                           config.OUTPUT_PREFIX + '_' + suffix + '.tex')


def print_metrics_to_latex(metrics, filename):
    r"""Print metrics to latex and format them as \\bscellA{}."""
    metrics = metrics.copy()

    def cell_format(value, char):
        return '\\bscell%s[%.3f]{%3.0f}' % (char, value, value * 100)

    def float_format_short(value):
        return '%.3f' % (value,)

    def float_format_long(value):
        return '%.4f' % (value,)

    def float_formatA(value):
        return cell_format(value, 'A')

    def float_formatB(value):
        return cell_format(value, 'B')

    # use formatB for all 'ROC' columns
    formatters = {}
    for value in metrics.columns.values:
        if value[1] == 'PR':
            formatters[value] = float_formatA
        elif value[1] == 'ROC':
            formatters[value] = float_formatB
        elif value[1] == 'ACC':
            formatters[value] = float_format_long
        else:
            formatters[value] = None

    # workaround because the index is not properly formatted in Latex
    metrics = metrics.reset_index()

    metrics.to_latex(filename,
                     float_format=float_format_short,
                     formatters=formatters,
                     escape=False, index=False)


# This method is called by multiple processes
def compute_metrics(index, meta, y_true, y_score, y_pred):
    _logger.debug("Computing metrics...")
    utils.collect_garbage()

    result                 = collections.OrderedDict()  # noqa
    result['ALL']          = compute_metrics_for_mask(index, get_content_mask(meta, 'ALL')      , 'ALL'         , y_true, y_score, y_pred)  # noqa
    result['ITEM_HEAD']    = compute_metrics_for_mask(index, get_content_mask(meta, 'ITEM_HEAD'), 'ITEM_HEAD'   , y_true, y_score, y_pred)  # noqa
    result['ITEM_BODY']    = compute_metrics_for_mask(index, get_content_mask(meta, 'ITEM_BODY'), 'ITEM_BODY'   , y_true, y_score, y_pred)  # noqa
    result['REGISTERED']   = compute_metrics_for_mask(index, get_user_mask(meta, 'REGISTERED')  , 'REGISTERED'  , y_true, y_score, y_pred)  # noqa
    result['UNREGISTERED'] = compute_metrics_for_mask(index, get_user_mask(meta, 'UNREGISTERED'), 'UNREGISTERED', y_true, y_score, y_pred)  # noqa

    result = pd.concat(result.values(), axis=1, keys=result.keys())

    utils.collect_garbage()

    _logger.debug("Computing metrics...done.")

    return result


def get_content_mask(meta, content_type):
    content_types = meta[dataset.CONTENT_TYPE]

    if content_type == 'ALL':
        content_type_mask = np.ones((len(content_types),), dtype=bool)
    elif content_type == 'ITEM_HEAD':
        content_type_mask = np.array(content_types.values == 'TEXT')
    elif content_type == 'ITEM_BODY':
        content_type_mask = np.array(
            (content_types.values == 'STATEMENT') |
            (content_types.values == 'SITELINK')
        )
    else:
        content_type_mask = np.array(content_types.values == content_type)

    return content_type_mask


def get_user_mask(meta, user_type):
    registered_mask = np.asarray(meta[dataset.IS_REGISTERED_USER])

    if user_type == 'REGISTERED':
        return registered_mask
    elif user_type == 'UNREGISTERED':
        return ~registered_mask
    else:
        raise Exception("Unsupported user type '%s'" % str(user_type))


def compute_metrics_for_mask(
        index, mask, mask_name, y_true, y_score, y_pred):

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_score is not None:
        y_score = y_score[mask]

    result = collections.OrderedDict()

    if len(y_true) > 0 and sum(y_true) > 0:
        # Metrics based on prediction

        result['ACC'] = sklearn.metrics.accuracy_score(y_true, y_pred)
        result['P'] = sklearn.metrics.precision_score(y_true, y_pred)
        result['R'] = sklearn.metrics.recall_score(y_true, y_pred)
        result['F'] = sklearn.metrics.f1_score(y_true, y_pred)

        result['RESULTS'] = len(y_pred)
        result['VANDALISM'] = np.sum(y_true)

        # Metrics based on probabilistic score
        if y_score is not None:
            result['ROC'] = sklearn.metrics.roc_auc_score(y_true, y_score)
            fpr, tpr, roc_thresholds = _roc_curve(y_true, y_score)
            precision_values, recall_values, auc_pr = \
                _goadrich_precision_recall_curve(y_true, y_score)
            result['PR'] = auc_pr

            result['fprValues'] = [_format_values(fpr)]
            result['tprValues'] = [_format_values(tpr)]
            result['rocThresholds'] = [_format_values(roc_thresholds)]
            result['precisionValues'] = [_format_values(precision_values)]
            result['recallValues'] = [_format_values(recall_values)]

    else:
        _logger.warn(
            "No positive example for " + str(index) + " and " +
            str(mask_name))

    if len(result.keys()) == 0:
        result['ROC'] = 0
        result['fprValues'] = [np.zeros(2)]
        result['tprValues'] = [np.zeros(2)]
        result['rocThresholds'] = [np.zeros(2)]
        result['PR'] = 0
        result['precisionValues'] = [np.zeros(2)]
        result['recallValues'] = [np.zeros(2)]

    result = pd.DataFrame(result)
    if isinstance(index, str):
        result.index = [index]
    else:
        result.index = index

    return result


def _format_values(values):
    result = ','.join(values)
    return result


#######################################################################
# Downsampling curve (for better performance)
#######################################################################
def _downsample_curve(x, y):
    """Downsample curve randomly."""
    if len(x) != len(y):
        raise Exception("x and y have different length: %d != %d" %
                        (len(x), len(y)))
    else:
        length = len(x)

    sample_size = min(length, config.EVALUATION_MAX_POINTS_ON_CURVE)
    np.random.seed(1)
    idx = np.random.choice(np.arange(length - 1), sample_size - 1, replace=False)

    # always keep the last element because it is special
    # (see sklearn documentation of pr_curve and roc_curve)
    idx = np.append(idx, length - 1)
    idx = np.sort(idx)
    result_x = x[idx]
    result_y = y[idx]

    return result_x, result_y


def _downsample_curve2(x, y, thresholds):
    """Downsample curve by interpolation for x values from 0.01 to 1.00."""
    result_x = np.arange(0.01, 1.005, 0.01)

    f = scipy.interpolate.interp1d(x, y)
    result_y = f(result_x)

    f = scipy.interpolate.interp1d(x, thresholds)
    result_thresholds = f(result_x)

    return result_x, result_y, result_thresholds


def _convert_curve_to_str(x, y, thresholds):
    result_x = ["%.2f" % value for value in x]
    result_y = ["%f" % value for value in y]
    result_thresholds = ["%f" % value for value in thresholds]

    return result_x, result_y, result_thresholds


########################################################################
# Evaluation
########################################################################
def goadrich_pr_auc_score(y_true, y_score):
    _, _, auc_pr = _goadrich_precision_recall_curve(y_true, y_score)
    return auc_pr


def _goadrich_precision_recall_curve(y_true, probas_pred):
    if not ((len(y_true) > 0) and (sum(y_true) > 0)):
        return None, None, None

    precision, recall, _ = \
        sklearn.metrics.precision_recall_curve(y_true, probas_pred)
    precision, recall = _downsample_curve(precision, recall)

    temp_prefix = config.TEMP_PREFIX + __name__ + '-'
    temp_directory = tempfile.TemporaryDirectory(prefix=temp_prefix)

    with open(temp_directory.name + '/pr.csv', 'w', newline='') as csvfile:
        pr_writer = csv.writer(csvfile, delimiter='\t')
        for i in range(0, len(recall)):
            pr_writer.writerow([recall[i], precision[i]])

    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count

    directory_of_this_file = os.path.dirname(os.path.abspath(__file__))

    path_to_jar = os.path.join(directory_of_this_file, "../lib/auc.jar")
    path_to_csv = os.path.join(temp_directory.name, "pr.csv")

    try:
        output = subprocess.check_output(["java",
                                          "-jar",
                                          path_to_jar,
                                          path_to_csv,
                                          "PR",
                                          str(int(pos_count)),
                                          str(int(neg_count))],
                                         universal_newlines=True)
    except subprocess.CalledProcessError as e:
        _logger.error(e.cmd)
        _logger.error(e.output)
        raise

    with open(temp_directory.name + '/pr.csv.spr', 'r', newline='') as csvfile:
        pr_reader = csv.reader(csvfile, delimiter='\t')

        interpolated_precision = []
        interpolated_recall = []
        for row in pr_reader:
            interpolated_recall.append(row[0])
            interpolated_precision.append(row[1])

    match = re.search(
        '(?<=Area Under the Curve for Precision - Recall is )(.*)', output)
    auc_pr = float(match.group(0))

    temp_directory.cleanup()

    return interpolated_precision, interpolated_recall, auc_pr


def _roc_curve(y_true, y_score):
    """Return fpr and tpr values."""
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    fpr, tpr, thresholds = _downsample_curve2(fpr, tpr, thresholds)
    fpr, tpr, thresholds = _convert_curve_to_str(fpr, tpr, thresholds)

    return fpr, tpr, thresholds
