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

import itertools
import logging

import pandas as pd
from sklearn import ensemble
from sklearn.externals.joblib import Parallel, delayed

import config
from src import evaluationutils


_logger = logging.getLogger()


########################################################################
# Feature Ranking
########################################################################
def rank_features(training, validation):
    _logger.info("Ranking features...")

    metrics = _compute_metrics_for_single_features(training, validation)

    group_metrics = _compute_metrics_for_feature_groups(training, validation)
    metrics = pd.concat([metrics, group_metrics], axis=0)

    _output_sorted_by_group(
        validation.get_time_label(), validation.get_system_name(),
        metrics, validation.get_group_names(), validation.get_subgroup_names())

    _logger.info("Ranking features... done.")


def _compute_metrics_for_single_features(training, validation):
    """Return a Pandas data frame with metrics for every single feature."""
    arguments = []
    for feature in validation.get_features():
        # each feature name is a tuple itself and
        # here we take the last element of this tuple
        training2 = training.select_feature(feature[-1])
        validation2 = validation.select_feature(feature[-1])
        argument = (training2, validation2, feature, )
        arguments.append(argument)

    result_list = Parallel(n_jobs=config.FEATURE_RANKING_N_JOBS,
                           backend='multiprocessing')(
        delayed(_compute_feature_metrics_star)(x) for x in arguments)

    result = pd.concat(result_list, axis=0)

    return result


def _compute_metrics_for_feature_groups(training, validation):
    arguments = []
    for subgroup in validation.get_subgroups():
        # each feature name is a tuple itself and here we take the last
        # element of this tuple
        training2 = training.select_subgroup(subgroup[-1])
        validation2 = validation.select_subgroup(subgroup[-1])
        argument = (training2, validation2, subgroup + ('ALL', ), )
        arguments.append(argument)

    for group in validation.get_groups():
        training2 = training.select_group(group)
        validation2 = validation.select_group(group)
        argument = (training2, validation2, (group, 'ALL', 'ALL'),)
        arguments.append(argument)

    result_list = Parallel(n_jobs=config.FEATURE_RANKING_N_JOBS,
                           backend='multiprocessing')(
        delayed(_compute_feature_metrics_star)(x) for x in arguments)

    result = pd.concat(result_list, axis=0)
    return result


# This method is called by multiple processes
def _compute_feature_metrics_star(args):
    return _compute_feature_metrics(*args)


# This method is called by multiple processes
def _compute_feature_metrics(training, validation, label):
    _logger.debug("Computing metrics for %s..." % str(label))

    index = pd.MultiIndex.from_tuples(
        [label], names=['Group', 'Subgroup', 'Feature'])

    _logger.debug("Using random forest...")
    clf = ensemble.RandomForestClassifier(random_state=1, verbose=0, n_jobs=-1)

    evaluationutils.fit(clf, training, index)

    y_pred, y_score = evaluationutils.predict(clf, validation, index)
    validation_result = evaluationutils.compute_metrics(
        index, validation.get_metrics_meta(), validation.get_Y(), y_score, y_pred)

    # computing the feature metrics on the training set is useful for
    # identifying overfitting
    training_y_pred, training_y_score = evaluationutils.predict(clf, training, index)
    training_result = evaluationutils.compute_metrics_for_mask(
        index, evaluationutils.get_content_mask(training.get_metrics_meta(), 'ALL'), 'ALL',
        training.get_Y(), training_y_score, training_y_pred)
    training_result.columns = list(itertools.product(
        ['TRAINING'], training_result.columns.values))

    result = pd.concat([validation_result, training_result], axis=1)

    return result


def _output_sorted_by_auc_pr(time_label, system_name, metrics):
    """Output the metrics sorted by area under precision-recall curve."""
    _logger.debug("output_sorted_by_auc_pr...")
    metrics.sort_values([('ALL', 'PR')], ascending=False, inplace=True)
    metrics.to_csv(config.OUTPUT_PREFIX + "_" + time_label + "_" +
                   system_name + "_feature_ranking.csv")

    latex = metrics.loc[:, evaluationutils.COLUMNS]
    # latex.reset_index(drop=True, inplace=True)
    latex.to_latex(config.OUTPUT_PREFIX + "_" + time_label + "_" +
                   system_name + "_feature_ranking.tex", float_format='{:.3f}'.format)

    n_features = min(9, len(metrics) - 1)
    selection = metrics.iloc[0:n_features] \
                       .loc[:, [('ALL', 'Feature'), ('ALL', 'PR')]]
    _logger.info("Top 10 for all content\n" +
                 (selection.to_string(float_format='{:.4f}'.format)))
    _logger.debug("output_sorted_by_auc_pr... done.")


def _output_sorted_by_group(
        time_label, system_name, metrics, group_names, subgroup_names):
    """Output the metrics sorted by group and by PR-AUC within a group."""
    _logger.debug('_output_sorted_by_group...')

    sort_columns = ['_Group', '_Subgroup', '_Order', '_Feature']
    ascending_columns = [True, True, False, True]
    metrics['_Group'] = metrics.index.get_level_values('Group')
    metrics['_Subgroup'] = metrics.index.get_level_values('Subgroup')
    metrics['_Feature'] = metrics.index.get_level_values('Feature')

    subgroup_names = ['ALL'] + subgroup_names

    # Define the order of groups and subgroups
    metrics['_Group'] = metrics['_Group'].astype('category').cat.set_categories(
        group_names, ordered=True)
    metrics['_Subgroup'] = metrics['_Subgroup'].astype('category').cat.set_categories(
        subgroup_names, ordered=True)

    # Sort the features by AUC_PR and make sure the subgroup is always shown
    # before the single features
    metrics['_Order'] = metrics[('ALL', 'PR')]
    # without this line, the following line causes a PerformanceWarning
    metrics.sort_index(inplace=True)
    metrics.loc[(metrics['_Feature'] == 'ALL'), '_Order'] = 1.0

    metrics.sort_values(by=sort_columns,
                        ascending=ascending_columns, inplace=True)

    metrics = metrics.drop(sort_columns, axis=1)

    metrics.to_csv(config.OUTPUT_PREFIX + "_" + time_label + "_" +
                   system_name + "_feature_groups.csv")

    latex_names = metrics.apply(_compute_latex_name, axis=1)
    metrics.set_index(latex_names, inplace=True)

    metrics = evaluationutils.remove_columns(metrics, evaluationutils.CURVES)
    metrics = evaluationutils.remove_columns(metrics, evaluationutils.STATISTICS)
    evaluationutils.print_metrics_to_latex(
        metrics, config.OUTPUT_PREFIX + "_" + time_label + "_" +
        system_name + "_feature_groups.tex")

    _logger.debug('_output_sorted_by_group... done.')


def _compute_latex_name(row):
    group = row.name[0]
    subgroup = row.name[1]
    feature = row.name[2]

    # Is group?
    if subgroup == 'ALL' and feature == 'ALL':
        result = "\\quad %s" % group
    # Is subgroup?
    elif feature == 'ALL':
        result = "\\quad\quad %s" % subgroup
    # Is feature?
    else:
        result = "\\quad\\quad\\quad %s" % feature

    return result
