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

import numpy as np
import pandas as pd

import config
import logging

from src import featurelist
from src.dataset import DataSet
from src import constants
from src import evaluationutils
from src import utils
from src import storage

from src.pipeline import loading


TIME_LABEL = 'EVALUATION'
SYSTEM_NAME = 'SYSTEM'

REVISION_ID = 'revisionId'
VANDALISM_SCORE = 'vandalismScore'
TIMESTAMP = 'timestamp'

RENAME_MAPPING = {
    'REVISION_ID': REVISION_ID,
    'VANDALISM_SCORE': VANDALISM_SCORE,
}

EVALUATION_RESULTS_SUFFIX = 'evaluation-results'
EVALUATION_OVER_TIME_SUFFIX = 'evaluation-results-over-time'
EVALUATION_RESULTS_CLEANED_SUFFIX = '-cleaned'


_logger = logging.getLogger()


def main(files):
    utils.print_system_info()
    utils.init_pandas()

    _logger.info("FILES=" + str(files))

    # Load feature file for some statistics
    features = featurelist.get_meta_list() + featurelist.get_label_list()
    df = loading.load_df(files, featurelist.get_columns(features))
    test_set_start = DataSet.get_index_for_revision_id_from_df(
        df, constants.TEST_SET_START)
    tail_set_start = DataSet.get_index_for_revision_id_from_df(
        df, constants.TAIL_SET_START)
    df = df[test_set_start:tail_set_start]
    data = DataSet()
    data.set_meta(df.iloc[:, :-1])
    data.set_Y(df.iloc[:, -1].astype(np.float32))
    data.set_X(np.zeros((len(data), 1)))
    _logger.debug("Length of data: " + str(len(data)))

    # Load scores
    scores = pd.DataFrame()
    scores[REVISION_ID] = data.get_revision_ids()
    scores.set_index(REVISION_ID, inplace=True)

    for team, score_file in files['teams'].items():
        team_scores = load_vandalism_scores(score_file)
        team_scores.set_index(REVISION_ID, inplace=True)
        scores[team] = team_scores[VANDALISM_SCORE]

    scores.dropna(inplace=True)
    if len(data) != len(scores):
        raise Exception(
            "number of scores does not fit test set size: " +
            "len(data)={0} but len(scores)={1}".format(len(data), len(scores)))

    _logger.debug("Length of scores: " + str(len(data)))

    # Evaluate teams
    meta_scores = compute_meta_scores(scores)
    scores = pd.concat([scores, meta_scores], axis=1)

    evaluate_teams(scores, data, save_scores=['META'])
    evaluate_teams_over_time(scores, data, EVALUATION_OVER_TIME_SUFFIX)

    scores, data = clean_data(scores, data)
    evaluate_teams(scores, data, suffix=EVALUATION_RESULTS_CLEANED_SUFFIX)


def evaluate_teams(scores, data, suffix='', save_scores=[]):
    metrics = pd.DataFrame()
    for team in scores:
        prob = scores[team].values
        pred = evaluationutils.get_pred_from_prob(prob)
        save_prob = (team in save_scores)
        m = evaluationutils.evaluate(team + suffix, pred, prob, data, save_prob)
        evaluationutils.print_metrics(m, append_global=True)
        metrics = metrics.append(m)

    metrics = metrics.sort_values(by=('ALL', 'ROC'), ascending=False)
    metrics = metrics[evaluationutils.COLUMNS]

    evaluationutils.print_metrics(
        metrics, EVALUATION_RESULTS_SUFFIX + suffix, append_global=False)


def evaluate_teams_over_time(scores, data, suffix):
    metrics = pd.DataFrame()
    for team in scores:
        prob = scores[team].values
        pred = evaluationutils.get_pred_from_prob(prob)

        start_time = pd.Timestamp('2016-5-1T00:00:00Z')
        end_time   = pd.Timestamp('2016-7-1T00:00:00Z')
        time_offset = pd.DateOffset(weeks=1)

        time = start_time

        while time < end_time:
            mask = np.asarray(
                (data.get_meta()['timestamp'] >= time) &
                (data.get_meta()['timestamp'] < time + time_offset)
            )
            cur_data = data[mask]
            cur_pred = pred[mask]
            cur_prob = prob[mask]

            label = "{0} {1:%Y-%m-%d}".format(team, time)

            m = evaluationutils.evaluate(
                label, cur_pred, cur_prob, cur_data, save_prob=False)
            evaluationutils.print_metrics(m, append_global=True)
            metrics = metrics.append(m)

            time += time_offset

        evaluation_results = metrics[evaluationutils.COLUMNS]
        evaluationutils.print_metrics(evaluation_results,
                                      suffix,
                                      append_global=False)


def compute_meta_scores(scores):
    scores = scores.mean(axis=1)

    result = pd.DataFrame()
    result['META'] = scores

    return result


# for investigation of the issue plot
# (data.get_meta()[(data.get_Y() == 1) &
# (data.get_meta()['revisionId'] > 349000000) &
# (data.get_meta()['revisionId'] < 350000000) ]['revisionId']
# ).hist(bins=100)
# see https://www.wikidata.org/w/index.php?title=Special:Contributions&offset=20160621090000&limit=500&contribs=user&target=Innocent+bystander&namespace=&tagfilter=
#
def clean_data(scores, data):
    start_time = pd.Timestamp('2016-6-19T00:00:00Z')
    end_time   = pd.Timestamp('2016-6-20T00:00:00Z')
    user_name = 'Innocent bystander'

    outliers = np.asarray(data.get_Y() == 1)  # rollback reverted
    outliers &= np.asarray(data.get_meta()['userName'] == user_name)
    outliers &= (
        np.asarray(data.get_meta()['timestamp'] >= start_time) &
        np.asarray(data.get_meta()['timestamp'] < end_time)
    )

    _logger.debug(
        'outliers: Innocent bystander reverted on June 19: %d' %
        outliers.sum())

    _logger.debug(
        'outliers: first revision id: %d' %
        data.get_meta()[REVISION_ID][np.where(outliers)[0][0]]  # 349162103
    )
    _logger.debug(
        'outliers: last revision id: %d' %
        data.get_meta()[REVISION_ID][np.where(outliers)[0][-1]]  # 349183610
    )
    data.get_meta()[REVISION_ID][outliers]. \
        to_csv(config.OUTPUT_PREFIX + '_outliers' + '.csv', index=False)
    _logger.debug(
        'outliers:\n' +
        str(data.get_meta()['revisionTags'][outliers].value_counts().head())
    )
    _logger.debug(
        'outliers:\n' +
        str(data.get_meta()['revisionHashTag'][outliers].value_counts().head())
    )
    _logger.debug(
        'outliers:\n' +
        str(data.get_meta()['property'][outliers].value_counts().head())
    )
    _logger.debug('reverted: ' + str(np.sum(data.get_Y())))

    scores_cleaned = scores[~outliers].copy()
    data_cleaned = data[~outliers].shallow_copy()

    return scores_cleaned, data_cleaned


def load_vandalism_scores(filepath):
    scores = storage.load_predictions_from_path(filepath)
    scores.rename(columns=RENAME_MAPPING, inplace=True)
    return scores
