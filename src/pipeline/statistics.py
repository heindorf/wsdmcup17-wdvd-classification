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

from collections import OrderedDict
import logging

import numpy as np
import pandas as pd

import config
from src import constants
from src.dataset import DataSet

_logger = logging.getLogger()

# Use thousand separator and no decimal points
_FLOAT_FORMAT = '{:,.0f}'.format

# columns
REVISION_ID = 'revisionId'
SESSION_ID = 'revisionSessionId'
CONTENT_TYPE = 'contentType'
ROLLBACK_REVERTED = 'rollbackReverted'
ITEM_ID = 'itemId'
USER_NAME = 'userName'
REVISION_ACTION = 'revisionAction'
TIMESTAMP = 'timestamp'

REVISIONT_TAGS = 'revisionTags'
LANGUAGE_WORD_RATIO = 'languageWordRatio'
REVISION_LANGUAGE = 'revisionLanguage'
USER_COUNTRY = 'userCountry'


def compute_statistics(data):
    _logger.info("Computing statistics...")

    _logger.debug(
        data['revisionAction']
        .str
        .cat(data['revisionSubaction'], sep='_', na_rep='na')
        .value_counts())

    _compute_feature_statistics(data)

    _compute_corpus_statistics(data)
    _compute_corpus_statistics_over_time(data)
    _compute_dataset_statistics(data)
    _compute_session_statistics(data)
    _compute_backpressure_statistics(data)

    # computes some statistics about selected features
    # _compute_special_feature_statistics(data)

    _logger.info("Computing statistics... done.")


def _compute_feature_statistics(data):
    _logger.debug("Computing descriptive statistics...")

    data.describe(include='all').to_csv(
        config.OUTPUT_PREFIX + "_feature_statistics.csv")

    _logger.debug("Computing descriptive statistics... done.")


def _compute_corpus_statistics(data):
    """Compute statistics for the whole corpus.

    Evaluate corpus in terms of total unique users, items, sessions,
    and revisions with a breakdown by content type and by vandalism
    status (vandalism/non-vandalism).
    """
    def compute_data_frame(data):
        head_mask = data[CONTENT_TYPE] == 'TEXT'
        stmt_mask = (data[CONTENT_TYPE] == 'STATEMENT')
        sitelink_mask = (data[CONTENT_TYPE] == 'SITELINK')
        body_mask = (stmt_mask | sitelink_mask)

        result = OrderedDict()
        result['Entire corpus'] = compute_column_group(data)
        result['Item head'] = compute_column_group(data[head_mask])
        result['Item body'] = compute_column_group(data[body_mask])
        # result['STATEMENT'] = compute_column_group(data[stmt_mask])
        # result['SITELINK'] = compute_column_group(data[sitelink_mask])

        result = pd.concat(result, axis=1, keys=result.keys())
        return result

    def compute_column_group(data):
        vandalism_mask = data[ROLLBACK_REVERTED].astype(np.bool)
        regular_mask = ~vandalism_mask

        result = OrderedDict()
        result['Total'] = compute_column(data)
        result['Vandalism'] = compute_column(data[vandalism_mask])
        result['Regular'] = compute_column(data[regular_mask])

        result = pd.concat(result, axis=1, keys=result.keys())
        return result

    def compute_column(data):
        result = pd.Series()

        result['Revisions'] = data[REVISION_ID].nunique()
        result['Sessions'] = data[SESSION_ID].nunique()
        result['Items'] = data[ITEM_ID].nunique()
        result['Users'] = data[USER_NAME].nunique()

        return result

    def compute_actions(data):
        vandalism_counts = \
            data[data[ROLLBACK_REVERTED]][REVISION_ACTION].value_counts()
        vandalism_counts.name = 'vandalism_count'

        total_counts = data[REVISION_ACTION].value_counts()
        total_counts.name = 'total_count'

        counts = pd.concat([vandalism_counts, total_counts], axis=1)
        counts.sort_values('vandalism_count', inplace=True, ascending=False)

        _logger.debug(
            'Counts: \n' +
            str(counts)
        )

    statistics = compute_data_frame(data)
    statistics.to_csv(config.OUTPUT_PREFIX + "_corpus_statistics.csv")

    statistics = _round_to_thousands(statistics)
    statistics.to_latex(
        config.OUTPUT_PREFIX + "_corpus_statistics.tex",
        float_format=_FLOAT_FORMAT)

    _logger.info(statistics)


def _compute_corpus_statistics_over_time(data):
    df = data.copy()
    df = data.loc[:, [TIMESTAMP, CONTENT_TYPE, ROLLBACK_REVERTED]]
    df = pd.get_dummies(df, columns=[CONTENT_TYPE])

    df = df.rename(columns={'contentType_TEXT': 'TEXT',
                            'contentType_STATEMENT': 'STATEMENT',
                            'contentType_SITELINK': 'SITELINK',
                            'contentType_MISC': 'MISC',
                            'rollbackReverted': 'VANDALISM'})
    df['TEXT_VANDALISM']      = (df['TEXT'] & df['VANDALISM'])
    df['STATEMENT_VANDALISM'] = (df['STATEMENT'] & df['VANDALISM'])
    df['SITELINK_VANDALISM']  = (df['SITELINK'] & df['VANDALISM'])
    df['MISC_VANDALISM']      = (df['MISC'] & df['VANDALISM'])
    df['REVISIONS'] = 1
    df['VANDALISM'] = df['VANDALISM'].astype(np.bool)

    df.set_index(TIMESTAMP, inplace=True)

    df = df[['REVISIONS', 'VANDALISM',
             'TEXT', 'TEXT_VANDALISM',
             'STATEMENT', 'STATEMENT_VANDALISM',
             'SITELINK', 'SITELINK_VANDALISM',
             'MISC', 'MISC_VANDALISM']]

    grouped = df.groupby(pd.TimeGrouper(freq='M'))

    result = grouped.sum()

    result.to_csv(config.OUTPUT_PREFIX + "_corpus_statistics_over_time.csv")


def _compute_dataset_statistics(data):
    """
    Compute dataset statistics for training, validation and test set.

    Evaluate datasets in terms of time period covered, revisions, sessions,
    items, and users.
    """
    def compute_data_frame(data):
        _logger.debug("Splitting statistics...")
        training_set_start_index = 0  # compute statistics from start of dataset
        validation_set_start_index = \
            DataSet.get_index_for_revision_id_from_df(data, constants.VALIDATION_SET_START)
        test_set_start_index = \
            DataSet.get_index_for_revision_id_from_df(data, constants.TEST_SET_START)
        tail_set_start_index = \
            DataSet.get_index_for_revision_id_from_df(data, constants.TAIL_SET_START)

        training_set = data[training_set_start_index:validation_set_start_index]
        validation_set = data[validation_set_start_index:test_set_start_index]
        test_set = data[test_set_start_index:tail_set_start_index]

        result = []
        result.append(compute_splitting_statistics_row(training_set, 'Training'))
        result.append(compute_splitting_statistics_row(validation_set, 'Validation'))
        result.append(compute_splitting_statistics_row(test_set, 'Test'))

        result = pd.concat(result, axis=0)
        return result

    def compute_splitting_statistics_row(data, label):
        result = pd.Series()
        result['From'] = data[TIMESTAMP].min()
        result['To'] = data[TIMESTAMP].max()
        result['Vandalism Revisions'] = data[ROLLBACK_REVERTED].sum()
        result['Revisions'] = data[REVISION_ID].nunique()
        result['Sessions'] = data[SESSION_ID].nunique()
        result['Items'] = data[ITEM_ID].nunique()
        result['Users'] = data[USER_NAME].nunique()

        result = result.to_frame().transpose()

        result.index = [label]

        return result

    result = compute_data_frame(data)

    result.to_csv(config.OUTPUT_PREFIX + "_dataset_statistics.csv")

    columns_to_round = [
        'Vandalism Revisions',
        'Revisions',
        'Sessions',
        'Items',
        'Users']
    result[columns_to_round] = _round_to_thousands(result[columns_to_round])
    date_format = '%b %e, %Y'
    result['From'] = result['From'].dt.strftime(date_format)
    result['To'] = result['To'].dt.strftime(date_format)
    result.to_latex(config.OUTPUT_PREFIX + "_dataset_statistics.tex",
                    float_format=_FLOAT_FORMAT)

    _logger.debug("Splitting statistics... done.")


def _compute_session_statistics(data):
    """Compute statistics about revision groups (a.k.a. sessions)."""
    groupsize = data.groupby(by=SESSION_ID).size()
    groupsize.name = 'groupsize'
    groupsize = groupsize.to_frame()
    groupsize[SESSION_ID] = groupsize.index

    # merge revisions with groupsizes
    joined_groupsize = data[[REVISION_ID, SESSION_ID]].join(
        groupsize, on=SESSION_ID, how='left', lsuffix='_left', rsuffix='_right')

    # distribution of group sizes among revisions)
    counts = joined_groupsize['groupsize'].value_counts()

#     logger.info("Distribution of group sizes of revisions:\n " + str(counts))

    # revisions which are not alone in their group
    part_of_larger_group = sum(counts[counts.index > 1])
    all_revisions = len(data)
    _logger.info(
        "Fraction of revisions which are not alone in their session: " +
        "%d / %d = %.2f" %
        (part_of_larger_group, all_revisions, part_of_larger_group / all_revisions))


def _compute_special_feature_statistics(data):  # noqa
    def compute_feature_statistics_main(data):
        compute_vandalism_probability(
            data[REVISIONT_TAGS],
            data[ROLLBACK_REVERTED],
            get_revision_tag_mapping(data))
        compute_vandalism_probability(
            data[LANGUAGE_WORD_RATIO],
            data[ROLLBACK_REVERTED],
            get_language_word_ratio_mapping(data))
        compute_vandalism_probability(
            data[REVISION_LANGUAGE],
            data[ROLLBACK_REVERTED],
            get_revision_language_mapping(data))
        compute_vandalism_probability(
            data[USER_COUNTRY],
            data[ROLLBACK_REVERTED])

    def compute_vandalism_probability(data, y_true, mapping=None, name=None):
        TOP_K = 5

        if name is None:
            name = data.name

        data = data.astype(str)

        _logger.debug("Computing vandalism probability for feature %s", name)

        result = compute_vandalism_probability2(data, y_true)
        result.to_csv(config.OUTPUT_PREFIX + "_feature_%s.csv" % (name))

        if mapping is not None:
            result = apply_mapping(result, mapping)
            result.to_csv(config.OUTPUT_PREFIX + "_feature_%s_mapped.csv" % (name))

        nonNan = result.index.tolist()
        if 'nan' in nonNan:
            nonNan.remove('nan')
        top = nonNan[:TOP_K]
        truncated_mapping = {value: value if value in top else 'misc' for value in nonNan}
        truncated_mapping['nan'] = 'nan'

        truncated_result = apply_mapping(result, truncated_mapping)

        # insert first row of table which shows statistics for nonNan values
        nonNan_mapping = {value: 'nonNan' for value in nonNan}
        nonNan_mapping['nan'] = np.NaN  # the mapping for 'nan' should be undefined such that it does not appear in the result
        nonNan_result = apply_mapping(result, nonNan_mapping)
        truncated_result = nonNan_result.append(truncated_result)

        truncated_result.to_csv(
            config.OUTPUT_PREFIX + "_feature_%s_truncated.csv" % (name))

        truncated_result[['vandalismRevisions', 'totalRevisions']] = \
            _round_to_thousands(
                truncated_result[['vandalismRevisions', 'totalRevisions']])

        truncated_result['vandalismProbability'] = \
            truncated_result['vandalismProbability'] * 100

        truncated_result.to_latex(
            config.OUTPUT_PREFIX + "_feature_%s_truncated.tex" % (name),
            float_format=_FLOAT_FORMAT,
            formatters={'vandalismProbability': '{:,.2f}%'.format})

    def compute_vandalism_probability2(data, y_true):
        result = y_true.groupby(data).sum().to_frame()
        result.columns = ["vandalismRevisions"]
        result['totalRevisions'] = y_true.groupby(data).size()
        result['vandalismProbability'] = (result['vandalismRevisions'] /
                                          result['totalRevisions'])
        result.sort_values(by='vandalismRevisions', ascending=False, inplace=True)

        return result

    def apply_mapping(df, mapping):
        series = df.index.to_series()

        mapped = series.replace(mapping)

        # groupby ignores nan values (hence, nan must be encoded as string 'nan')
        result = df.groupby(mapped).sum()
        result['vandalismProbability'] = (result['vandalismRevisions'] /
                                          result['totalRevisions'])
        result.sort_values(by='vandalismRevisions', ascending=False, inplace=True)

        # move 'misc' and 'nan' values to the end
        result = move_row_to_end(result, 'misc')
        result = move_row_to_end(result, 'nan')

        return result

    def update_vandalism_probability(df):
        df['vandalismProbability'] = (df['vandalismRevisions'] /
                                      df['totalRevisions'])

    def move_row_to_end(df, idx_value):
        result = df.copy()

        idx = result.index
        if idx_value in idx:
            loc = idx.get_loc(idx_value)
            idx = idx.delete(loc)
            idx = idx.insert(idx.size, idx_value)
            result = result.reindex(idx)

        return result

    def get_revision_language_mapping(data):
        language_values = data[REVISION_LANGUAGE].astype('str').unique()
        project_suffixes = ['wiki', 'wikisource', 'wikiquote', 'wikinews', 'wikivoyage']
        language_variant_prefixes = ['de-', 'en-']  # Further information: https://www.wikidata.org/wiki/Wikidata:Requests_for_comment/Labels_and_descriptions_in_language_variants

        language_dict = {}
        for language_value in language_values:
            key = language_value
            value = language_value

            # make empty string count as revision without language
            if value == '':
                value = 'nan'

            # make enwiki, enwikisource, enwikiquote, ... all count as English
            for suffix in project_suffixes:
                if language_value.endswith(suffix):
                    value = language_value[:-len(suffix)]

            # make en, en, ca, en-gb, ...all count as English
            # make de, de-at, de-ch, ... all count as German
            for language_variant in language_variant_prefixes:
                if value.startswith(language_variant):
                    value = value[:len(language_variant) - 1]

            language_dict[key] = value

        return language_dict

    # Further information:
    # Tags: https://www.wikidata.org/wiki/Special:Tags
    # Abuse Filter: https://www.wikidata.org/wiki/Special:AbuseFilter
    def get_revision_tag_mapping(data):
        revision_tag_values = data[REVISIONT_TAGS].astype('str').unique()
        editing_tools = ['OAuth CID', 'HHVM']
        revision_tag_dict = {}

        for revision_tag_value in revision_tag_values:
            if any(tool_str in revision_tag_value for tool_str in editing_tools):
                revision_tag_dict[revision_tag_value] = 'EDITING TOOLS'
            elif revision_tag_value == '':
                revision_tag_dict[revision_tag_value] = 'nan'
            else:
                revision_tag_dict[revision_tag_value] = 'ABUSE FILTER'

        return revision_tag_dict

    def get_language_word_ratio_mapping(data):
        word_ratio_values = data[LANGUAGE_WORD_RATIO].astype('str').unique()
        word_ratio_dict = {}

        for value in word_ratio_values:
            if float(value) > 0:
                word_ratio_dict[value] = True
            elif (float(value) == -1.0) or value == 'nan':
                word_ratio_dict[value] = 'nan'
            else:
                word_ratio_dict[value] = False

        return word_ratio_dict

    compute_feature_statistics_main(data)


def _compute_backpressure_statistics(data):
    # Restrict computation to test dataset
    test_set_start_index = \
        DataSet.get_index_for_revision_id_from_df(data, constants.TEST_SET_START)
    tail_set_start_index = \
        DataSet.get_index_for_revision_id_from_df(data, constants.TAIL_SET_START)
    data = data[test_set_start_index:tail_set_start_index]

    data = data[[
        REVISION_ID,
        ITEM_ID,
        USER_NAME,
        REVISION_ACTION,
        ROLLBACK_REVERTED]]

    REVISION_ID_INDEX = 0  # noqa
    ITEM_ID_INDEX = 1
    USER_NAME_INDEX = 2
    REVISION_ACTION_INDEX = 3
    ROLLBACK_REVERTED_INDEX = 4  # noqa

    data = data.values

    result = np.full(len(data), np.nan)
    revealed = pd.DataFrame()

    for i in range(len(data)):
        user_name = data[i][USER_NAME_INDEX]
        item_id = data[i][ITEM_ID_INDEX]

        prev_rev = data[i]

        for j in range(i + 1, min(len(data), i + 16)):
            rev = data[j]

            if rev[ITEM_ID_INDEX] == item_id:
                # Rollback within same session (same item id and same user name)
                if rev[USER_NAME_INDEX] == user_name:
                    if rev[REVISION_ACTION_INDEX] == 'rollback':
                        result[i] = True
                        revealed = revealed.append(
                            pd.Series(prev_rev), ignore_index=True)
                        break
                # Rollback at beginning of next session
                else:
                    if rev[REVISION_ACTION_INDEX] == 'rollback':
                        result[i] = True
                        revealed = revealed.append(
                            pd.Series(prev_rev), ignore_index=True)
                        break
                    else:
                        result[i] = False
                        revealed = revealed.append(
                            pd.Series(prev_rev), ignore_index=True)
                        break

    n_revisions = result.size
    n_revealed_total = (~(np.isnan(result))).sum()
    n_revealed_regular = (result == True).sum()  # noqa
    n_revealed_vandalism = (result == False).sum()  # noqa

    _logger.info('n_revisions: ' + str(n_revisions))
    _logger.info('n_revealed_total: ' + str(n_revealed_total))
    _logger.info('n_revealed_vandalism: ' + str(n_revealed_vandalism))
    _logger.info('n_revealed_regular: ' + str(n_revealed_regular))


def _round_to_thousands(statistics):
    statistics = statistics / 1000  # numbers in thousand
    statistics = statistics.round()

    return statistics
