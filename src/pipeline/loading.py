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
import os

import numpy as np
import pandas as pd

import config
from src import featurelist
from src import transformers
from src import utils

_logger = logging.getLogger()

TRUE_FALSE = {'T': True, 'F': False}

ROLLBACK_REVERTED2 = 'rollbackReverted'


def load_df(files, columns):
    """Load features from disk and return a Pandas data frame."""
    _logger.info("Loading data...")

    data = _read_cached_data(files, columns)

    _logger.info("Loading data... done.")

    return data


def _read_cached_data(files, columns):
    utils.collect_garbage()

    if config.LOADING_USE_MEMORY_CACHE:
        data = _read_files_with_memory_cache(files, columns)
    elif config.LOADING_USE_DISK_CACHE:
        data = _read_files_with_disk_cache(files, columns)
    else:
        data = _read_files(files, columns)

    memory_usage = data.memory_usage(index=True, deep=True)
    memory_usage['totalMemoryUsage'] = memory_usage.sum()
    _logger.debug("Memory usage of loaded data:\n%s" %
                  str(memory_usage.sort_values(ascending=False)))

    return data


def _read_files_with_memory_cache(files, columns):
    global _data_cache

    if '_data_cache' not in globals():
        _logger.debug("reading csv...")
        # read all columns
        data = _read_files(files, None)
        _logger.debug("reading csv... done.")

        _data_cache = data

    _logger.debug("Copying data in memory...")
    data = _data_cache.copy()

    memory_usage = data.memory_usage(index=True, deep=True)
    memory_usage['totalMemoryUsage'] = memory_usage.sum()
    _logger.debug("Memory usage of cache:\n%s" %
                  str(memory_usage.sort_values(ascending=False)))

    if columns is not None:
        data = data[columns]

    _logger.debug("Copying data in memory... done.")

    return data


def _read_files_with_disk_cache(files, columns):
    cache_file = os.path.basename(files) + '.cached.p'
    if not os.path.exists(cache_file):
        _logger.debug("Reading csv...")
        # read all columns
        data = _read_files(files, None)
        _logger.debug("Reading csv... done.")

        _logger.debug('Pickling...')
        data.to_pickle(cache_file)
        _logger.debug('Pickling...done')

    _logger.debug('Unpickling...')
    data = pd.read_pickle(cache_file)
    _logger.debug('Unpickling...done.')

    data = data[columns]

    return data


def _read_files(files, columns):
    feature_columns = columns[:-1] if columns is not None else None
    truth_columns = columns[:1] + columns[-1:] if columns is not None else None

    inv_map = {v: k for k, v in featurelist.RENAME_MAPPING.items()}
    if truth_columns is not None:
        truth_columns = \
            [(inv_map[col] if (col in inv_map) else col) for col in truth_columns]

    data = _concat_files(files['features'], feature_columns)
    truth = _concat_files(files['truth'], truth_columns)

    data = data.join(truth)

    _check_completeness(data)
    data.reset_index(inplace=True)

    return data


def _concat_files(files, columns):
    data = pd.DataFrame()

    for file in files:
        _logger.debug("Reading data from file %s..." % file)
        df = read_data(file, columns)
        df.set_index('revisionId', inplace=True)

        if data.empty:
            data = df
        else:
            data = pd.concat([data, df])

        data = _convert_to_categories(data)
        data = _convert_to_datetime(data)
        _check_completeness_of_data_types(data.columns, featurelist.DATA_TYPES)

        _logger.debug("Reading data from file... done.")

    # Sorts the data by revision id which simplifies
    # the implementation of some transformers later.

    utils.collect_garbage()

    _logger.debug('sorting...')
    data.sort_index(inplace=True)
    _logger.debug('sorting...done')

    utils.collect_garbage()

    return data


def read_data(filepath_or_buffer, usecols=None):
    """Read the given features from the given file (either csv or csv.bz2)."""
    data = pd.read_csv(
        filepath_or_buffer,
        quotechar='"',
        low_memory=True,
        keep_default_na=False,
        na_values=['', u'\ufffd'],
        dtype=get_pre_read_data_types(),
        usecols=usecols,
        engine='c',
    )

    data = _convert_bool_columns(data)

    data = _rename_columns(data)

    return data


pre_read_data_types = None


def get_pre_read_data_types():
    global pre_read_data_types
    if pre_read_data_types is None:
        _logger.debug("precompute data types")
        data_types = featurelist.get_data_types()

        # Pandas does not support reading categorical columns from CSV file.
        # Hence, we first read them as string and convert them later.
        replaced_data_types = data_types.copy()
        replaced_data_types = _replace_by_str('category', replaced_data_types)
        # Pandas does not support reading Boolean columns that contain NA values.
        # Hence, we first read them as string and convert them later.
        replaced_data_types = _replace_by_str(np.bool, replaced_data_types)
        replaced_data_types = _replace_by_str('datetime', replaced_data_types)

        pre_read_data_types = replaced_data_types
    return pre_read_data_types


def _convert_bool_columns(data):
    bool_columns = _get_affected_columns(
        data.columns, np.bool, featurelist.DATA_TYPES)

    for column in bool_columns:
        data[column] = data[column].map(TRUE_FALSE)

    return data


def _rename_columns(data):
    data.rename(columns=featurelist.RENAME_MAPPING, copy=False, inplace=True)
    return data


def _convert_to_categories(data):
    _logger.debug("Categorizing data...")

    cat_columns = _get_affected_columns(
        data.columns, 'category', featurelist.DATA_TYPES)

    transformer = transformers.CategoryTransformer(cat_columns)
    data = transformer.fit_transform(data)
    return data


def _convert_to_datetime(data):
    _logger.debug("Converting datetime...")
    datetime_columns = \
        _get_affected_columns(data.columns, 'datetime', featurelist.DATA_TYPES)

    for column in datetime_columns:
        data.loc[:, column] = pd.to_datetime(data.loc[:, column],
                                             format='%Y-%m-%dT%H:%M:%SZ',
                                             utc=True)
    return data


def _check_completeness_of_data_types(columns, data_types):
    """Check whether a datatype has been specified for every feature."""
    for column in columns:
        if column not in data_types.keys():
            _logger.warn("No data type specified for column " + column)


def _check_completeness(data):
    rollback_nan = data[ROLLBACK_REVERTED2].isnull().sum()
    if rollback_nan > 0:
        _logger.warn("Column {0}: {1} of of {2} valures are NaN".format(
            ROLLBACK_REVERTED2, rollback_nan, len(data)))


def _get_affected_columns(columns, data_type, data_types):
    """Return columns affected by given data types.

    Given a list of columns and a dictionary of data_types,
    finds all columns that have the datatype data_type.
    Returns all columns having this datatype.
    """
    affected_columns = []
    for column in columns:
        if data_types[column] == data_type:
            affected_columns = affected_columns + [column]

    return affected_columns


def _replace_by_str(data_type_to_replace, data_types):
    """Replace the given data types with the type string.

    Returns a new datatype dictionary in which all occurrences of
    data_type_to_replace have been replaced by type string.

    For example, this function is used to replace the datatypes
    'category' and 'np.bool' by 'str' before reading the csv file.
    """
    # replaces data_type_to_replace by str in data_types
    new_data_types = data_types.copy()
    for key, value in data_types.items():
        if value == data_type_to_replace:
            new_data_types[key] = str

    return new_data_types


# Only used for debugging when reading CSV file fails
def _binary_search(filename, columns):
    data_types = featurelist.get_data_types()

    _logger.debug("Binary search...")

    skiprows = 0
    nrows = 2**24

    while nrows >= 1:
        _logger.debug("skiprows: %s, nrows %s" % (str(skiprows), str(nrows)))

        error = False
        try:
            pd.read_csv(filename,
                        quotechar='"',
                        low_memory=True,
                        keep_default_na=False,
                        na_values=['NA', 'NaN', 'null', u'\ufffd'],
                        encoding='utf-8',
                        dtype=data_types,
                        usecols=columns,
                        true_values=['T'],
                        false_values=['F'],
                        engine='c',
                        buffer_lines=512 * 1024,
                        skiprows=range(1, int(skiprows)),
                        nrows=nrows)
        except ValueError as msg:
            error = True
            _logger.debug("ValueError: " + str(msg))

        if not error:
            skiprows = skiprows + nrows

        nrows = nrows / 2
