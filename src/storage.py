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

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import config

_TEMP_DIR = None


def get_temporary_directory():
    global _TEMP_DIR
    if _TEMP_DIR is None:
        _TEMP_DIR = tempfile.TemporaryDirectory(
            prefix=config.TEMP_PREFIX + __name__ + '-')
    return _TEMP_DIR


def get_preprocessor_path(time_label, system_name):
    return (os.path.dirname(config.OUTPUT_PREFIX) +
            '/models/{0}_{1}_Preprocessor.pkl'
            .format(time_label, system_name))


def get_clf_path(clf_name):
    return (os.path.dirname(config.OUTPUT_PREFIX) +
            "/models/{0}.pkl".format(clf_name))


def get_prediction_path(name, tmp):
    if tmp:
        base_path = get_temporary_directory().name
    else:
        base_path = os.path.dirname(config.OUTPUT_PREFIX)

    filepath = (base_path + '/scores/{0}.csv.bz2'.format(name))
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return filepath


def dump_preprocessor(preprocessor, time_label, system_name):
    dump(preprocessor, get_preprocessor_path(time_label, system_name))


def load_preprocessor(time_label, system_name):
    preprocessor = load(get_preprocessor_path(time_label, system_name))
    return preprocessor


def dump_clf(clf, clf_name):
    dump(clf, get_clf_path(clf_name))


def load_clf(clf_name):
    clf = load(get_clf_path(clf_name))
    clf.set_params(n_jobs=1)
    if 'base_estimator__n_jobs' in clf.get_params().keys():
        clf.set_params(base_estimator__n_jobs=1)
    return clf


def dump_predictions(name, dataset, prob, tmp=False):
    filepath = get_prediction_path(name, tmp)
    result = pd.DataFrame()
    result['REVISION_ID'] = dataset.get_revision_ids()
    result['VANDALISM_SCORE'] = prob.astype(np.float32)
    result.to_csv(filepath, compression='bz2', index=False, mode='x')


def load_predictions(name, tmp=False):
    filepath = get_prediction_path(name, tmp)
    return load_predictions_from_path(filepath)


def load_predictions_from_path(filepath):
    df = pd.read_csv(filepath)
    df['VANDALISM_SCORE'] = df['VANDALISM_SCORE'].astype(np.float32)
    return df


def dump(obj, filepath):
    if os.path.exists(filepath):
        raise FileExistsError("File exists: '{0}'".format(filepath))

    dirname = os.path.dirname(filepath)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    joblib.dump(obj, filepath)


def load(filepath):
    result = joblib.load(filepath)
    return result
