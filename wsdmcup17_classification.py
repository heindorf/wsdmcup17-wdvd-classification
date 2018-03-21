#!/usr/bin/env python3

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


# Ignore Scikit Learn Deprecation Warnings
# see: https://github.com/scikit-learn/scikit-learn/issues/2531#issuecomment-29432889
import warnings
_old_showwarning = warnings.showwarning


def _new_showwarning(message, category, filename, lineno, file=None, line=None):
    if category != DeprecationWarning:
        _old_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _new_showwarning


import argparse  # noqa
import logging  # noqa
import sys  # noqa
import traceback  # noqa

from src import utils  # noqa
from src import learn  # noqa
from src import queuelogger  # noqa

_logger = logging.getLogger()


def _parse_args():
    parser = argparse.ArgumentParser(
        description='WSDM Cup 2017 Classification.')

    parser.add_argument('FEATURES',
                        help='raw feature file')
    parser.add_argument('TRUTH',
                        help='Truth files (in bz2) separated by semicolon(;)')
    parser.add_argument('RESULTS',
                        help='path prefix for storing results')

    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    files = {}
    files['features'] = args.FEATURES.split(';')
    files['truth'] = args.TRUTH.split(';')
    output_prefix = args.RESULTS

    utils.init_output_prefix(output_prefix)

    try:
        queuelogger.start_logging_processes()
        queuelogger.configure_logger(_logger)

        result = learn.main(files)
    except:  # noqa
        _logger.error(traceback.format_exc())
    finally:
        queuelogger.stop_logging_processes()

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(result)
else:
    # Make sure all output to stdout and stderr is logged.
    # In particular, capture output of joblib.Parallel.
    queuelogger.configure_logger(_logger)
