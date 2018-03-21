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

import config
import cProfile
import datetime
import gc
import io
import logging
import platform
import os
import pstats
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn

from src import constants

_logger = logging.getLogger()


def init_output_prefix(output_prefix):
    config.OUTPUT_PREFIX = output_prefix
    _init_output_directory()


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / 1024 / 1024
    return mem


def collect_garbage():
    _logger.debug("Total memory usage before gc.collect (in MB) : %d",
                  memory_usage_psutil())
    gc.collect()
    _logger.debug("Total memory usage after gc.collect (in MB): %d",
                  memory_usage_psutil())


# function can be called with paramater dir(), locals(), globals()
def print_variables(dictionary):
    variables = pd.Series()
    for var, obj in dictionary.items():
        variables[var] = sys.getsizeof(obj)

    variables.sort_values(ascending=False, inplace=True)

    print(variables[:5])


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)


profilers = {}


def enable_profiler():
    get_current_profiler().enable()


def disable_profiler():
    get_current_profiler().disable()


def get_current_profiler():
    thread_id = threading.get_ident()
    global profilers
    if thread_id not in profilers:
        profilers[thread_id] = cProfile.Profile()

    return profilers[thread_id]


def print_profile(verbose=0):
    get_current_profiler().disable()

    s = io.StringIO()
    ps = pstats.Stats(None, stream=s)

    for _, value in profilers.items():
        ps.add(value)

    _logger.debug("Number of profilers: " + str(len(profilers)))

    ps.dump_stats('profile.pr')

    RESTRICTIONS = 30

    if verbose > 0:
        s.truncate(0)
        s.seek(0)
        ps.sort_stats('cumtime').print_callees(RESTRICTIONS)
        _logger.debug("Profiling results for callees:\n" + s.getvalue())

        s.truncate(0)
        s.seek(0)
        ps.sort_stats('cumtime').print_callers(RESTRICTIONS)
        _logger.debug("Profiling results for callers:\n" + s.getvalue())

        s.truncate(0)
        s.seek(0)
        ps.sort_stats('cumtime').print_stats(RESTRICTIONS)
        _logger.debug("Profiling results with dirs:\n" + s.getvalue())

    s.truncate(0)
    s.seek(0)
    ps.strip_dirs().sort_stats('cumtime').print_stats(RESTRICTIONS)
    _logger.debug("Profiling results without dirs:\n" + s.getvalue())

    get_current_profiler().enable()


def _init_output_directory():
    dirname = os.path.dirname(config.OUTPUT_PREFIX)

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif os.listdir(dirname) != []:
        input_var = input("Directory not empty: \"" +
                          dirname +
                          "\". Remove all its content? (yes/no)")
        input_var = input_var.lower()
        if input_var in ['y', 'yes']:
            for the_file in os.listdir(dirname):
                file_path = os.path.join(dirname, the_file)
                os.unlink(file_path)


def print_system_info():
    if config.USE_TEST_SET:
        _logger.info("##################################################")
        _logger.info("# COMPUTATION ON TEST SET!!!")
        _logger.info("##################################################")

    # Host
    _logger.info("Host: " + platform.node())
    _logger.info("Processor: " + platform.processor())
    _logger.info("Memory (in MB): " +
                 str(int(psutil.virtual_memory().total / 1024 / 1024)))

    # Operating system
    _logger.info("Platform: " + platform.platform())

    # Python
    _logger.info("Python interpreter: " + sys.executable)
    _logger.info("Python version: " + sys.version)

    # Libraries
    _logger.info("Numpy version: " + np.__version__)
    _logger.info("Scipy version: " + scipy.__version__)
    _logger.info("Pandas version: " + pd.__version__)
    _logger.info("Scikit-learn version: " + sklearn.__version__)
    _logger.info("Psutil version: " + psutil.__version__)

    # Classification
    _logger.info("Script file: " + os.path.abspath(sys.argv[0]))
    _logger.info("Script version: " + config.__version__)
    _logger.info("Script run time: " + str(datetime.datetime.now()))

    # Configuration
    for key, value in config.get_globals().items():
        if not key.startswith("__") and not key.startswith("_"):
            _logger.info(key + "=" + str(value))

    # Constants
    for key, value in constants.get_globals().items():
        if not key.startswith("__") and not key.startswith("_"):
            _logger.info(key + "=" + str(value))


def init_pandas():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 2000)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', 200)
