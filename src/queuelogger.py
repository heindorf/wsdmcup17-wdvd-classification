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

import logging.handlers
import multiprocessing
import pickle
import os
import sys
import traceback

import config

_QUEUE_FILE_NAME = 'queue.p'
_QUEUE = None
_LISTENER = None


#######################################################################
# Worker process that writes log records to queue
#######################################################################

def configure_logger(logger):
    """Configure the logger.

    Configure the logger, for example, by setting handlers and redirecting
    stdout and stderr to the logger.
    """
    try:
        # Causes an exception when called for the manager process or
        # listener process because the queue has not been set yet.
        # This is on purpose to prevent executing this function which
        # redirects sys.stdout and sys.stderr
        global _QUEUE
        _QUEUE = pickle.load(open(_QUEUE_FILE_NAME, 'rb'))

        queue_handler = logging.handlers.QueueHandler(_QUEUE)

        # On linux, the logging handlers are inherited from the parent
        # process. Hence it is necessary to remove them first.
        for handler in logger.handlers:
            logger.removeHandler(handler)

        logger.addHandler(queue_handler)
        logger.setLevel(config.LOG_LEVEL)

        # Redirect everything sent to stdout and stderr to the logger
        # (libraries that sent warnings to stdout or stderr include
        # scikit-learn, pandas and joblibsometimes sent warnings to stdout
        # or stderr)
        sys.stdout = _LoggingStream(logger, 'STDOUT', logging.INFO)
        sys.stderr = _LoggingStream(logger, 'STDERR', logging.WARN)
    except IOError:
        pass


class _LoggingStream(object):
    """Stream similar to stdout/stderr that writes to a logger."""

    def __init__(self, logger, stream_name, log_level=logging.INFO):
        self.logger = logger
        self.stream_name = stream_name
        self.log_level = log_level
        self.linebuf = ''
        self.encoding = 'utf-8'

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(
                self.log_level,
                '[' + self.stream_name + "] " + line.rstrip())

    # For example, the multiprocessing library calls sys.stderr.flush()
    def flush(self):
        pass


def set_context(time_label, system_name):
    """Change the system name of the logger."""
    context_change = _ContextChange(time_label, system_name)
    _QUEUE.put(context_change)


class _ContextChange():
    def __init__(self, new_time_label, new_system_name):
        self.__new_time_label = new_time_label
        self.__new_system_name = new_system_name

    def get_new_time_label(self):
        return self.__new_time_label

    def get_new_system_name(self):
        return self.__new_system_name


########################################################################
# Listener process that retrieves log records from queue.
########################################################################

def _run_listener(output_prefix, queue, lock):
    """Run the listener process' top-level loop.

    The listener process waits for logging LogRecords
    on the queue and handles them. It quits when it gets the
    special record 'None'.
    """
    # The listener process has been successfully loaded now
    # (in particular, the main file has been executed now)
    lock.release()

    _configure_listener(output_prefix)
    while True:
        try:
            record = queue.get()

            # We send None as a sentinel to tell the listener to quit.
            if record is None:
                break
            elif isinstance(record, _ContextChange):
                _set_listener_formatters(record)
            else:
                logger = logging.getLogger(record.name)
                logger.handle(record)
        # Has the program terminated?
        except (KeyboardInterrupt, SystemExit, EOFError):
            raise  # leave loop and close logger
        except:  # noqa
            print('Other exception:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def _configure_listener(output_prefix):
    """Configure the logger."""
    logger = logging.getLogger()

    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    # Sometimes causes exception on Windows.
    # see also: https://support.microsoft.com/en-us/kb/899149
    # see also: http://stackoverflow.com/questions/10411359/unpickling-large-objects-stored-on-network-drives
    file_handler = logging.FileHandler(output_prefix + ".log", encoding="UTF-8")
    logger.addHandler(file_handler)

    _set_listener_formatters(_ContextChange("INIT", "INIT"))


def _set_listener_formatters(context_change):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        '[%%(asctime)s] [%%(levelname)8s] [%s] [%s] [%%(module)s] [%%(processName)s] %%(message)s' %
        (context_change.get_new_time_label(), context_change.get_new_system_name()),
        datefmt='%Y-%m-%d %H:%M:%S')

    for handler in logger.handlers:
        handler.setFormatter(formatter)


########################################################################
# Manager process that manages queue
########################################################################

def start_logging_processes():
    try:
        # Make sure there is no queue file from the last run of the
        # program. The queue file is used to communicate the
        # logging queue from the initial process to processes
        # spawned later.
        os.remove(_QUEUE_FILE_NAME)
    except OSError:
        pass

    # Start a new manager process for the queue
    global _QUEUE
    _QUEUE = multiprocessing.Manager().Queue(-1)

    # Start a new listener process and wait until it has been
    # loaded
    _start_listener_process()

    # Write queue to disk to communicate it to all worker processes
    pickle.dump(_QUEUE, open(_QUEUE_FILE_NAME, 'wb'))


def _start_listener_process():
    """Start a logger process that handles log events.

    The method does not return before the process runs.
    """
    output_prefix = config.OUTPUT_PREFIX

    lock = multiprocessing.Lock()
    lock.acquire()

    global _LISTENER
    _LISTENER = multiprocessing.Process(
        target=_run_listener, args=(output_prefix, _QUEUE, lock,))
    _LISTENER.start()

    # The acquired lock is released by listener process after
    # the listener process has been loaded
    lock.acquire()


def stop_logging_processes():
    """Stop the logger process."""
    _QUEUE.put_nowait(None)
    _LISTENER.join()
    try:
        os.remove(_QUEUE_FILE_NAME)
    except OSError:
        pass
