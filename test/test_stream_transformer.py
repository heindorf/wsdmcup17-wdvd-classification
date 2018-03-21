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

import unittest

import numpy as np

from src.streamtransformers import group_reduce_lookahead


class Test(unittest.TestCase):
    def testStreamGroupReduceTransformerMaximum(self):
        g = np.array([5, 10, 5, 10, 10, 7, 7, 7, 7,  5])  # noqa
        v = np.array([1,  2, 3,  4,  5, 6, 7, 8, 9, 10])  # noqa

        expected_result = np.asarray([3, 4, 3, 5, 5, 8, 9, 9, 9, 10])

        actual_result = group_reduce_lookahead(g, v, np.maximum, 3)

        np.testing.assert_array_equal(actual_result, expected_result)

    def testStreamGroupReduceTransformerPerformance(self):
        LENGTH = 100000
        g = np.random.randint(low=0, high=LENGTH / 2, size=LENGTH)
        v = np.random.rand(LENGTH, 40)
        group_reduce_lookahead(g, v, np.maximum, 3)

    def testStreamGroupReduceTransformerMean(self):
        g = np.array([5, 10, 5, 10, 10, 7, 7, 7, 7,  5])  # noqa
        v = np.array([1,  2, 3,  4,  5, 6, 7, 8, 9, 10])  # noqa

        expected_result = \
            np.asarray([2, 3, 2, 11 / 3, 11 / 3, 7, 7.5, 7.5, 7.5, 14 / 3])

        sum_result = group_reduce_lookahead(g, v, np.add, 3)
        count_result = group_reduce_lookahead(g, [1] * len(g), np.add, 3)

        actual_result = sum_result / count_result

        np.testing.assert_array_equal(actual_result, expected_result)
