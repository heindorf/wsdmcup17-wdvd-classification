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


########################################################################
# Online Transformers
########################################################################
class StreamGroupSplitTransformer:
    """Further divides groups.

    Operates on streams of (r, g, s) pairs where r denotes an id,
    g denotes a group id, and s denotes whether the should be split.

    In case a group is split, all the following group ids g are set to the
    first id r in this group.

    Returns a new group id.
    """

    def __init__(self):
        self.d = {}
        self.group_splits = 0

    def partial_fit_transform(self, r, g, s):
        # group encountered for the first time?
        if g not in self.d:
            self.d[g] = g
        # group encountered previously AND group split encountered currently
        elif s:
            self.d[g] = r
            self.group_splits += 1

        return self.d[g]


class StreamGroupReduceTransformer:
    """Operates on streams of (g,v) pairs where g denotes a group and v a value.

    Reduces the stream within every group by applying the two-parameter function func.
    """

    def __init__(self, func):
        self.func = func
        self.d = {}

    def partial_fit(self, g, v):
        if g in self.d:
            self.d[g] = self.func(self.d[g], v)
        else:
            self.d[g] = v
        return self.d[g]

    def transform(self, g):
        return self.d[g]


def group_reduce_lookahead(g, X, func, lookahead):
    """Apply function func cumulatively while looking ahead."""
    if lookahead > len(g):
        lookahead = len(g)  # unlimited lookahead

    result = [np.nan] * len(g)

    transformer = StreamGroupReduceTransformer(func)

    for i in range(len(g) + lookahead - 1):
        if i < len(g):
            # add current element to lookahead data structure
            cur_g = g[i]
            cur_v = X[i]
            transformer.partial_fit(cur_g, cur_v)

        prev_i = i - lookahead + 1
        if prev_i >= 0:
            # compute result
            prev_g = g[prev_i]
            result[prev_i] = transformer.transform(prev_g)

    result = np.asarray(result)
    return result
