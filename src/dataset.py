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

import math

import numpy as np
import pandas as pd

REVISION_ID = 'revisionId'
TIMESTAMP = 'timestamp'
REVISION_SESSION_ID = 'revisionSessionId'
CONTENT_TYPE = 'contentType'
USER_NAME = 'userName'
IS_REGISTERED_USER = 'isRegisteredUser'


class DataSet:
    def __init__(self):
        self.__time_label = 'No time label'
        self.__system_name = 'No system'  # system name (e.g., WDVD, FILTER or ORES)
        self.__meta = pd.DataFrame()      # meta data about each instance (rows in X)
        self.__features = []              # list of feature names (columns in X)
        self.__X = None                   # numpy array of features
        self.__y = None                   # numpy array of labels

    def shallow_copy(self):
        result = DataSet()
        result.set_time_label(self.__time_label)
        result.set_system_name(self.__system_name)
        result.set_meta(self.__meta)
        result.set_features(self.__features)
        result.set_X(self.__X)
        result.set_Y(self.__y)

        return result

    def __len__(self):
        return len(self.__y)

    def __getitem__(self, sliced):
        result = DataSet()
        result.set_time_label(self.__time_label)
        result.set_system_name(self.__system_name)
        result.set_meta(self.__meta[sliced])
        result.set_features(self.__features)
        result.set_X(self.__X[sliced, :])
        result.set_Y(self.__y[sliced])

        return result

    def get_X(self):
        return self.__X

    def set_X(self, X):
        self.__X = np.ascontiguousarray(X)

    def get_Y(self):
        return self.__y

    def set_Y(self, y):
        self.__y = np.ascontiguousarray(y)

    def set_features(self, features):
        self.__features = features

    def get_features(self):
        return list(self.__features)

    def get_groups(self):
        result = [feature[0] for feature in self.__features]
        result = self.remove_duplicates(result)
        return result

    def get_subgroups(self):
        result = [(feature[0], feature[1]) for feature in self.__features]
        result = self.remove_duplicates(result)
        return result

    def get_group_names(self):
        result = [feature[0] for feature in self.__features]
        result = self.remove_duplicates(result)
        return result

    def get_subgroup_names(self):
        result = [feature[1] for feature in self.__features]
        result = self.remove_duplicates(result)
        return result

    def get_feature_names(self):
        result = [feature[2] for feature in self.__features]
        return result

    def set_feature_names(self, feature_names):
        features = [('NO_GROUP', 'NO_SUBGROUP', feature_name)
                    for feature_name in feature_names]
        self.__features = features

    def get_revision_ids(self):
        return self.__meta[REVISION_ID]

    def get_group_ids(self):
        return self.__meta[REVISION_SESSION_ID]

    def get_content_types(self):
        return self.__meta[CONTENT_TYPE]

    def get_user_name(self):
        return self.__meta[USER_NAME]

    def get_meta(self):
        return self.__meta

    def get_metrics_meta(self):
        return self.get_meta()[[CONTENT_TYPE, IS_REGISTERED_USER]].copy()

    def set_meta(self, meta):
        self.__meta = meta
        self.__meta.reset_index(drop=True, inplace=True)

    def set_group_ids(self, group_ids):
        self.__meta[REVISION_SESSION_ID] = group_ids
        self.__meta.reset_index(drop=True, inplace=True)

    def get_system_name(self):
        return self.__system_name

    def set_system_name(self, system_name):
        self.__system_name = system_name

    def get_time_label(self):
        return self.__time_label

    def set_time_label(self, time_label):
        self.__time_label = time_label

    def select_features(self, feature_indices):
        result = self.shallow_copy()  # uses a lot of memory?!?
        result.set_X(self.__X[:, feature_indices])

        features = [self.__features[feature_index]
                    for feature_index in feature_indices]
        result.set_features(features)
        return result

    def select_features_by_name(self, feature_names):
        cur_feature_names = self.get_feature_names()
        feature_indices = [cur_feature_names.index(feature_name)
                           for feature_name in feature_names]

        result = self.select_features(feature_indices)
        return result

    def select_feature(self, feature_name):
        feature_indices = self.get_feature_names().index(feature_name)
        result = self.select_features([feature_indices])
        return result

    def select_group(self, group_name):
        if group_name == 'ALL':
            return self

        feature_indices = []
        for i in range(len(self.__features)):
            if self.__features[i][0] == group_name:
                feature_indices += [i]

        result = self.select_features(feature_indices)
        return result

    def select_subgroup(self, subgroup_name):
        feature_indices = []
        for i in range(len(self.__features)):
            if self.__features[i][1] == subgroup_name:
                feature_indices += [i]

        result = self.select_features(feature_indices)
        return result

    def add_feature(self, X, name):
        X = np.reshape(X, (-1, 1))
        self.set_X(np.hstack((self.__X, X)))
        self.__features = self.__features + [name]

    # Does not necessarily create a copy
    def sample(self, fraction, seed=1):
        result = DataSet()

        np.random.seed(seed)
        selection = np.random.choice(
            len(self.__X),
            size=math.floor(fraction * len(self.__X)),
            replace=False)

        result.set_meta(self.__meta.iloc[selection])
        result.set_X(self.__X[selection])
        result.set_Y(self.__y[selection])
        result.set_features(self.__features)
        result.set_system_name(self.__system_name)

        return result

    def apply_mask(self, mask):
        result = DataSet()

        result.set_meta(self.__meta.loc[mask, :])
        result.set_X(self.__X[mask])
        result.set_Y(self.__y[mask])
        result.set_features(self.__features)
        result.set_system_name(self.__system_name)

        return result

    def append(self, dataset):
        result = DataSet()

        result.set_meta(self.__meta.append(dataset.__meta, ignore_index=True))
        result.set_X(np.concatenate((self.__X, dataset.__X)))
        result.set_Y(np.concatenate((self.__y, dataset.__y)))
        result.set_features(self.__features)
        result.set_system_name(self.__system_name)

        return result

    def filter_body(self):
        mask = np.array(((self.get_content_types() == 'STATEMENT') |
                         (self.get_content_types() == 'SITELINK')))

        result = self.apply_mask(mask)
        return result

    def filter_head(self):
        mask = np.array((self.get_content_types() == 'TEXT'))

        result = self.apply_mask(mask)
        return result

    def get_positives(self):
        mask = self.__y.astype(np.bool)

        result = self.apply_mask(mask)
        return result

    def undersample(self, n_times):
        non_reverted_indices = np.where(~self.__y)[0]
        selection = np.random.choice(
            non_reverted_indices,
            size=math.floor(n_times * sum(self.__y)),
            replace=False)
        mask = self.__y.astype(np.bool)  # select all reverted revisions
        mask[selection] = True  # additionally select the sampled revisions

        result = self.apply_mask(mask)
        return result

    def oversample(self, n_times):
        indices = np.arange(len(self.__y))
        reverted_indices = np.where(self.__y)[0]

        indices = np.append(indices, [reverted_indices] * (n_times - 1))
        indices = np.sort(indices)

        result = DataSet()
        result.set_meta(self.__meta.iloc[indices])
        result.set_X(self.__X[indices])
        result.set_Y(self.__y[indices])
        result.set_features(self.__features)
        result.set_system_name(self.__system_name)

        return result

    def random_split(self, fraction):
        np.random.seed(1)
        length = len(self.__X)
        selection = np.random.choice(
            length,
            size=math.floor(fraction * length),
            replace=False)
        mask1 = np.zeros(length, dtype=bool)
        mask1[selection] = True
        mask2 = ~mask1

        result1 = self.apply_mask(mask1)
        result2 = self.apply_mask(mask2)

        return result1, result2

    def split(self, fraction):
        length = len(self.__X)
        split_index = math.floor(length * fraction)

        return self[:split_index], self[split_index:]

    def get_number_of_features(self):
        return self.get_X().shape[1]

    @staticmethod
    def remove_duplicates(values):
        output = []
        seen = set()
        for value in values:
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output

    def get_vandalism_fraction(self):
        vandalism_fraction = self.get_Y().mean()
        return vandalism_fraction

    def get_index_for_revision_id(self, revision_id):
        result = self.get_index_for_revision_id_from_df(self.get_meta(), revision_id)
        return result

    def get_index_for_date(self, date):
        result = self.get_index_for_date_from_df(self.get_meta(), date)
        return result

    @staticmethod
    def get_index_for_revision_id_from_df(df, revision_id):
        """Return the first index that does NOT come before the revision_id."""
        result = df[REVISION_ID].searchsorted(revision_id)[0]
        return result

    @staticmethod
    def get_index_for_date_from_df(df, timestamp):
        """Return the first index that does NOT come before the timestamp."""
        result = df[TIMESTAMP].searchsorted(timestamp)[0]
        return result
