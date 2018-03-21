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

import numpy as np

from src.feature import Feature
from src.ores import ores_featurelist
from src.transformers import BooleanImputer, MedianImputer, MinusOneImputer
from src.transformers import FrequencyTransformer, LogTransformer


#######################################################################
# Features
#######################################################################


def get_feature_list():
    result = _get_feature_list_from_dict(FEATURES)

    return result


def get_ores_feature_list():
    result = _get_feature_list_from_dict(ores_featurelist.BASELINE_FEATURES)
    return result


def get_filter_feature_list():
    result = OrderedDict()
    result['filter'] = OrderedDict()

    result['filter']['filter'] = [
        Feature('revisionTags', [FrequencyTransformer()], 'revisionTagsFreq')
    ]

    result = _get_feature_list_from_dict(result)
    return result


def get_meta_list():
    meta_list = [
        Feature('revisionId'),         # for splitting into training, validation and test set
        Feature('revisionSessionId'),  # for multiple instance learning
        Feature('contentType'),        # for evaluation by content type
        Feature('isRegisteredUser', [], 'isRegisteredUserMeta'),  # for evaluation by user type
        Feature('timestamp'),          # for cleaning outliers
        Feature('userName'),           # for cleaning outliers
        Feature('revisionHashTag'),    # for cleaning outliers
        Feature('property'),           # for cleaning outliers
        Feature('revisionTags'),       # for cleaning outliers
    ]
    return meta_list


def get_label_list():
    label_list = [Feature('rollbackReverted')]
    return label_list


def get_columns(feature_list):
    meta_list = get_meta_list()
    label_list = get_label_list()

    feature_list = meta_list + feature_list + label_list

    result = []

    for feature in feature_list:
        input_names = feature.get_input_names()
        for input_name in input_names:
            if (input_name is not None) and (input_name not in result):
                result.append(input_name)

    return result


# based on the output names of the features
def get_feature_names(feature_list):
    result = []

    for feature in feature_list:
        output_name = feature.get_output_name()
        if (output_name is not None) and (output_name not in result):
            result.append(output_name)

    return result


def _get_feature_list_from_dict(feature_dict):
    result = []
    for group_name, group in feature_dict.items():
        for subgroup_name, subgroup in group.items():
            for feature in subgroup:
                feature.set_group(group_name)
                feature.set_subgroup(subgroup_name)
                result += [feature]

    return result


FEATURES = OrderedDict()
FEATURES['contextual'] = OrderedDict()
FEATURES['content'] = OrderedDict()

FEATURES['content']['character'] = [
    Feature('alphanumericRatio', [MedianImputer()]),
    Feature('asciiRatio', [MedianImputer()]),
    Feature('bracketRatio', [MedianImputer()]),
    Feature('digitRatio', [MedianImputer()]),
    Feature('latinRatio', [MedianImputer()]),
    Feature('longestCharacterSequence', [MinusOneImputer()]),
    Feature('lowerCaseRatio', [MedianImputer()]),
    Feature('nonLatinRatio', [MedianImputer()]),
    Feature('punctuationRatio', [MedianImputer()]),
    Feature('upperCaseRatio', [MedianImputer()]),
    Feature('whitespaceRatio', [MedianImputer()]),
]

FEATURES['content']['word'] = [
    Feature('badWordRatio', [MedianImputer()]),
    Feature('containsLanguageWord', []),
    Feature('containsURL'),
    Feature('languageWordRatio', [MedianImputer()]),
    Feature('longestWord', [MinusOneImputer()]),
    Feature('lowerCaseWordRatio', [MedianImputer()]),
    Feature('proportionOfQidAdded'),
    Feature('proportionOfLinksAdded'),

    Feature('upperCaseWordRatio', [MedianImputer()]),
]

FEATURES['content']['sentence'] = [
    Feature('commentCommentSimilarity', [MinusOneImputer()]),
    Feature('commentLabelSimilarity', [MinusOneImputer()]),
    Feature('commentSitelinkSimilarity', [MinusOneImputer()]),
    Feature('commentTailLength', [MinusOneImputer()]),
]

FEATURES['content']['statement'] = [
    Feature('literalValue', [FrequencyTransformer()], 'literalValueFreq'),
    Feature('itemValue', [FrequencyTransformer()], 'itemValueFreq'),
    Feature('property', [FrequencyTransformer()], 'propertyFreq'),
]

FEATURES['contextual']['user'] = [
    Feature('isRegisteredUser', []),
    Feature('isPrivilegedUser', []),

    Feature('userCityName', [FrequencyTransformer()], 'userCityFreq'),
    Feature('userCountryCode', [FrequencyTransformer()], 'userCountryFreq'),
    Feature('userCountyName', [FrequencyTransformer()], 'userCountyFreq'),
    Feature('userContinentCode', [FrequencyTransformer()], 'userContinentFreq'),
    Feature('userName', [FrequencyTransformer()], 'userFreq'),
    Feature(['cumUserUniqueItems'], []),
    Feature('userRegionCode', [FrequencyTransformer()], 'userRegionFreq'),
    Feature('userTimeZone', [FrequencyTransformer()], 'userTimeZoneFreq'),
]

FEATURES['contextual']['item'] = [
    Feature('logCumItemUniqueUsers'),
    Feature('itemId', [FrequencyTransformer(), LogTransformer()], 'logItemFreq'),
]

FEATURES['contextual']['revision'] = [
    Feature('commentLength', [MinusOneImputer()]),
    Feature('isLatinLanguage', [BooleanImputer()]),
    Feature('positionWithinSession'),
    Feature('revisionAction', [FrequencyTransformer()], 'revisionActionFreq'),
    Feature('revisionLanguage', [FrequencyTransformer()], 'revisionLanguageFreq'),
    Feature('revisionPrevAction', [FrequencyTransformer()], 'revisionPrevActionFreq'),
    Feature('revisionSubaction', [FrequencyTransformer()], 'revisionSubactionFreq'),
    Feature('revisionTags', [FrequencyTransformer()], 'revisionTagFreq'),

]


def get_data_types():
    return DATA_TYPES


DATA_TYPES = {
    # Meta features
    'revisionId': np.int32,
    'revisionSessionId': np.int32,
    'userId': np.int32,
    'itemId': np.int32,
    'contentType': 'category',
    'timestamp': 'datetime',
    'commentTail': 'category',
    'englishItemLabel': str,
    'superItemId': np.float32,
    'minorRevision': np.bool,

    # Character features
    'alphanumericRatio': np.float32,
    'asciiRatio': np.float32,
    'bracketRatio': np.float32,
    'digitRatio': np.float32,
    'latinRatio': np.float32,
    'longestCharacterSequence': np.float32,
    'lowerCaseRatio': np.float32,
    'nonLatinRatio': np.float32,
    'punctuationRatio': np.float32,
    'upperCaseRatio': np.float32,
    'whitespaceRatio': np.float32,

    # Misc character features
    'arabicRatio': np.float32,
    'bengaliRatio': np.float32,
    'brahmiRatio': np.float32,
    'cyrillicRatio': np.float32,
    'hanRatio': np.float32,
    'hindiRatio': np.float32,
    'malayalamRatio': np.float32,
    'tamilRatio': np.float32,
    'teluguRatio': np.float32,

    # Word features
    'badWordRatio': np.float32,
    'containsLanguageWord': np.bool,
    'containsURL': np.bool,
    'languageWordRatio': np.float32,
    'longestWord': np.float32,
    'lowerCaseWordRatio': np.float32,
    'proportionOfQidAdded': np.float32,
    'proportionOfLinksAdded': np.float32,
    'proportionOfLanguageAdded': np.float32,
    'upperCaseWordRatio': np.float32,

    # Misc word features
    'containsBadWord': np.bool,
    'containsLanguageWord2': np.bool,

    # Sentence features
    'commentCommentSimilarity': np.float32,
    'commentLabelSimilarity': np.float32,
    'commentSitelinkSimilarity': np.float32,
    'commentTailLength': np.float32,

    # Misc sentence features
    'wordsFromCommentInText': np.float32,
    'wordsFromCommentInTextWithoutStopWords': np.float32,

    # Statement features
    'property': 'category',
    'itemValue': 'category',
    'literalValue': str,
    'dataType': 'category',

    # User features
    'isRegisteredUser': np.bool,
    'isPrivilegedUser': np.bool,
    'isBotUser': np.bool,
    'cumUserUniqueItems': np.int32,

    'userCityName': 'category',
    'userCountryCode': 'category',
    'userCountyName': 'category',
    'userContinentCode': 'category',
    'userName': 'category',
    'userRegionCode': 'category',
    'userTimeZone': 'category',

    'isAdvancedUser': np.bool,
    'isAdminUser': np.bool,
    'isCuratorUser': np.bool,
    'userSecondsSinceFirstRevision': np.int32,
    'userSecondsSinceFirstRevisionRegistered': np.int32,

    # Item features
    'hasListLabel': np.bool,
    'isHuman': np.bool,
    'labelCapitalizedWordRatio': np.float32,
    'labelContainsFemaleFirstName': np.bool,
    'labelContainsMaleFirstName': np.bool,
    'logCumItemUniqueUsers': np.int32,

    # Misc item features
    'numberOfLabels': np.int32,
    'numberOfDescriptions': np.int32,
    'numberOfAliases': np.int32,
    'numberOfStatements': np.int32,
    'numberOfSitelinks': np.int32,
    'numberOfQualifiers': np.int32,
    'numberOfReferences': np.int32,
    'numberOfBadges': np.int32,
    'numberOfProperties': np.int32,

    'latestInstanceOfItemId': np.float32,
    'isLivingPerson': np.bool,
    'latestEnglishItemLabel': str,

    # Revision features
    'commentLength': np.float32,
    'isLatinLanguage': np.bool,
    'positionWithinSession': np.int32,
    'revisionAction': 'category',
    'revisionLanguage': 'category',
    'revisionPrevAction': 'category',
    'revisionSubaction': 'category',
    'revisionTags': 'category',
    'revisionHashTag': 'category',
    'parentRevisionInCorpus': np.bool,

    # Misc revision features
    'param1': np.float32,
    'param3': 'category',
    'param4': 'category',
    'bytesIncrease': np.float32,
    'revisionSize': np.int32,
    'timeSinceLastRevision': np.float32,

    # Diff features
    'numberOfAliasesAdded': np.int32,
    'numberOfAliasesRemoved': np.int32,

    'numberOfBadgesAdded': np.int32,
    'numberOfBadgesRemoved': np.int32,

    'numberOfClaimsAdded': np.int32,
    'numberOfClaimsChanged': np.int32,
    'numberOfClaimsRemoved': np.int32,
    'numberOfDescriptionsAdded': np.int32,
    'numberOfDescriptionsChanged': np.int32,
    'numberOfDescriptionsRemoved': np.int32,
    'numberOfLabelsAdded': np.int32,
    'numberOfLabelsChanged': np.int32,
    'numberOfLabelsRemoved': np.int32,
    'numberOfSitelinksAdded': np.int32,
    'numberOfSitelinksChanged': np.int32,
    'numberOfSitelinksRemoved': np.int32,
    'numberOfIdentifiersChanged': np.int32,
    'numberOfSourcesAdded': np.int32,
    'numberOfSourcesRemoved': np.int32,
    'numberOfQualifiersAdded': np.int32,
    'numberOfQualifiersRemoved': np.int32,

    'englishLabelTouched': np.bool,
    'hasP21Changed': np.bool,
    'hasP27Changed': np.bool,
    'hasP54Changed': np.bool,
    'hasP569Changed': np.bool,
    'hasP18Changed': np.bool,
    'hasP109Changed': np.bool,
    'hasP373Changed': np.bool,
    'hasP856Changed': np.bool,

    # Labels
    'undoRestoreReverted': np.bool,
    'rollbackReverted': np.bool,

    #######################################
    # META FILE
    #######################################
    'REVISION_ID': np.int32,
    'ROLLBACK_REVERTED': np.bool,
    'UNDO_RESTORE_REVERTED': np.bool

}

RENAME_MAPPING = {
    'REVISION_ID': 'revisionId',
    'ROLLBACK_REVERTED': 'rollbackReverted',
    'UNDO_RESTORE_REVERTED': 'undoRestoreReverted'
}
