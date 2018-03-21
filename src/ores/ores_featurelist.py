from collections import OrderedDict

import numpy as np

from src.feature import Feature
from src.ores.baselinetransformers import EqualsTransformer

# Sources for those features
#     https://arxiv.org/pdf/1703.03861.pdf
#     https://github.com/wiki-ai/wb-vandalism/blob/master/wb_vandalism/feature_lists/wikidata.py
#         (the branch master is not up to date!)
#     https://github.com/wiki-ai/wb-vandalism/tree/sample_subsets/wb_vandalism/feature_lists
#         (this seems to be the most up to date branch, commit 644ec0d64fdf1f0524438626d207431cbb3625cd)

# Sources for descriptions
#     Compare https://github.com/wikimedia/mediawiki-extensions-Wikibase/blob/master/lib/i18n/en.json

# Command for verifying whether all categories are found
#     data['revisionAction'].str.cat(data['revisionSubaction'], sep='_', na_rep='na').value_counts()

BASELINE_FEATURES = OrderedDict()
BASELINE_FEATURES['baseline'] = OrderedDict()

BASELINE_FEATURES['baseline']['general'] = [

    # Added/removed/changed sitelinks
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetsitelink', np.nan))], 'wbsetsitelink_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetsitelink', 'add'))], 'wbsetsitelink_add'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetsitelink', 'set'))], 'wbsetsitelink_set'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetsitelink', 'remove'))], 'wbsetsitelink_remove'),

    # Added/removed/changed labels
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetlabel', 'add'))], 'wbsetlabel_add'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetlabel', 'set'))], 'wbsetlabel_set'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetlabel', 'remove'))], 'wbsetlabel_remove'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('special', 'setlabel-set'))], 'special_setlabel-set'),

    # Added/removed/changed descriptions
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetdescription', 'add'))], 'wbsetdescription_add'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetdescription', 'set'))], 'wbsetdescription_set'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetdescription', 'remove'))], 'wbsetdescription_remove'),

    # Added/removed/changed statements
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbcreateclaim', np.nan))], 'wbcreateclaim_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbcreateclaim', 'create'))], 'wbcreateclaim_create'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetclaim', 'create'))], 'wbsetclaim_create'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetclaim', 'update'))], 'wbsetclaim_update'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbremoveclaims', np.nan))], 'wbremoveclaims_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbremoveclaims', 'remove'))], 'wbremoveclaims_remove'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetclaim', 'update-rank'))], 'wbsetclaim_update-rank'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetclaimvalue', np.nan))], 'wbsetclaimvalue_na'),

    # Added/removed/changed aliases
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetaliases', 'add'))], 'wbsetaliases_add'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetaliases', 'add-remove'))], 'wbsetaliases_add-remove'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetaliases', 'set'))], 'wbsetaliases_set'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetaliases', 'remove'))], 'wbsetaliases_remove'),

    # Added/removed badges!
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetsitelink', 'set-badges'))], 'wbsetsitelink_set-badges'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetsitelink', 'add-both'))], 'wbsetsitelink_add-both'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetsitelink', 'set-both'))], 'wbsetsitelink_set-both'),

    # Added/removed qualifiers
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetqualifier', np.nan))], 'wbsetqualifier_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetqualifier', 'add'))], 'wbsetqualifier_add'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetqualifier', 'update'))], 'wbsetqualifier_update'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetclaim', 'update-qualifiers'))], 'wbsetclaim_update-qualifiers'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbremovequalifiers', 'remove'))], 'wbremovequalifiers_remove'),

    # Added/removed references
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetreference', np.nan))], 'wbsetreference_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetreference', 'add'))], 'wbsetreference_add'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetreference', 'set'))], 'wbsetreference_set'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbremovereferences', np.nan))], 'wbremovereferences_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbremovereferences', 'remove'))], 'wbremovereferences_remove'),

    # Misc
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetlabeldescriptionaliases', np.nan))], 'wbsetlabeldescriptionaliases_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbcreateredirect', np.nan))], 'wbcreateredirect_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wblinktitles', 'connect'))], 'wblinktitles_connect'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wblinktitles', 'create'))], 'wblinktitles_create'),

    Feature('numberOfSitelinks'),
    Feature('numberOfLabels'),
    Feature('numberOfDescriptions'),
    Feature('numberOfStatements'),
    Feature('numberOfAliases'),
    Feature('numberOfBadges'),
    Feature('numberOfQualifiers'),
    Feature('numberOfReferences'),
    Feature('numberOfProperties'),  # feature in source code, but not in paper by Sarabadani et al. 2017

    Feature(['dataType'], [EqualsTransformer(('external-id',))], 'identifier_changed'),
]

BASELINE_FEATURES['baseline']['vandalism'] = [
    Feature('proportionOfQidAdded'),
    Feature('proportionOfLinksAdded'),
    Feature('proportionOfLanguageAdded'),

    # Has English label changed?
    Feature(['revisionAction', 'revisionLanguage'], [EqualsTransformer(('wbsetlabel', 'en'))], 'en_label_touched'),

    # Changed properties (P21, P27, P54, P569, P18, P109, P373, P856)
    Feature(['property'], [EqualsTransformer(('P21',))], 'P21'),    # sex or gender
    Feature(['property'], [EqualsTransformer(('P27',))], 'P27'),    # country of citizenship
    Feature(['property'], [EqualsTransformer(('P54',))], 'P54'),    # member of sports team
    Feature(['property'], [EqualsTransformer(('P569',))], 'P569'),  # date of birth
    Feature(['property'], [EqualsTransformer(('P18',))], 'P18'),    # image
    Feature(['property'], [EqualsTransformer(('P109',))], 'P109'),  # signature
    Feature(['property'], [EqualsTransformer(('P373',))], 'P373'),  # commons category
    Feature(['property'], [EqualsTransformer(('P856',))], 'P856'),  # official website

    Feature('isLivingPerson'),
    Feature('isHuman'),
]


BASELINE_FEATURES['baseline']['non-vandalism'] = [
    # Is it a client edit?
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('clientsitelink', 'update'))], 'clientsitelink_update'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('clientsitelink', 'remove'))], 'clientsitelink_remove'),

    # Is it a merge?
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbmergeitems', 'from'))], 'wbmergeitems_from'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbmergeitems', 'to'))], 'wbmergeitems_to'),

    # Revert, rollback, restore
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('undo', np.nan))], 'undo_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('rollback', np.nan))], 'rollback_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('restore', np.nan))], 'restore_na'),

    # Is it creating a new item?
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbcreate', 'new'))], 'wbcreate_new'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('pageCreation', np.nan))], 'pageCreation_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('special', 'create-item'))], 'special_create-item'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbsetentity', np.nan))], 'wbsetentity_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbeditentity', np.nan))], 'wbeditentity_na'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbeditentity', 'create'))], 'wbeditentity_create'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbeditentity', 'update'))], 'wbeditentity_update'),
    Feature(['revisionAction', 'revisionSubaction'], [EqualsTransformer(('wbeditentity', 'override'))], 'wbeditentity_override'),

]

BASELINE_FEATURES['baseline']['editor'] = [
    Feature('isBotUser'),
    Feature('isAdvancedUser'),
    Feature('isAdminUser'),
    Feature('isCuratorUser'),

    Feature('isRegisteredUser'),

    Feature('userSecondsSinceFirstRevisionRegistered')  # Log scaling in paper by Sarabadani et al. 2017 but not in source code
]
