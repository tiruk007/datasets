# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for ConllDatasetBuilder.

This file contains list of reusable tags which are commonly used in CoNLL-based
datasets.
"""
import collections

from tensorflow_datasets.core.dataset_builders.conll import conll_dataset_builder
from tensorflow_datasets.core.features.class_label_feature import ClassLabel
from tensorflow_datasets.core.features.sequence_feature import Sequence
from tensorflow_datasets.core.features.text_feature import Text

# Standard POS tags for family of CoNLL-based datasets.

# Used by: conll2002
CONLL_2002_ES_POS_TAGS = (
    "AO",
    "AQ",
    "CC",
    "CS",
    "DA",
    "DE",
    "DD",
    "DI",
    "DN",
    "DP",
    "DT",
    "Faa",
    "Fat",
    "Fc",
    "Fd",
    "Fe",
    "Fg",
    "Fh",
    "Fia",
    "Fit",
    "Fp",
    "Fpa",
    "Fpt",
    "Fs",
    "Ft",
    "Fx",
    "Fz",
    "I",
    "NC",
    "NP",
    "P0",
    "PD",
    "PI",
    "PN",
    "PP",
    "PR",
    "PT",
    "PX",
    "RG",
    "RN",
    "SP",
    "VAI",
    "VAM",
    "VAN",
    "VAP",
    "VAS",
    "VMG",
    "VMI",
    "VMM",
    "VMN",
    "VMP",
    "VMS",
    "VSG",
    "VSI",
    "VSM",
    "VSN",
    "VSP",
    "VSS",
    "Y",
    "Z",
)

CONLL_2002_NL_POS_TAGS = (
    "Adj",
    "Adv",
    "Art",
    "Conj",
    "Int",
    "Misc",
    "N",
    "Num",
    "Prep",
    "Pron",
    "Punc",
    "V",
)

# Used by: conll2003
CONLL_2003_POS_TAGS = (
    '"',
    "''",
    "#",
    "$",
    "(",
    ")",
    ",",
    ".",
    ":",
    "``",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "NN|SYM",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
)

# Standard chunk tags for family of CoNLL-based datasets.

# Used by: conll2003
CHUNK_TAGS = (
    "O",
    "B-ADJP",
    "I-ADJP",
    "B-ADVP",
    "I-ADVP",
    "B-CONJP",
    "I-CONJP",
    "B-INTJ",
    "I-INTJ",
    "B-LST",
    "I-LST",
    "B-NP",
    "I-NP",
    "B-PP",
    "I-PP",
    "B-PRT",
    "I-PRT",
    "B-SBAR",
    "I-SBAR",
    "B-UCP",
    "I-UCP",
    "B-VP",
    "I-VP",
)

# Standard NER tags for family of CoNLL-based datasets.

# Used by: conll2002, conll2003
NER_TAGS = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
)

# Standard separators for family of CoNLL-based datasets.

# Used by: conll2002, conll2003
CONLL_2002_SEPARATOR = " "

# Standard orderedfeature dicts for family of CoNLL-based datasets.
# Importantly, the order of the features should map the order of the feature
# columns in the input files.

# Used by: conll2002
CONLL_2002_ES_ORDERED_FEATURES = collections.OrderedDict({
    "tokens": Sequence(Text()),
    "pos": Sequence(ClassLabel(names=CONLL_2002_ES_POS_TAGS)),
    "ner": Sequence(ClassLabel(names=NER_TAGS)),
})

CONLL_2002_NL_ORDERED_FEATURES = collections.OrderedDict({
    "tokens": Sequence(Text()),
    "pos": Sequence(ClassLabel(names=CONLL_2002_NL_POS_TAGS)),
    "ner": Sequence(ClassLabel(names=NER_TAGS)),
})

# Used by: conll2003
CONLL_2003_ORDERED_FEATURES = collections.OrderedDict({
    "tokens": Sequence(Text()),
    "pos": Sequence(ClassLabel(names=CONLL_2003_POS_TAGS)),
    "chunks": Sequence(ClassLabel(names=CHUNK_TAGS)),
    "ner": Sequence(ClassLabel(names=NER_TAGS)),
})

# Standard ConllBuilderConfig for family of CoNLL-based datasets.

# Used by: conll2002
CONLL_2002_ES_CONFIG = conll_dataset_builder.ConllBuilderConfig(
    name="es",
    separator=CONLL_2002_SEPARATOR,
    ordered_features=CONLL_2002_ES_ORDERED_FEATURES,
)

CONLL_2002_NL_CONFIG = conll_dataset_builder.ConllBuilderConfig(
    name="nl",
    separator=CONLL_2002_SEPARATOR,
    ordered_features=CONLL_2002_NL_ORDERED_FEATURES,
)

# Used by: conll2003
CONLL_2003_CONFIG = conll_dataset_builder.ConllBuilderConfig(
    name="conll2003",
    separator=CONLL_2002_SEPARATOR,
    ordered_features=CONLL_2003_ORDERED_FEATURES,
)
