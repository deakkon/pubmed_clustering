import numpy as np
np.random.seed(42)

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from utils.preprocess import preprocess_text
from collections import Counter
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

from utils.exploration import Exploration
from utils.transformers import TokenizePreprocessor, sentence_tokenize, vectorizer

preprocess_text = preprocess_text()
tokenizer = TokenizePreprocessor()
ex = Exploration()

#print(preprocess_text.gold_data_labeled.head(5))
gold_data = preprocess_text.get_text(preprocess_text.gold_data_labeled.PMID.values.tolist(), 'gold_text',preprocess_text.gold_data_labeled.Label.values.tolist())
labels_true = preprocess_text.gold_data_labeled.Label.values.tolist()

test_data = preprocess_text.get_text(preprocess_text.test_data.PMID.values.tolist(), 'test_data')
print(len(set(labels_true)), Counter(labels_true))

ex.ground_truth_cluster_analysis(gold_data,
                                labels_true,
                                ['lsa','tfidf', "lda"],
                                ['title'],
                                ['abstract'],
                                ['title', 'abstract'],
                                ['title', 'NE'],
                                ['title', 'abstract', 'NE']
                                 )