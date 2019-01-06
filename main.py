import numpy as np
np.random.seed(42)

from utils.preprocess import preprocess_text
from sklearn.cluster import *
from sklearn import metrics
from collections import Counter
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from utils.exploration import Exploration

pt = preprocess_text()
ex = Exploration()

#print(pt.gold_data_labeled.head(5))
gold_data = pt.get_text(pt.gold_data_labeled.PMID.values.tolist(), 'gold_text',pt.gold_data_labeled.Label.values.tolist())
labels_true = pt.gold_data_labeled.Label.values.tolist()

test_data = pt.get_text(pt.test_data.PMID.values.tolist(), 'test_data')
print(len(set(labels_true)), Counter(labels_true))

#title
ex.visualize_tsne(gold_data,
                  labels_true,
                  ['title'], ['abstract'], ['title', 'abstract'])

# #abstract
# gold_text = [x['abstract'] for x in gold_data]
# ex.visualize_tsne(gold_text, labels_true)
#
# # title and abstract
# gold_text = [x['title']+x['abstract'] for x in gold_data]
# ex.visualize_tsne(gold_text, labels_true)

# hdpmodel = HdpModel(corpus=X, id2word=id2word)
# hdpmodel.show_topics()

# clustering = DBSCAN(eps=0.1, min_samples=5).fit_predict(X)
# print(clustering, Counter(clustering), len(set(clustering)))
#
# clustering = AffinityPropagation(damping=0.99).fit_predict(X)
# print(clustering, Counter(clustering), len(set(clustering)))
#
# clustering = KMeans(n_clusters=len(set(labels_true))).fit_predict(X)
# print(clustering, Counter(clustering), len(set(clustering)))

