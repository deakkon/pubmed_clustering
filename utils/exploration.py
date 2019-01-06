import numpy as np
np.random.seed(42)

from yellowbrick.text import TSNEVisualizer
from yellowbrick.features.pca import PCADecomposition

from utils.preprocess import preprocess_text
from utils.transformers import TokenizePreprocessor

class Exploration():

    def __init__(self):
        self.tp = TokenizePreprocessor()
        self.pt = preprocess_text()

    def visualize_tsne(self, documents, labels, *argv):
        # feature_base: ['title'], ['abstract'], ['title', 'abstract'],
        for arg in argv:

            if len(arg) == 1:
                data = [doc[arg[0]] for doc in documents]

            if len(arg) == 2:
                data = [doc[arg[0]]+doc[arg[1]] for doc in documents]

            # print(data)
            tokenized_gold_data = self.tp.transform(data)
            X, _, _ = self.pt.transform_text(tokenized_gold_data, labels)
            # Create the visualizer and draw the vectors
            tsne = TSNEVisualizer()
            tsne.fit(X, labels)
            tsne.poof()

    def cluster_analysis(self, documents, labels, *argv):
        # ANALISE CLUSTER STATISTICS
        pass