from sklearn.cluster import *
from sklearn import metrics
from hdbscan import *

class cluster_documents():

    def __init__(self):
        self.clustering_algos = [

        ]

        self.ground_truth_centroids = None
        self.prediction_centroids = None

    def optimize_model(self):
        pass

    def cluster(self, data, labels=None, model=None):
        

    def evaluate(self, cluster_labels, ground_truth_labels):
        pass