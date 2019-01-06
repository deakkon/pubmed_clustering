from sklearn.cluster import *

class cluster_documents():

    def __init__(self):
        self.param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
        self.cluster_algorithms = [

        ]


    def optimize_model(self):
        pass

    def cluster(self, data, labels=None):
        pass