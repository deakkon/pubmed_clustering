import numpy as np
np.random.seed(42)

from yellowbrick.text import TSNEVisualizer

from utils.preprocess import preprocess_text
from utils.transformers import TokenizePreprocessor

from sklearn.metrics import pairwise_distances
from scipy.spatial import distance

from sklearn.externals import joblib

from sklearn.cluster import *
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import metrics
import hdbscan

import pandas as pd

class Exploration(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.tp = TokenizePreprocessor()
        self.pt = preprocess_text()


    def prepare_data(self, arg, documents):

        if len(arg) == 1:
            data = [doc[arg[0]] for doc in documents]

        if len(arg) == 2:
            data = [doc[arg[0]]+doc[arg[1]] for doc in documents]

        if len(arg) == 3:
            data = [doc[arg[0]]+doc[arg[1]]+doc[arg[2]] for doc in documents]

        return data

    def KL(self, a, b):
        return np.sum(np.where(a != 0, a * np.log(a / b), 0))

    def JS(self, a, b):

        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)

        # print(a,b)

        M = np.add(a,b)* .5
        # print(.5 * np.add(self.KL(a,M), self.KL(b,M)))
        return .5 * np.add(self.KL(a,M), self.KL(b,M))

    def visualize_tsne(self, X, labels, title, save_figure=True):
        # feature_base: ['title'], ['abstract'], ['title', 'abstract'],
        # for arg in argv:
        #     data = self.prepare_data(arg, documents)
        #     tokenized_gold_data = self.tp.transform(data)
        #     X, _, _ = self.pt.transform_text(tokenized_gold_data, labels)
        #     # Create the visualizer and draw the vectors
        # title = "_".join(title)
        tsne = TSNEVisualizer(title=title,decompose_by=50 if X.shape[1] > 50 else X.shape[1]-1)
        tsne.fit(X, labels)

        if save_figure:
            tsne.poof(outpath='img/'+title+'.png')
        else:
            tsne.poof()

    def get_cluster_info(self, data, cluster_labels, method):
        # data: matrix with data (e.g. tf-idf values)
        # cluster_labels: dict with 'label':[indices] values

        clusters={x:{'min':None, 'max':None, 'centroid':None} for x,y in cluster_labels.items()}
        centroid_matrix = np.zeros((len(cluster_labels), data.shape[1]))
        # print(centroid_matrix.shape)
        counter = 0

        for label, indices in cluster_labels.items():
            cluster_data = data[indices,:]
            cluster_centroid = cluster_data.mean(axis=0)

            # cluster_covariance = np.cov(cluster_data.A)
            # print(cluster_covariance.shape)
            # print(np.argwhere(cluster_covariance > 0), np.argwhere(cluster_covariance > 0).shape)
            # input(cluster_covariance)

            # cluster_eigenvalues = np.linalg.eig(cluster_covariance)
            # input(cluster_eigenvalues)

            # print(label, distance.cdist(cluster_data.A, cluster_centroid, metric='mahalanobis'))#, VI=np.linalg.inv(cluster_covariance)))

            cluster_variance = []
            for i in range(0, cluster_data.shape[0]):
                # print(np.array(cluster_centroid))
                # print(cluster_data[i,:].A)

                if method in ['lda', 'hdp']:
                    dist = self.JS(cluster_centroid.tolist(), cluster_data[i, :].tolist())

                if method in ['tfidf',  'lsa']:
                    try:
                        dist = distance.euclidean(np.array(cluster_centroid), cluster_data[i,:].A)
                    except AttributeError:
                        dist = distance.euclidean(np.array(cluster_centroid), cluster_data[i, :])

                cluster_variance.append(dist)
                # input()

            centroid_matrix[counter]=cluster_centroid
            clusters[label]['centroid'] = cluster_centroid
            clusters[label]['min'] = min(cluster_variance)
            clusters[label]['max'] = max(cluster_variance)
            # print(clusters[label])
            counter += 1

        # for cluster, cluster_info in clusters.items():
        #     print(cluster, cluster_info['min'], cluster_info['max'])

        centroid_distances = distance.squareform(distance.pdist(centroid_matrix))
        cumulative_distance = 0
        for i in range(0,centroid_distances.shape[0]):
            for j in range(0, centroid_distances.shape[1]):
                if i != j:
                    cumulative_distance += centroid_distances[i][j] - clusters[list(cluster_labels.keys())[i]]['max']

        # print(centroid_distances, cumulative_distance)
        return centroid_matrix, round(cumulative_distance, 3)

    def get_cluster_info_probabilitites(self, data, cluster_labels):
        # usd when docuemnt representet with lda or hdp, as they are probabilitites of a document beloning to detected topics.
        # DEPRECATED

        clusters={x:{'min':None, 'max':None, 'centroid':None} for x,y in cluster_labels.items()}
        centroid_matrix = np.zeros((len(cluster_labels), data.shape[1]))
        # print(centroid_matrix.shape)
        counter = 0

        for label, indices in cluster_labels.items():
            # print(label)
            cluster_data = data[indices,:]
            cluster_centroid = cluster_data.mean(axis=0)

            cluster_variance = []
            for i in range(0, cluster_data.shape[0]):
                # print("cluster_data\t", cluster_data[i,:])
                # print("cluster_data mean\t", cluster_data[i,:].tolist())
                dist = self.JS(cluster_centroid.tolist(), cluster_data[i,:].tolist())
                cluster_variance.append(dist)

            centroid_matrix[counter]=cluster_centroid
            clusters[label]['centroid'] = cluster_centroid
            clusters[label]['min'] = min(cluster_variance)
            clusters[label]['max'] = max(cluster_variance)
            # print(clusters[label])
            counter += 1

        # for cluster, cluster_info in clusters.items():
        #     print(cluster, cluster_info['min'], cluster_info['max'])

        # print(centroid_matrix)
        # print(distance.pdist(centroid_matrix))
        print(distance.squareform(distance.pdist(centroid_matrix)))
        return centroid_matrix

    def ground_truth_cluster_analysis(self, documents, labels, methods, *argv):
        # ANALISE CLUSTER STATISTICS: GROUND TRUTH
        # documents: TOKENIZED DOCUEMNTS
        # labels: CLUSTER LABELS
        # *argv: ['title'], ['abstract'], ['title', 'abstract']

        cluster_indices = {label:[i for i, x in enumerate(labels) if x == label] for label in labels}
        # print(cluster_indices)

        best={"transformer":None,
              "selector":None,
              "representation": None,
              "clustering":None}

        results = {
            'method':[],
            'arg':[],
            'n_gram':[],
            'homogeneity_score':[],
            'ami':[]
        }

        ami = 0
        homogenity = 0

        for arg in argv:
            for method in methods:
                for n_gram in [(1,1), (1,2), (1,3), (1,4)]:

                    title = "_".join(arg+[method, str(n_gram[0]), str(n_gram[1])])
                    data = self.prepare_data(arg, documents)
                    tokenized_gold_data = self.tp.transform(data)
                    transformed_text, transformer, representation, selector = self.pt.transform_text(tokenized_gold_data,
                                                                                                     labels,
                                                                                                     vectorizer=method,
                                                                                                     ngramRange=n_gram)
                    # print(transformed_text)
                    # if method == "tfidf":
                    #     centroid_matrix = self.get_cluster_info(transformed_text, cluster_indices)
                    # else:
                    centroid_matrix, cumulative_distance = self.get_cluster_info(transformed_text, cluster_indices, method)
                    # self.visualize_tsne(transformed_text, labels, title, save_figure=False)

                    ## performance measure
                    # feed transformed_text in to a clustering algorithm
                    if method == 'lsa':
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric=self.JS)#, gen_min_span_tree=True, prediction_data=True)
                    else:
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)  # , gen_min_span_tree=True, prediction_data=True)

                    clusterer.fit_predict(transformed_text)
                    homogenity_tmp = round(metrics.homogeneity_score(labels, clusterer.labels_), 3)
                    ami_tmp = round(metrics.adjusted_mutual_info_score(labels, clusterer.labels_), 3)

                    # input(clusterer.labels_)
                    print(method, '\t', arg, '\t', n_gram, '\t', ami_tmp, '\t', homogenity_tmp, '\t', cumulative_distance)
                    print("--------------------")

                    results['method'].append(method)
                    results['arg'].append("_".join(arg))
                    results['n_gram'].append('_'.join([str(n_gram[0]), str(n_gram[1])]))
                    results['homogeneity_score'].append(homogenity_tmp)
                    results['ami'].append(ami_tmp)

                    if (ami_tmp > ami) and (homogenity_tmp > homogenity):
                        best['transformer'] = transformer
                        best['selector'] = selector
                        best['representation'] = representation
                        best['clustering'] = clusterer

            # calculate most similar centorids betwen the ground truthe centroids and centroids of new data to get matching labels
            # feed in to evaluation metrics

        # save best performing for inference
        joblib.dump(best, "prepared/utils/models.util")
        df = pd.DataFrame.from_dict(results)
        df.to_csv('report/results/results.csv', sep='\t')

    def inference(self, documents, labels, feature_input, *argv):

        if feature_input not in [['title'], ['abstract'], ['title', 'abstract'], ['title', 'NE']]:
            raise ValueError("feature_input is one of {}".format(feature_input))

        models = joblib.load("prepared/utils/models.util")
        data = self.prepare_data(feature_input, documents)
        tokenized_data = self.tp.transform(data)

        if models['transformer']:
            transformed_text = models['transformer'].transform(tokenized_data)

        if models['selector']:
            transformed_text = models['selector'].transfom(transformed_text)

        if models['representation']:
            transformed_text = models['representation'].transfom(transformed_text)

        clustered_text = models['clustering'].transfom(transformed_text)
