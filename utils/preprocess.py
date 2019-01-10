import numpy as np
np.random.seed(42)

import pandas as pd
import pubmed_parser as pp
import pickle
from nltk.corpus import stopwords
import requests
from io import StringIO
from tqdm import tqdm

from sklearn.feature_selection import SelectPercentile,SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler

from utils.transformers import TokenizePreprocessor, sentence_tokenize, vectorizer

import gensim
from gensim.test.utils import common_dictionary, common_corpus
from gensim.sklearn_api import HdpTransformer

from sklearn.base import BaseEstimator, TransformerMixin

class preprocess_text(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.gold_data_labeled = pd.read_csv('instructions/pmids_gold_set_labeled.txt', sep='\t', header=None, names=['PMID', 'Label'])
        self.gold_data_unlabeled = pd.read_csv('instructions/pmids_gold_set_unlabeled.txt', sep='\t', header=None, names=['PMID'])
        self.test_data = pd.read_csv('instructions/pmids_test_set_unlabeled.txt', sep='\t', header=None, names=['PMID'])
        self.keys = ['title', 'abstract', 'keywords', 'pmid']
        self.tokenizer = TokenizePreprocessor()
        self.stop_words = set(stopwords.words('english'))


    def get_pubtator(self, pmid):

        url = "https://www.ncbi.nlm.nih.gov/research/bionlp/pubtator2/api/v1/publications/export/pubtator?pmids={}".format(pmid)
        r = requests.get(url)
        entities = r.text.split('\n', 2)[2:][0]
        entities_text = StringIO(entities)
        df =  pd.read_csv(entities_text, sep="\t", names=['PMID', 'Start', 'Stop', 'NE', 'type', 'ID'])
        return " ".join(df.NE.values.tolist())

    def get_text(self, pmids, file_name, labels=None):
        # pmids: list of pmids to get text for from PubMed via pubmed_parser

        data = []
        try:
            with open('prepared/data/'+file_name, "rb") as f:
                data = pickle.load(f)
            print("Loaded {} dataset.".format(file_name))
        except FileNotFoundError:
            if labels:
                for pmid, label in tqdm(zip(pmids, labels), ascii=True, desc="Preparing {} data.".format(file_name), total=len(labels)):
                    tmp_dict = {
                        k:v for k,v in pp.parse_xml_web(pmid).items() if k in self.keys
                    }
                    tmp_dict['label']=label
                    tmp_dict['NE'] = self.get_pubtator(pmid)
                    data.append(tmp_dict)

            else:
                for pmid in pmids:
                    tmp_dict = {
                        k: v for k, v in pp.parse_xml_web(pmid).items() if k in self.keys
                    }
                    data.append(tmp_dict)

            with open('prepared/data/'+file_name, "wb") as f:
                pickle.dump(data, f)

        return data

    def dummy_tokenizer(self, doc, stop_words=False):
        # dummy tokenizer, used when we have pretokenized tekst as it returns the tokenized document(s)
        if stop_words:
            doc = [w for w in doc if not w in self.stop_words]
        #print(doc)
        return doc

    def transform_text(self,
                        trainData,
                        trainLabels,
                       vectorizer=None,
                        ngramRange=(1,2),
                        analyzerLevel='word',
                        feature_selector='chi2'
                        ):

        if vectorizer not in ['tfidf', 'lda', 'hdp', 'lsa']:
            raise ValueError("feature_selector must be one of {}".format(['tfidf', 'lda', 'hdp', 'lsa']))

        if feature_selector not in ['chi2', None]:
            raise ValueError("feature_selector must be one of {}".format(['lsa','chi2',None]))

        if feature_selector and vectorizer == 'tfidf':
            print("Transforming documents and performing feature selection with {}".format(feature_selector))
        else:
            print("Transforming documents with no feature selection")


        transformed_text = None
        transformer = None
        selector = None
        representation = None


        if vectorizer == 'tfidf' or 'lsa':# or 'pca':
            transformer = TfidfVectorizer(ngram_range=ngramRange,
                                    analyzer=analyzerLevel,
                                    norm='l2',
                                    decode_error='replace',
                                    # max_df=max_df_freq,
                                    min_df=2,
                                    sublinear_tf=True,
                                    lowercase=True,
                                    strip_accents='unicode',
                                     #stop_words='english',
                                    tokenizer=self.dummy_tokenizer,
                                    preprocessor=self.dummy_tokenizer)
            representation=transformer

        else:
            transformer = CountVectorizer(# max_df=0.95,
                                         min_df=2,
                                        decode_error='replace',
                                        ngram_range=ngramRange,
                                         tokenizer=self.dummy_tokenizer,
                                         preprocessor=self.dummy_tokenizer)



        if vectorizer == 'lda':
            trainData = transformer.fit_transform(trainData)
            representation=LatentDirichletAllocation(n_components=10,
                                                     topic_word_prior=0.01,
                                                     max_iter=1000,
                                                     n_jobs=-1)

        if vectorizer == 'lsa':
            trainData = transformer.fit_transform(trainData)
            # trainData = transformer.fit_transform(trainData)
            representation=TruncatedSVD(n_components=100, n_iter=1000)

        if vectorizer == "hdp":
            raise ValueError("NOT WORKING; REPLACE WITH FUCNTION WHICH EXECUTES GENSIMS HDP!")

        transformed_text = representation.fit_transform(trainData)
        # print(transformed_text)

        if feature_selector == 'chi2' and vectorizer=='tfidf':
            selector = SelectPercentile(chi2, 15)
            transformed_text = selector.fit_transform(transformed_text, trainLabels)

        # if vectorizer in ['lda','hdp']:
        #     print("\nTopics in LDA model:")
        #     tf_feature_names = transformer.get_feature_names()
        #     self.print_top_words(representation, tf_feature_names, 10)

        print('Transformed train data set feature space size:\t {}'.format(transformed_text.shape))
        return transformed_text, transformer, representation, selector

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)#, "\n=========")

    def get_hdp_representation(self,
                        trainData,
                        trainLabels,
                        ngramRange=(1,2),
                        analyzerLevel='word',
                        feature_selector='chi2',
                        vectorizer=None):
        pass

