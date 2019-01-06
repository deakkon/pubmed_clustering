import numpy as np
np.random.seed(42)

import pandas as pd
import sklearn

import pubmed_parser as pp
import pickle
import traceback
from morph import flatten
from nltk.corpus import stopwords

from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from utils.transformers import TokenizePreprocessor, sentence_tokenize, vectorizer

class preprocess_text():

    def __init__(self):

        self.gold_data_labeled = pd.read_csv('instructions/pmids_gold_set_labeled.txt', sep='\t', header=None, names=['PMID', 'Label'])
        self.gold_data_unlabeled = pd.read_csv('instructions/pmids_gold_set_unlabeled.txt', sep='\t', header=None, names=['PMID'])
        self.test_data = pd.read_csv('instructions/pmids_test_set_unlabeled.txt', sep='\t', header=None, names=['PMID'])
        self.keys = ['title', 'abstract', 'keywords', 'pmid']
        self.tokenizer = TokenizePreprocessor()
        self.stop_words = set(stopwords.words('english'))

    def get_text(self, pmids, file_name, labels=None):
        # pmids: list of pmids to get text for

        print("Preparing {} dataset.".format(file_name))

        data = []
        try:
            with open('prepared/data/'+file_name, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            if labels:
                for pmid, label in zip(pmids, labels):
                    tmp_dict = {
                        k:v for k,v in pp.parse_xml_web(pmid).items() if k in self.keys
                    }
                    tmp_dict['label']=label
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

    def dummy_tokenizer(self, doc, stop_words=True):
        # dummy tokenizer, used when we have pretokenized tekst as it returns the tokenized document(s)
        if stop_words:
            doc = [w for w in doc if not w in self.stop_words]
        #print(doc)
        return doc

    def transform_text(self,
                        trainData,
                        trainLabels,
                        ngramRange=(1,1),
                        max_df_freq=0.9,
                        analyzerLevel='word',
                        feature_selector='chi2',
                        standard_scaler=False):

        transformed_text = None
        selector = None
        tfidf_vect = None

        if feature_selector not in ['tsvd', 'chi2', None]:
            raise ValueError("feature_selector must be one of {}".format(['pca','chi2',None]))

        if feature_selector:
            print("Transforming documents and performing feature selection with {}".format(feature_selector))
        else:
            print("Transforming documents with no feature selection")

        tfidf_vect = TfidfVectorizer(ngram_range=ngramRange,
                                    analyzer=analyzerLevel,
                                    norm='l2',
                                    decode_error='replace',
                                    max_df=max_df_freq,
                                    sublinear_tf=True,
                                    lowercase=True,
                                    strip_accents='unicode',
                                     #stop_words='english',
                                    tokenizer=self.dummy_tokenizer,
                                    preprocessor=self.dummy_tokenizer)

        transformed_text = tfidf_vect.fit_transform(trainData)
        # print(transformed_text.shape)

        if feature_selector == 'chi2':
            selector = SelectPercentile(chi2, 15)
            transformed_text = selector.fit_transform(transformed_text, trainLabels)

        if feature_selector == 'tsvd':
            selector = TruncatedSVD(n_components=100, n_iter=100)
            transformed_text = selector.fit_transform(transformed_text)

        if standard_scaler:
            scaler = StandardScaler(with_mean=False)
            transformed_text = scaler.fit_transform(transformed_text)

        print('Transformed train data set feature space size:\t {}'.format(transformed_text.shape))
        return transformed_text, tfidf_vect, selector