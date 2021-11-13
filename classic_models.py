from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import gensim.downloader as api

from laserembeddings import Laser

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import random
import io

import pandas as pd
import torch
import numpy as np

from tqdm import tqdm


class Classic:
    def __init__(self, args):
        # fix the random
        random.seed(args['seed_val'])
        np.random.seed(args['seed_val'])
        torch.manual_seed(args['seed_val'])
        torch.cuda.manual_seed_all(args['seed_val'])

        # intialise variables on the basis of embedding
        # requested
        if(args['embedding'] == 'ngram'):
            self.kwargs = {
                'ngram_range': (1, 2),  # Use 1-grams + 2-grams.
                'dtype': 'float32',
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': 'word',  # Split text into word tokens.
                'min_df': 2,
            }
            self.top_k = args['top_k']
        elif(args['embedding'] == 'w2v'):
            self.embeddings,id2word,self.word2id = self.load_vec(args['embedding_path'])
        elif(args['embedding']=='fasttext'):
            self.embeddings,id2word,self.word2id = self.load_vec(args['embedding_path'])
        elif(args['embedding']=='laser'):
            self.laser=Laser()
        elif(args['embedding']=='fasttext_codemixed'):
             self.model = args['model']

    def get_bow_embedding(self, train, test):
        # generate Bag-Of-Words embedding using Count
        # Vectoriser function of sklearn
        vectorizer = CountVectorizer()
        vectorizer.fit(train)

        x_train = vectorizer.transform(train)
        x_test = vectorizer.transform(test)

        return x_train.toarray(), x_test.toarray()

    def get_tfidf_embedding(self, train, test):
        # generate Tf-IDF embedding using Tfidf
        # Vectoriser function of sklearn
        vectorizer = TfidfVectorizer()
        vectorizer.fit(train)

        x_train = vectorizer.transform(train)
        x_test = vectorizer.transform(test)

        return x_train.toarray(), x_test.toarray()

    def ngram_vectorise(self, train_text, train_labels, test_text):
        # Use uni-grams and bi-grams of the test
        # vectorise using tf-idf vectoriser
        # and select top 20K features
        vectorizer = TfidfVectorizer(**self.kwargs)

        x_train = vectorizer.fit_transform(train_text)

        x_test = vectorizer.transform(test_text)

        selector = SelectKBest(f_classif, k=min(self.top_k, x_train.shape[1]))
        selector.fit(x_train, train_labels)

        x_train = selector.transform(x_train).astype('float32')
        x_test = selector.transform(x_test).astype('float32')

        return x_train.toarray(), x_test.toarray()
    
    def load_vec(self,emb_path, nmax=50000):
        vectors = []
        word2id = {}
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', 
                     errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                # file is stored as <word> <array of floats>
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                assert word not in word2id, 'word found twice'
                vectors.append(vect)
                word2id[word] = len(word2id)
                if len(word2id) == nmax:
                    break
        id2word = {v: k for k, v in word2id.items()}
        embeddings = np.vstack(vectors)
        # add padding and unkown word embedding
        merged_vec = self.add_pad_unk(embeddings)
        return merged_vec, id2word, word2id
    
    def add_pad_unk(self,vector):
        # padding is zeros vector
        pad_vec = np.zeros((1,vector.shape[1])) 
        # unkown is average of all
        unk_vec = np.mean(vector,axis=0,keepdims=True) 
        
        merged_vec=np.append(vector, unk_vec, axis=0)
        merged_vec=np.append(merged_vec, pad_vec, axis=0)
        
        return merged_vec
    
    def encode(self,data):
        new_data=[]

        for row in data:
            encoded=np.empty((0,300))
            words=row.split(' ')
            for word in words:
                word=word.lower()
                try:
                    vec=self.embeddings[self.word2id[word]]
                except KeyError:
                    # unkown word
                    vec=np.zeros((300))
                vec =np.expand_dims(vec,axis=0)
                encoded = np.append(encoded,vec,axis=0)
            new_data.append(np.mean(encoded,axis=0))
        return np.array(new_data)
    
    def encode_codemixed(self,data):
        new_data=[]

        for row in data:
            encoded=np.empty((0,300))
            words=row.split(' ')
            for word in words:
                word=word.lower()
                try:
                    vec=self.model.wv[word]
                except KeyError:
                    # unkown word
                    vec=np.zeros((300))
                vec =np.expand_dims(vec,axis=0)
                encoded = np.append(encoded,vec,axis=0)
            new_data.append(np.mean(encoded,axis=0))
        return np.array(new_data)
    
    def get_pretrained_embedding(self,train,test):
        x_train = self.encode(train)
        x_test = self.encode(test)
        
        return x_train,x_test
    
    def get_laser_embedding(self,train,test,lang):
        x_train = self.laser.embed_sentences(train,lang=lang)
        x_test = self.laser.embed_sentences(test,lang=lang)

        return x_train,x_test
    
    def get_codemixed_embedding(self,train,test):
        x_train = self.encode_codemixed(train)
        x_test = self.encode_codemixed(test)

        return x_train,x_test

    def evalMetric(self, y_true, y_pred):
        # calculate all the metrics 
        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro')
        f1Score = f1_score(y_true, y_pred)
        area_under_c = roc_auc_score(y_true, y_pred)
        recallScore = recall_score(y_true, y_pred)
        precisionScore = precision_score(y_true, y_pred)

        nonhate_f1Score = f1_score(y_true, y_pred, pos_label=0)
        non_recallScore = recall_score(y_true, y_pred, pos_label=0)
        non_precisionScore = precision_score(y_true, y_pred, pos_label=0)
        return {"accuracy": accuracy, 'mF1Score': mf1Score, 
            'f1Score': f1Score, 'auc': area_under_c,
            'precision': precisionScore, 
            'recall': recallScore, 
            'non_hatef1Score': nonhate_f1Score, 
            'non_recallScore': non_recallScore, 
            'non_precisionScore': non_precisionScore}

    def encode_data(self, args, X_train, X_test, Y_train):
        # call appropirate encoding generator on the basis of
        # arguments
        if(args['embedding'] == 'bow'):
            x_train, x_test = self.get_bow_embedding(X_train, X_test)
        elif(args['embedding'] == 'tfidf'):
            x_train, x_test = self.get_tfidf_embedding(X_train, X_test)
        elif(args['embedding'] == 'ngram'):
            x_train, x_test = self.ngram_vectorise(X_train, Y_train, X_test)
        elif(args['embedding'] == 'w2v'):
            x_train, x_test = self.get_pretrained_embedding(X_train, X_test)
        elif(args['embedding']=='fasttext'):
            x_train,x_test = self.get_pretrained_embedding(X_train,X_test)
        elif(args['embedding']=='laser'):
            x_train,x_test = self.get_laser_embedding(X_train,X_test,args['lang'])
        elif(args['embedding']=='fasttext_codemixed'):
            x_train,x_test = self.get_codemixed_embedding(X_train,X_test)
        return x_train, x_test

    def initialise_models(self, weights):
        # initalise all Classical Machine Learning Models
        DT = DecisionTreeClassifier(class_weight=weights,
                                    criterion='gini', max_depth=100, max_features=1.0,
                                    max_leaf_nodes=10, min_impurity_split=1e-07,
                                    min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.10, presort=False,
                                    random_state=42, splitter='best')

        SVM = SVC(kernel="rbf", class_weight=weights, probability=True)

        RF = RandomForestClassifier(n_estimators=1000, random_state=0,
                                    n_jobs=1000, max_depth=100, bootstrap=True,
                                    class_weight=weights)

        AB = AdaBoostClassifier()

        KNN = KNeighborsClassifier(leaf_size=1, p=2, n_neighbors=20)

        GNB = GaussianNB()

        LR = LogisticRegression(C=1.0, class_weight=weights, dual=False,
                                fit_intercept=True, intercept_scaling=1, max_iter=100,
                                multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                                solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

        XGBC = XGBClassifier(learning_rate=0.1, n_estimators=10000,
                             max_depth=6, min_child_weight=6, gamma=0, subsample=0.6,
                             colsample_bytree=0.8, reg_alpha=0.005,
                             objective='binary:logistic', nthread=2,
                             scale_pos_weight=1, seed=42, class_weight=weights)

        return [SVM, XGBC, RF, DT, AB, KNN, GNB, LR]

    def run(self, args, df_train, df_test):
        # generate train and test values
        X_train = df_train['Text'].values
        Y_train = df_train['Label'].values
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values

        # encode data
        x_train, x_test = self.encode_data(args, X_train, X_test, Y_train)

        # get classifiers
        classifiers = self.initialise_models(args['weights'])

        names = ['SVC', 'XGBoost', 'Random Forest', 'Decision Tree', 'AdaBoost',
                 'KNN', 'Gausian NB', 'Logistic Regression']

        res = []

        # fit every classifier on train and test it on 
        # test set
        for i, cf in enumerate(classifiers):
            cf.fit(x_train, Y_train)
            y_pred = cf.predict(x_test)
            metric = self.evalMetric(Y_test, y_pred)
            metric['name']=names[i]
            # save results
            res.append(metric)

        return res
