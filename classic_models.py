from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import gensim.downloader as api

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import random

import pandas as pd
import torch
import numpy as np


class Classic:
    def __init__(self, args):
        # fix the random
        random.seed(args['seed_val'])
        np.random.seed(args['seed_val'])
        torch.manual_seed(args['seed_val'])
        torch.cuda.manual_seed_all(args['seed_val'])

        if(args['embedding'] == 'ngram'):
            self.kwargs = {
                'ngram_range': (1, 2),  # Use 1-grams + 2-grams.
                'dtype': 'float32',
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': 'word',  # Split text into word tokens.
                'min_df': 2,
            }
        elif(args['embedding'] == 'w2v'):
            self.wv = api.load('word2vec-google-news-300')
            self.top_k = args['top_k']

    def get_bow_embedding(self, train, test):
        vectorizer = CountVectorizer()
        vectorizer.fit(train)

        x_train = vectorizer.transform(train)
        x_test = vectorizer.transform(test)

        return x_train, x_test

    def get_tfidf_embedding(self, train, test):
        vectorizer = TfidfVectorizer()
        vectorizer.fit(train)

        x_train = vectorizer.transform(train)
        x_test = vectorizer.transform(test)

        return x_train, x_test

    def ngram_vectorise(self, train_text, train_labels, test_text):
        vectorizer = TfidfVectorizer(**self.kwargs)

        x_train = vectorizer.fit_transform(train_text)

        x_test = vectorizer.transform(test_text)

        print("Features Calculated: ", x_train.shape[1])

        selector = SelectKBest(f_classif, k=min(self.top_k, x_train.shape[1]))
        selector.fit(x_train, train_labels)

        x_train = selector.transform(x_train).astype('float32')
        x_test = selector.transform(x_test).astype('float32')

        return x_train, x_test

    def encode_w2v(self, wv, data):
        new_data = []
        for sentence in data:
            sample = []
            for word in sentence:
                if word in wv:
                    sample.append(wv[word])
                else:
                    sample.append(np.zeros(shape=(300)))
            new_data.append(np.mean(np.array(sample), axis=0))
        return np.array(new_data)

    def get_w2v_embedding(self, train, test):
        x_train = self.encode_w2v(self.wv, train)
        x_test = self.encode_w2v(self.wv, test)

        return x_train, x_test

    def evalMetric(self, y_true, y_pred, prefix):
        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro')
        f1Score = f1_score(y_true, y_pred, labels=np.unique(y_pred))
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_c = auc(fpr, tpr)
        recallScore = recall_score(y_true, y_pred, labels=np.unique(y_pred))
        precisionScore = precision_score(
            y_true, y_pred, labels=np.unique(y_pred))
        return dict({prefix+"accuracy": accuracy, prefix+'mF1Score': mf1Score,
                     prefix+'f1Score': f1Score, prefix+'precision': precisionScore,
                     prefix+'recall': recallScore})

    def encode_data(self, args, X_train, X_test, Y_train):
        if(args['embedding'] == 'bow'):
            x_train, x_test = self.get_bow_embedding(X_train, X_test)
        elif(args['embedding'] == 'tfidf'):
            x_train, x_test = self.get_tfidf_embedding(X_train, X_test)
        elif(args['embedding'] == 'ngram'):
            x_train, x_test = self.ngram_vectorise(X_train, Y_train, X_test)
        elif(args['embedding'] == 'w2v'):
            x_train, x_test = self.get_w2v_embedding(X_train, X_test)

        return x_train, x_test

    def initialise_models(self, weights):
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

        XGBC = XGBClassifier(learning_rate=0.1, n_estimators=100000,
                             max_depth=6, min_child_weight=6, gamma=0, subsample=0.6,
                             colsample_bytree=0.8, reg_alpha=0.005,
                             objective='binary:logistic', nthread=2,
                             scale_pos_weight=1, seed=42, class_weight=weights)

        return [SVM, XGBC, RF, DT, AB, KNN, GNB, LR]

    def run(self, args, df_train, df_test):
        X_train = df_train['Text'].values
        Y_train = df_train['Label'].values
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values

        x_train, x_test = self.encode_data(args, X_train, X_test, Y_train)

        classifiers = self.initialise_models(args['weights'])

        names = ['SVC', 'XGBoost', 'Random Forest', 'Decision Tree', 'AdaBoost',
                 'KNN', 'Gausian NB', 'Logistic Regression']

        res = []

        for i, cf in enumerate(classifiers):
            cf.fit(x_train, Y_train)
            y_pred = cf.predict(x_test)
            metric = self.evalMetric(Y_test, y_pred, names[i])
            res.append(metric)

        return res
