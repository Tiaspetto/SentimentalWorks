#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import Crossfolder
import numpy as np
import csv
import sys
import pandas as pd
import re
import nltk

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from plaindeepneural import*

corpus_file = "signal-news1.jsonl"
word_list_path = "opinion-lexicon-English/"

# reg for match pure number words
num_reg = r'\b\d+(\.|\s+)?\d*\b'

# reg for match words short than 3
ls3_reg = r'\b[a-z]{1,2}\b'

# reg for match non-alphanumeric characters
alpha_reg = r'[^a-zA-Z\d\s]'


def preprocessing(mystr):
    sentences = sent_tokenize(mystr)
    processed_line = []

    for sent in sentences:
        sent = re.sub(alpha_reg, "", sent)
        sent = re.sub(num_reg, " ", sent)
        sent = re.sub(ls3_reg, "", sent)
        processed_line.append(sent)

    return(" ".join(processed_line))


def calculate_result(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='micro')
    m_recall = metrics.recall_score(actual, pred, average='micro')
    print('predict info:')
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    print(
        'f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred, average='micro')))
    return metrics.f1_score(actual, pred, average='micro')


def read_csv_file(filename):
    ten_folders = Crossfolder.TenFolders(10)
    focused_topics = ["topic.earn", "topic.acq", "topic.money.fx", "topic.grain",
                      "topic.crude", "topic.trade", "topic.interest", "topic.ship", "topic.wheat", "topic.corn"]
    focused_topics_index = []
    with open(filename, newline='',) as f:
        reader = csv.reader(f)
        try:
            row_index = 0
            for row in reader:
                if row_index > 0:
                    pid = int(row[0])
                    topics = row[3:137]
                    topics = [int(x) for x in topics]
                    text = row[138].lower()+" "+row[139].lower()
                    folder_id = pid % 10
                    
                    text = preprocessing(text)
                    train_test_flag = row[2]

                    labels = []
                    flag = False
                    for index in focused_topics_index:
                        if topics[index] == 1:
                            flag = True
                        labels.append(topics[index])

                    if train_test_flag == "train" and flag == True:
                        ten_folders.add_data(text)
                        ten_folders.add_labels(labels)
                    elif flag == True:
                        ten_folders.add_test_data(text)
                        ten_folders.add_test_labels(labels)

                elif row_index == 0:
                    for i in range(3, 138):
                        if row[i] in focused_topics:
                            focused_topics_index.append(i)

                row_index += 1

            return ten_folders

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(
                filename, reader.line_num, e))


if __name__ == '__main__':
    folders = read_csv_file("reutersCSV.csv")
    for i in range(folders.get_nfolders()):
        print("=========================================================================")
        print("set folder: ", i, " as test set")
        data_train, labels_train, data_test, labels_test = folders.orgnize_crossfolder(
            i)
        train_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                           max_df=0.5,
                                           stop_words='english')
        fea_train = train_vectorizer.fit_transform(data_train)

        test_vectorizer = TfidfVectorizer(
            vocabulary=train_vectorizer.vocabulary_)
        fea_test = test_vectorizer.fit_transform(data_test)
        
        if i == 0:
            weights = np.asarray(fea_train.mean(axis=0)).ravel().tolist()
            weights_df = pd.DataFrame({'term': train_vectorizer.get_feature_names(), 'weight': weights})
            weights_df = weights_df.sort_values(by='weight', ascending=False).head(20)

            print(weights_df)

        svd = TruncatedSVD(50)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        fea_train = lsa.fit_transform(fea_train)
        fea_test = lsa.transform(fea_test)
        
        print("------------------------------------------------------------------------")
        print("calculating DECISION TREE please waite...")
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(fea_train, np.array(labels_train))

        dt.pred = dt.predict(fea_test)
        calculate_result(np.array(labels_test), dt.pred)

        print("------------------------------------------------------------------------")
        print("calculating KNN please waite...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(fea_train, np.array(labels_train))

        knn.pred = knn.predict(fea_test)
        calculate_result(np.array(labels_test), knn.pred)


        print("------------------------------------------------------------------------")
        print("calculating Nueral networks please waite...")
        paramters = L_layer_model(fea_train.T, np.array(labels_train).T, [50,100,50,10])
        nnpred = NN_predict(fea_test.T, np.array(labels_test).T, paramters)
        calculate_result(np.array(labels_test), np.array(nnpred))

    print("=========================================================================")
    print("Test optimal model on test set")
    data_train, labels_train, data_test, labels_test = folders.simple_test()
    train_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                       max_df=0.5,
                                       stop_words='english')
    fea_train = train_vectorizer.fit_transform(data_train)

    test_vectorizer = TfidfVectorizer(
        vocabulary=train_vectorizer.vocabulary_)
    fea_test = test_vectorizer.fit_transform(data_test)
    
    svd = TruncatedSVD(50)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    fea_train = lsa.fit_transform(fea_train)
    fea_test = lsa.transform(fea_test)

    print("---------------------------------optimal---------------------------------------")
    print("calculating Nueral networks please waite...")
    paramters = L_layer_model(fea_train.T, np.array(labels_train).T, [50,100,50,10])
    nnpred = NN_predict(fea_test.T, np.array(labels_test).T, paramters)
    calculate_result(np.array(labels_test), np.array(nnpred))
