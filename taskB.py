import pickle

import pandas as pd
import numpy as np
import copy
from pathlib import Path

from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.tokenize import casual_tokenize
from nltk import word_tokenize
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import torch
import transformers as ppb  # pytorch transformers
import csv
import preprocessing


def get_data(filename):
    data = pd.read_csv(filename, sep = "\t", header = 0, quoting=csv.QUOTE_NONE)
    sentences = data.iloc[1:, 1].values
    labels = data.iloc[1:, 3].values

    return sentences, labels

def get_test_data(filename):
    data = pd.read_table(filename, header=None, quoting=csv.QUOTE_NONE)
    sentences = data.iloc[1:, 1].values
    return sentences

def get_stem (lang, sentence):
    stemmer = SnowballStemmer(lang)
    stemmed = ''
    for word in casual_tokenize(sentence):
        word = stemmer.stem(word)
        stemmed = stemmed + word + ' '

    return stemmed



def tfidf(lang, train_x, test_x):

    #load
    load_train = Path("embeddings/tfidf_train_"+lang + ".pk")
    load_test = Path("embeddings/tfidf_test_"+lang + ".pk")

    if load_test.is_file() & load_train.is_file():
        with open("embeddings/tfidf_train_"+lang + ".pk", "rb") as fin:
            train_x = pickle.load(fin)

        with open("embeddings/tfidf_test_"+lang + ".pk", "rb") as fin:
            test_x = pickle.load(fin)

        # print("Loaded from file TFIDF.")
        return train_x, test_x

    else:

        # max_df: ignora parole che hanno document freq maggiore della soglia messa (se float è percentuale)
        # min_df: come prima ma minore (in questo caso almeno 7 docs)
        # max_features: costruisce vocabolario con un numero di parole definito, top, ordinate per frequenza
        vectorizer = TfidfVectorizer(max_features=50000, min_df=1, max_df=0.9, stop_words=stopwords.words(lang))
        fit_data = []
        for sent in train_x:
            fit_data.append(sent)
        for sent in test_x:
            fit_data.append(sent)

        vectorizer.fit(fit_data)

        train_x = vectorizer.transform(train_x).toarray()
        test_x = vectorizer.transform(test_x).toarray()

        # dump
        with open("embeddings/tfidf_train_"+lang + ".pk", "wb") as fout:
            pickle.dump(train_x, fout)

        with open("embeddings/tfidf_test_"+lang + ".pk", "wb") as fout:
            pickle.dump(test_x, fout)

        # print("TFIDF dumped on file.")

        return train_x, test_x


def bert(lang, data_x, name):

    # load
    load_data_x = Path("embeddings/" + name + lang + ".pk")

    if load_data_x.is_file():
        with open("embeddings/"+ name + lang + ".pk", "rb") as fin:
            data_features = pickle.load(fin)

        # print("Loaded from file BERT.")
        return data_features

    else:

        # pretrained model e tokenizatore
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'bert-base-multilingual-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        # vettorizzazione comprende anche test set per creazione vocabolario
        tot_set = copy.deepcopy(data_x)
        # for i in test_x:
        #    tot_set.append(i)

        # tokenization
        tokens = []
        for tweet in tot_set:
            tmp = tokenizer.tokenize(tweet)
            tokens.append(tmp)
        ind_token = []
        for i in tokens:
            tmp = tokenizer.encode(i, add_special_tokens=True)
            ind_token.append(tmp)

        # padding per processare tutto insieme, più veloce
        max_len = 0
        for i in ind_token:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0] * (max_len - len(i)) for i in ind_token])
        # maschera per non considerare 0 del padding
        attention_mask = np.where(padded != 0, 1, 0)

        # passaggio a bert
        ids = torch.tensor(padded).to(torch.int64) #vuole long
        # print(ids.shape)
        attention_mask = torch.tensor(attention_mask)
        with torch.no_grad(): # non calcola i gradienti, non servono in bert (e senza questo mi esplode la ram)
            results = model(ids, attention_mask=attention_mask)

        # serve solo output corrispondente a primo token, ovvero [CLS], che conta come embedding dell'intera frase
            features = results[0][:, 0, :].numpy()  # features per classificazione

        data_features = []
        for i in range(0, len(data_x)):
            data_features.append(features[i])

        # dump
        with open("embeddings/" + name + lang + ".pk", "wb") as fout:
            pickle.dump(data_features, fout)

        # print("BERT dumped on file.")

        return data_features


def glove(lang, train_x, name):
    # load
    load_data_x = Path("embeddings/" + name + lang + ".pk")

    if load_data_x.is_file():
        with open("embeddings/" + name + lang + ".pk", "rb") as fin:
            data_features = pickle.load(fin)

        # print("Loaded from file GLOVE")
        return data_features

    else:

        # carico glove
        embeddings_index = dict()
        f = open('glove.twitter.27B/glove.twitter.27B.200d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        sentences_embeddings = []
        for sent in train_x:
            emb = []
            for i in word_tokenize(sent):
                tmp = embeddings_index.get(i)
                if tmp is not None:
                    emb.append(tmp)

            emb = np.array(emb)
            np.seterr('raise')

            v = emb.sum(axis=0)
            if type(v) != np.ndarray:

                sentences_embeddings.append(np.zeros(200))

            else:
                sentences_embeddings.append(v)
        np.seterr('raise')

        # dump
        with open("embeddings/" + name + lang + ".pk", "wb") as fout:
            pickle.dump(sentences_embeddings, fout)

        # print("GLOVE dumped on file.")
        return sentences_embeddings


def get_embeddings(lang, train_x, test_x, embeddings):
    # words embedding
    if embeddings == 'tfidf':
        train_x, test_x = tfidf(lang, train_x, test_x)

    elif embeddings == 'bert':
        train_x = bert(lang, train_x, "bert_train_")
        test_x = bert(lang, test_x, "bert_test_")

    elif embeddings == 'glove':
        train_x = glove(lang, train_x, "glove_train_")
        test_x = glove(lang, test_x, "glove_test_")

    elif embeddings == 'tfidf_bert':
        train_x_tf, test_x_tf = tfidf(lang, train_x, test_x)
        train_x_bert = bert(lang, train_x, "bert_train_")
        test_x_bert = bert(lang, test_x, "bert_test_")

        train_x = np.concatenate((np.array(train_x_tf), np.array(train_x_bert)), axis=1)
        test_x = np.concatenate((np.array(test_x_tf), np.array(test_x_bert)), axis=1)

    elif embeddings == 'tfidf_glove':
        train_x_tf, test_x_tf = tfidf(lang, train_x, test_x)
        train_x_glove = glove(lang, train_x, "glove_train_")
        test_x_glove = glove(lang, test_x, "glove_test_")

        train_x = np.concatenate((np.array(train_x_tf), np.array(train_x_glove)), axis=1)
        test_x = np.concatenate((np.array(test_x_tf), np.array(test_x_glove)), axis=1)

    elif embeddings == 'all':
        train_x_tf, test_x_tf = tfidf(lang, train_x, test_x)
        train_x_bert = bert(lang, train_x, "bert_train_")
        test_x_bert = bert(lang, test_x, "bert_test_")
        train_x_glove = glove(lang, train_x, "glove_train_")
        test_x_glove = glove(lang, test_x, "glove_test_")

        train_x = np.concatenate((np.array(train_x_tf), np.array(train_x_bert), np.array(train_x_glove)), axis=1)
        test_x = np.concatenate((np.array(test_x_tf), np.array(test_x_bert), np.array(test_x_glove)), axis=1)


    else:
        print("Wrong Embedding pick")
        exit(1)

    return train_x, test_x


def classifier(lang, train_x, train_y, test_x, model, embeddings):
    print("Lang: " + lang + "\nEmbeddings: " + embeddings + "\nModel: " + model)

    train_x, test_x = get_embeddings(lang, train_x, test_x, embeddings)


    feature_selection = SelectFromModel(ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=10))

    train_x = feature_selection.fit_transform(train_x, train_y)
    test_x = feature_selection.transform(test_x)


    # model classification
    print("Classification started...\n")
    # a questo punto, in train_x abbiamo la rappresentazione .transform del trainingSet
    if model == 'log_reg':
        params = [{'C': [0.01, 0.1, 0.5, 1]}]  # param C indica quanto regolarizzare (ne è l'inverso)
        grid_search = GridSearchCV(LogisticRegression(max_iter=15000), refit=True, param_grid=params, cv=3)
        grid_search.fit(train_x, train_y)

        return grid_search.best_score_, grid_search.best_estimator_

    elif model == 'svm':
        params = [{'C': [0.01, 0.1, 0.5, 1, 10]}]  # param C indica quanto regolarizzare (ne è l'inverso)
        grid_search = GridSearchCV(SVC(kernel='linear', max_iter = 15000), param_grid=params, cv=3)
        grid_search.fit(train_x, train_y)


        return grid_search.best_score_, grid_search.best_estimator_

    else:
        print("Wrong Model pick")
        exit(1)

    return


def run():


    ########### split dataset e inserimento file giusti
    train_x, train_y = get_data('Dataset/haspeede2_dev_taskAB.tsv')
    test_x = get_test_data('Dataset/haspeede2_test_taskAB-tweets.tsv')
    lang = ['italian']
    # models = ['log_reg', 'svm']
    models = ['log_reg']
    # embeddings = ['tfidf', 'glove', 'bert', 'tfidf_bert', 'tfidf_glove', 'all']
    embeddings = ['tfidf']
    corpus = []
    for i in range (0, len(train_x)):
        sentence = preprocessing.pre_process(train_x[i].lower(), False);

        if sentence != ' ':
            # sentence = get_stem(lang[1], sentence)
            corpus.append(sentence)
        else:
            train_y = np.delete(train_y, i, 0)
    train_x = corpus

    test = []
    for i in range (0, len(test_x)):
        sentence = preprocessing.pre_process(test_x[i].lower(), False);
        # sentence = get_stem(lang[1], sentence)

        test.append(sentence)
    test_x = test

    best_score = 0.0
    best_model = None
    best_embedding = None

    for m in models:
        for e in embeddings:

            score, model = classifier(lang[0], train_x, train_y, test_x, m, e)
            train_x, test_x = get_embeddings(lang[0], train_x, test_x, e)
            model.fit(train_x, train_y)

            if score > best_score:
                best_score = score
                best_model = model
                best_embedding = e

    print(best_model)
    print( m + " " + e + ": " + str(score))

    train_x, test_x = get_embeddings(lang[0], train_x, test_x, best_embedding)
    best_model.fit(train_x, train_y)

    test_y = best_model.predict(test_x)

    with open("taskB/test_y_tweets_"+ m +"_"+ e + ".pk", "wb") as testyout:
        pickle.dump(test_y, testyout)

    with open("taskB/model_tweets_" + m + "_" + e + ".pk", "wb") as fout:
       pickle.dump(best_model, fout)
    print("Model saved.")


    return


run()
