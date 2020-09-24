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
    sentences = data.iloc[0:, 1].values
    labels = data.iloc[0:, 3].values

    return sentences, labels

def get_test_data(filename):
    data = pd.read_table(filename, header=None, quoting=csv.QUOTE_NONE)
    sentences = data.iloc[0:, 1].values
    return sentences

def get_stem (lang, sentence):
    stemmer = SnowballStemmer(lang)
    stemmed = ''
    for word in casual_tokenize(sentence):
        word = stemmer.stem(word)
        stemmed = stemmed + word + ' '

    return stemmed



def tfidf(lang, train_x, name):

    #load
    load_data_x = Path("embeddings/" + name + "_"+lang + ".pk")

    if load_data_x.is_file():
        with open("embeddings/" + name + "_" +lang + ".pk", "rb") as fin:
            data = pickle.load(fin)

        # print("Loaded from file TFIDF.")
        return data

    else:

        # max_df: ignora parole che hanno document freq maggiore della soglia messa (se float è percentuale)
        # min_df: come prima ma minore (in questo caso almeno 7 docs)
        # max_features: costruisce vocabolario con un numero di parole definito, top, ordinate per frequenza
        vectorizer = TfidfVectorizer(max_features=50000, min_df=1, max_df=0.9, stop_words=stopwords.words(lang))
        fit_data = []
        for sent in train_x:
            fit_data.append(sent)

        vectorizer.fit(fit_data)

        train_x = vectorizer.transform(train_x).toarray()

        # dump
        with open("embeddings/" + name + "_" +lang + ".pk", "wb") as fout:
            pickle.dump(train_x, fout)

        with open("embeddings/tfidf_vectorizer_" + lang + ".pk", "wb") as fout:
            pickle.dump(vectorizer,fout)
        # print("TFIDF dumped on file.")

        return train_x

def tfidf_test(lang, test_x, name):

    #load
    load_data_x = Path("embeddings/" + name + "_"+lang + ".pk")

    if load_data_x.is_file():
        with open("embeddings/" + name + "_" +lang + ".pk", "rb") as fin:
            data = pickle.load(fin)

        # print("Loaded from file TFIDF.")
        return data

    else:

        with open("embeddings/tfidf_vectorizer_" + lang + ".pk", "rb") as fin:
            vectorizer= pickle.load(fin)

            test_x = vectorizer.transform(test_x).toarray()

            # dump
            with open("embeddings/" + name + "_" +lang + ".pk", "wb") as fout:
                pickle.dump(test_x, fout)

        return test_x

def bert(lang, data_x, name):

    # load
    load_data_x = Path("embeddings/" + name + "_" +lang + ".pk")

    if load_data_x.is_file():
        with open("embeddings/"+ name + "_" + lang + ".pk", "rb") as fin:
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
        with open("embeddings/" + name + "_" + lang + ".pk", "wb") as fout:
            pickle.dump(data_features, fout)

        # print("BERT dumped on file.")

        return data_features


def glove(lang, train_x, name):
    # load
    load_data_x = Path("embeddings/" + name + "_" + lang + ".pk")

    if load_data_x.is_file():
        with open("embeddings/" + name + "_" + lang + ".pk", "rb") as fin:
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
        with open("embeddings/" + name + "_" + lang + ".pk", "wb") as fout:
            pickle.dump(sentences_embeddings, fout)

        # print("GLOVE dumped on file.")
        return sentences_embeddings


def get_embeddings(lang, data, name, embeddings, testing):
    # words embedding
    if embeddings == 'tfidf':
        if(testing):
            data = tfidf_test(lang, data, name+"tfidf")
        else:
            data = tfidf(lang, data, name+"tfidf")
    elif embeddings == 'bert':
        data = bert(lang, data, name+"bert")

    elif embeddings == 'glove':
        data = glove(lang, data, name+"glove")

    elif embeddings == 'tfidf_bert':
        data_tf = tfidf(lang, data)
        data_bert = bert(lang, data, name+"bert")

        data = np.concatenate((np.array(data_tf), np.array(data_bert)), axis=1)

    elif embeddings == 'tfidf_glove':
        data_tf = tfidf(lang, data, name+"tfidf")
        data_glove = glove(lang, data, name+"glove")

        data = np.concatenate((np.array(data_tf), np.array(data_glove)), axis=1)

    elif embeddings == 'all':
        data_tf = tfidf(lang, data, name+tfidf)
        data_bert = bert(lang, data, name+"bert")
        data_glove = glove(lang, data, name+"glove")

        data = np.concatenate((np.array(data_tf), np.array(data_bert), np.array(data_glove)), axis=1)


    else:
        print("Wrong Embedding pick")
        exit(1)

    return data



def classifier(train_x, train_y, model):

    feature_selection = SelectFromModel(ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=10))

    train_x = feature_selection.fit_transform(train_x, train_y)

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


    train_x, train_y = get_data('Dataset/haspeede2_dev_taskAB.tsv')
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

    best_score = 0.0
    best_model = None
    best_embedding = None

    for m in models:
        for e in embeddings:

            embedded_train_x = get_embeddings(lang[0], train_x, "train_", e, False)
            print("Lang: " + lang[0] + "\tEmbeddings: " + e + "\tModel: " + m)
            score, model = classifier(embedded_train_x, train_y, m)

            if score > best_score:
                best_score = score
                best_model = model
                best_embedding = e

    print("BEST MODEL: ")
    print(best_model)
    print( m + " " + e + ": " + str(score))

    embedded_train_x = get_embeddings(lang[0], train_x, "train_", best_embedding, False)
    best_model.fit(embedded_train_x, train_y)

    with open("taskB/model_" + m + "_" + e + ".pk", "wb") as fout:
       pickle.dump(best_model, fout)
    print("Model saved.")

    #####TESTING     on  TWEETS    #########

    test = []
    test_x = get_test_data('Dataset/haspeede2_test_taskAB-tweets.tsv')

    for i in range(0, len(test_x)):
        sentence = preprocessing.pre_process(test_x[i].lower(), False);
        test.append(sentence)
    test_x = test

    test_x = get_embeddings(lang[0], test_x, "test_tweets_", best_embedding, True)

    test_y = best_model.predict(test_x)

    with open("taskB/test_y_tweets_" + e + ".pk", "wb") as testyout:
        pickle.dump(test_y, testyout)

    #####TESTING     on  NEWS    #########

    test = []
    test_x = get_test_data('Dataset/haspeede2-test_taskAB-news.tsv')

    for i in range(0, len(test_x)):
        sentence = preprocessing.pre_process(test_x[i].lower(), False);
        test.append(sentence)
    test_x = test

    test_x = get_embeddings(lang[0], test_x, "test_news_", best_embedding, True)

    test_y = best_model.predict(test_x)

    with open("taskB/test_y_news_" + e + ".pk", "wb") as testyout:
        pickle.dump(test_y, testyout)



    return


run()
