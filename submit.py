import pickle
import csv

import pandas as pd
import numpy as np

def load_file(filename):
    with open(filename, "rb") as fin:
        file = pickle.load(fin)

    return file

def get_test_data(filename):
    data = pd.read_table(filename, header=None, quoting=csv.QUOTE_NONE)
    ids = data.iloc[0:, 0].values
    sentences = data.iloc[0:, 1].values
    return ids, sentences

def run():



    test_y_tweets_taskA_r1 = load_file('taskA/test_y_news_log_reg_tfidf.pk')
    test_y_tweets_taskB_r1 = load_file('taskB/test_y_news_log_reg_tfidf.pk')


    test_x_ids_tweets, test_x_tweets = get_test_data('Dataset/haspeede2-test_taskAB-news.tsv')

    print (len(test_y_tweets_taskA_r1) , len(test_y_tweets_taskB_r1) , len(test_x_ids_tweets) , len(test_x_tweets))
    print ("OK")
    with open('haspeede_news_taskAB_montanti_run1.tsv', 'w', encoding='utf-8', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='', quotechar='')
        for i in range(0, len(test_y_tweets_taskB_r1)):
            tsv_writer.writerow([test_x_ids_tweets[i], test_x_tweets[i], test_y_tweets_taskA_r1[i], test_y_tweets_taskB_r1[i]])





run()
