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
    ids = data.iloc[1:, 0].values
    sentences = data.iloc[1:, 1].values
    return ids, sentences

def run():



    test_y_tweets_taskA_r1 = load_file('taskA/test_y_tweets_log_reg_tfidf.pk')
    test_y_tweets_taskB_r1 = load_file('taskB/test_y_tweets_log_reg_tfidf.pk')

    #test_y_tweets_taskA_r2 = load_file('taskA/test_y_tweets_log_reg_tfidf.pk')
    #test_y_tweets_taskB_r2 = load_file('taskB/test_y_tweets_log_reg_tfidf.pk')

    test_x_ids_news, test_x_news = get_test_data('Dataset/haspeede2-test_taskAB-news.tsv')
    test_x_ids_tweets, test_x_tweets = get_test_data('Dataset/haspeede2_test_taskAB-tweets.tsv')

    if(len(test_y_tweets_taskA_r1) == len(test_y_tweets_taskB_r1) == len(test_x_ids_tweets == len(test_x_tweets))):
        print ("OK")
        with open('output_run1.tsv', 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for i in range(0, len(test_y_tweets_taskB_r1)):
                tsv_writer.writerow([test_x_ids_tweets, test_x_tweets, test_y_tweets_taskA_r1, test_y_tweets_taskB_r1])





run()
