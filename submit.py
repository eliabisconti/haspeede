import pickle

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def load_file(filename):
    with open(filename, "rb") as fin:
        file = pickle.load(fin)

    return file

def stampa(lab, cat, tar, file):
    # scrittura risultati
    file.write('ID  ')
    file.write("\t")
    file.write('Misoginy')
    file.write("\t")
    file.write('Category')
    file.write('\t')
    file.write('Target')
    file.write('\n')

    for i in range(0, len(lab)):
        # id
        file.write(str(1 + i))
        file.write("\t")
        # label
        file.write(str(lab[i]))
        file.write("\t")
        # category
        tmp = ' '
        if cat[i] == 1:
            tmp = 'stereotype'
        elif cat[i] == 2:
            tmp = 'dominance'
        elif cat[i] == 3:
            tmp = 'derailing'
        elif cat[i] == 4:
            tmp = 'sexual_harassment'
        elif cat[i] == 5:
            tmp = 'discredit'
        elif cat[i] == 0:
            tmp = '0'
        file.write(tmp)
        file.write("\t")
        # target
        if tar[i] == 1:
            tmp = 'active'
        elif tar[i] == 2:
            tmp = 'passive'
        else:
            tmp = '0'
        file.write(tmp)
        file.write("\n")


def run():

    model_ita_lab = load_file('TaskA/model_italian_log_reg_tfidf_glove.pk')
    model_ita_cat = load_file('taskBrun1/model_cat_italian_Random_Forest_tfidf.pk')
    model_ita_tar = load_file('taskBrun1/model_tar_italian_Random_Forest_tfidf.pk')



    test_x_ita_lab = load_file('TaskA/test_x_italian_tfidf_glove.pk')
    test_y_ita_lab = load_file('TaskA/test_y_italian.pk')


    test_x_ita_cat = load_file('taskBrun1/test_x_cat_italian_tfidf.pk')
    test_y_ita_cat = load_file('taskBrun1/test_y_cat_italian.pk')



    test_x_ita_tar = load_file('taskBrun1/test_x_tar_italian_tfidf.pk')
    test_y_ita_tar = load_file('taskBrun1/test_y_tar_italian.pk')


    #ita
    res_label = model_ita_lab.predict(test_x_ita_lab)
    score_lab = model_ita_lab.score(test_x_ita_lab, test_y_ita_lab)

    res_category = model_ita_cat.predict(test_x_ita_cat)
    score_cat = model_ita_cat.score(test_x_ita_cat, test_y_ita_cat)

    res_target = model_ita_tar.predict(test_x_ita_tar)
    score_tar = model_ita_tar.score(test_x_ita_tar, test_y_ita_tar)

    f_res = open("res_italian.txt", 'w')

    stampa(res_label, res_category, res_target, f_res)
    print("\nItalian Results saved on file.")
    
    scores_cat_divisi = f1_score(res_category, test_y_ita_cat, average=None, labels=[1, 2, 3, 4, 5])
    scores_cat = f1_score(res_category, test_y_ita_cat, average='macro', labels=[1, 2, 3, 4, 5])

    print('Categories F1 - Scores:')
    print(scores_cat_divisi)
    print('Avarage:')
    print(scores_cat)

    print('')
    scores_tar_divisi = f1_score(res_target, test_y_ita_tar, average=None, labels=[1, 2])
    scores_tar = f1_score(res_target, test_y_ita_tar, average='macro', labels=[1, 2])

    print('Targets F1 - Scores:')
    print(scores_tar_divisi)
    print('Avarage:')
    print(scores_tar)
    print('')
    tot_score = (scores_cat + scores_tar) / 2
    print('F1 Mean: ')
    print(tot_score)



run()
