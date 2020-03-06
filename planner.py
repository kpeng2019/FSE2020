from tools import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def RandomPlanner(name, par, explainer=None, smote=False, act=False):
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)
    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11)
    df2n = norm(df22)
    df3n = norm(df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    rejs = []
    score2 = []
    para = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
    else:
        clf1.fit(X_train1, y_train1)
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                       feature_names=df11.columns,
                                                       discretizer='entropy', feature_selection='lasso_path',
                                                       mode='classification')
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                 num_features=20, num_samples=1000)
                ind = ins.local_exp[1]
                temp = X_test1.values[i].copy()
                tem, plan = RandomWalk(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par, actionable=None)
                if (tem != X_test1.values[i]).any():
                    score.append(overlap(X_test1.values[i], plan, actual))
                    score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, score2


def planner(name, par, explainer=None, smote=False, act=False):
    # The planner can be classical LIME or TimeLIME
    # It returns
    # scores: a list of overlap scores between plans and future developer actions
    # bugchange: a list of NDPV for each test instance
    # score2: a list of overlap scores between plans and original code
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print(actionable)
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11)
    df2n = norm(df22)
    df3n = norm(df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    rejs = []
    score2 = []
    para = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
    else:
        clf1.fit(X_train1, y_train1)
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                       feature_names=df11.columns,
                                                       discretizer='entropy', feature_selection='lasso_path',
                                                       mode='classification')
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                #                 if False:
                if clf1.predict([X_test1.values[i]]) == 0:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20, num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    tem, plan, rej = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, 0)
                    score.append(overlap(X_test1.values[i], plan, actual))
                    score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                    rejs.append(rej)
                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                else:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20, num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    if act:
                        tem, plan, rej = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par,
                                              actionable=actionable)
                    else:
                        tem, plan, rej = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par, actionable=None)
                    if (tem != X_test1.values[i]).any():
                        score.append(overlap(X_test1.values[i], plan, actual))
                        rejs.append(rej)
                        score2.append(overlap(X_test1.values[i], plan, X_test1.values[i]))
                        bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
    #     print("Runtime:",time.time()-start_time)
        print(name[0])
        print('>>>')
        print('>>>')
        print('>>>')
    return score, bugchange, score2