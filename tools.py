import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.ensemble import RandomForestClassifier
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from scipy import stats
import collections

def prepareData(fname):
    # print(os.path.join())
    file = os.path.join("Data",fname)
    df = pd.read_csv(file,sep=',')
    for i in range(0,df.shape[0]):
        if df.iloc[i,-1] >0:
            df.iloc[i,-1]  = 1
        else:
            df.iloc[i,-1]  = 0
    return df


def bugs(fname):
    file = os.path.join("Data", fname)
    df = pd.read_csv(file,sep=',')
    return df.iloc[:,-1]

def translate1(sentence, name):
    # do not aim to change the column
    lst = sentence.strip().split(name)
    left, right = 0, 0
    if lst[0] == '':
        del lst[0]
    if len(lst) == 2:
        if '<=' in lst[1]:
            aa = lst[1].strip(' <=')
            right = float(aa)
        elif '<' in lst[1]:
            aa = lst[1].strip(' <')
            right = float(aa)
        if '<=' in lst[0]:
            aa = lst[0].strip(' <=')
            left = float(aa)
        elif '<' in lst[0]:
            aa = lst[0].strip(' <')
            left = float(aa)
    else:
        if '<=' in lst[0]:
            aa = lst[0].strip(' <=')
            right = float(aa)
            left = 0
        elif '<' in lst[0]:
            aa = lst[0].strip(' <')
            right = float(aa)
            left = 0
        if '>=' in lst[0]:
            aa = lst[0].strip(' >=')
            left = float(aa)
            right = 1
        elif '>' in lst[0]:
            aa = lst[0].strip(' >')
            left = float(aa)
            right = 1
    return left, right


def translate(sentence, name):
    flag = 0
    threshold = 0
    lst = sentence.strip().split(name)
    #     print('LST',lst)
    if lst[0] == '':
        del lst[0]
    if len(lst) == 2:
        if '<=' in lst[1]:
            flag = 1
            aa = lst[1].strip(' <=')
            threshold1 = float(aa)
        elif '<' in lst[1]:
            flag = 1
            aa = lst[1].strip(' <')
            threshold1 = float(aa)
        if '<=' in lst[0]:
            flag = -1
            aa = lst[0].strip(' <=')
            threshold0 = float(aa)
        elif '<' in lst[0]:
            flag = -1
            aa = lst[0].strip(' <')
            threshold0 = float(aa)
        if threshold0 == 0:
            result = threshold1
            flag = 1
        elif (1 - threshold1) >= (threshold0 - 0):
            result = threshold1
            flag = 1
        else:
            result = threshold0
            flag = -1
    else:
        if '<=' in lst[0]:
            flag = 1
            aa = lst[0].strip(' <=')
            threshold = float(aa)
        elif '<' in lst[0]:
            flag = 1
            aa = lst[0].strip(' <')
            threshold = float(aa)
        if '>=' in lst[0]:
            flag = -1
            aa = lst[0].strip(' >=')
            threshold = float(aa)
        elif '>' in lst[0]:
            flag = -1
            aa = lst[0].strip(' >')
            threshold = float(aa)
        result = threshold
    return flag, result


def flip(data_row, local_exp, ind, clf, cols, n_feature=3, actionable=None):
    counter = 0
    rejected = 0
    cache = []
    trans = []
    # print("B4:", np.round(clf.predict_proba([data_row])[0][0], 3))
    # Store feature index in cache.
    cnt = []
    for i in range(0, len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])
        if ind[i][1] > 0:
            cnt.append(i)
    #         if counter == n_feature:
    #             break
    tem = data_row.copy()
    #     if n_feature>len(trans):
    #         n_feature = len(trans)
    #     result = [[0,0]]*20
    result = [[0 for m in range(2)] for n in range(20)]
    for j in range(0, len(local_exp)):
        act = True
        if actionable:
            if actionable[j] == 0:
                act = False
        if j in cnt and counter < n_feature and act:
            # features needed to be altered
            print('Column:', trans[j][0])
            sig, num = translate(trans[j][0], cols[cache[j][0]])
            num = np.round(num, 3)
            if num == 0:
                l, r = translate1(trans[j][0], cols[cache[j][0]])
                result[cache[j][0]][0], result[cache[j][0]][1] = l, r
            elif sig == 1:
                #                 tem[cache[j][0]] = (num+1)/2
                if num <= .9:
                    tem[cache[j][0]] = num + .1
                else:
                    tem[cache[j][0]] = (num + 1) / 2
                result[cache[j][0]][0], result[cache[j][0]][1] = num, 1
            else:
                #                 tem[cache[j][0]] = num/2
                if num >= .1:
                    tem[cache[j][0]] = num - 0.1
                else:
                    tem[cache[j][0]] = (num + 0) / 2
                result[cache[j][0]][0], result[cache[j][0]][1] = 0, num
            print('Changed value:', tem[cache[j][0]])
            counter += 1
        else:
            if act == False:
                rejected += 1
            l, r = translate1(trans[j][0], cols[cache[j][0]])
            result[cache[j][0]][0], result[cache[j][0]][1] = l, r
    # print("Now:", np.round(clf.predict_proba([tem])[0][0], 3))
    return tem, result, rejected


def hedge(arr1,arr2):
    # return true is delta is smaller than small, which means no difference
    s1,s2 = np.std(arr1),np.std(arr2)
    m1,m2 = np.mean(arr1),np.mean(arr2)
    n1,n2 = len(arr1),len(arr2)
    num = (n1-1)*s1**2 + (n2-1)*s2**2
    denom = n1+n2-1-1
    sp = (num/denom)**.5
    delta = np.abs(m1-m2)/sp
    c = 1-3/(4*(denom)-1)
    return delta*c


def norm (df):
    # min-max scale the dataset
    X1 = df.iloc[:,:-1].values
    mm = MinMaxScaler()
    mm.fit(X1)
    X1 = mm.transform(X1)
    df1 = df.copy()
    df1.iloc[:,:-1] = X1
    return df1


def overlap(ori,plan,actual): # Jaccard similarity function
    cnt = 20
    right = 0
    for i in range(0,len(plan)):
        if actual[i]>=plan[i][0] and actual[i]<=plan[i][1]:
            right+=1
    return right/cnt


def RandomWalk(data_row,local_exp,ind,clf,cols,n_feature = 3,actionable = None):
    cache = []
    trans = []
    for i in range(0,len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])
    tem = data_row.copy()
    result =  [[ 0 for m in range(2)] for n in range(20)]
    for j in range(0,len(local_exp)):
        if np.random.rand(1)[0]>0.5:
            num1 = np.random.rand(1)[0]/2
            num2 = np.random.rand(1)[0]/2+.5
            result[cache[j][0]][0],result[cache[j][0]][1] = num1,num2
            tem[cache[j][0]]=(num1+num2)/2
        else:
            l,r = translate1(trans[j][0],cols[cache[j][0]])
            result[cache[j][0]][0],result[cache[j][0]][1] = l,r
    return tem,result


