from planner import *
import matplotlib.pyplot as plt


def main():
    # TimeLIME planner
    paras = [True]
    explainer = None
    fnames = [['jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv'],
              ['camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
              ['xalan-2.5.csv', 'xalan-2.6.csv', 'xalan-2.7.csv'],
              ['ant-1.5.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
              ['lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv'],
              ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
              ['poi-1.5.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
              ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv']
              ]
    scores, bcs,score_2_t = [],[],[]
    for par in paras:
        for name in fnames:
            score, bc, score_2 = planner(name, 20, explainer, smote=True, act=par)
            scores.append(score)
            bcs.append(bc)
            score_2_t.append(score_2)

    # Classical LIME planner
    paras = [False]
    explainer = None
    fnames = [['jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv'],
              ['camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
              ['xalan-2.5.csv', 'xalan-2.6.csv', 'xalan-2.7.csv'],
              ['ant-1.5.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
              ['lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv'],
              ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
              ['poi-1.5.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
              ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv']
              ]
    scores2, bcs2, score_2_f = [], [], []
    for par in paras:
        for name in fnames:
            score, bc, score_2 = planner(name, 20, explainer, smote=True, act=par)
            scores2.append(score)
            bcs2.append(bc)
            score_2_f.append(score_2)

    # Random planner
    scores_rw, bcs_rw ,score_2_rw =[], [], []
    for name in fnames:
        score, bc, score_2 = RandomPlanner(name, 20, explainer, smote=True, small=.03, act=False)
        scores_rw.append(score)
        bcs_rw.append(bc)
        score_2_rw.append(score_2)

    means2_t = [np.mean(each) for each in score_2_t]
    means2_f = [np.mean(each) for each in score_2_f]
    means2_rw = [np.mean(each) for each in score_2_rw]
    size1, size2, size3 = [], [], []
    for i in range(0, 8):
        size1.append(20 - means2_t[i] * 20)
        size2.append(20 - means2_f[i] * 20)
        size3.append(20 - means2_rw[i] * 20)
    # print('Mean size of TimeLIME plans:',size1)
    # print('Mean size of classical LIME plans:', size2)
    # print('Mean size of Random plans:', size3)
    plt.subplots(figsize=(9, 6))
    plt.rcParams.update({'font.size': 14})
    ind = np.arange(8)
    plt.xticks(ind, ['jedit', 'camel', 'xalan', 'ant', 'lucene', 'velocity', 'poi', 'synapse'])
    plt.scatter(np.arange(8), size1, marker='o', label='TimeLIME', s=100, color='r')
    plt.scatter(np.arange(8), size2, marker='^', label='Classical LIME', s=100)
    plt.scatter(np.arange(8), size3, marker='s', label='RandomWalk', s=100)
    plt.yticks([0, 2, 4, 6, 8, 10, 12])
    # plt.subplots_adjust(bottom=0.2,left=0,right=1.1)
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.savefig("rq1", dpi=200, bbox_inches='tight')
    return size1,size2,size3
if __name__ == "__main__":
    size1,size2,size3 = main()
    print('Mean size of TimeLIME plans:', size1)
    print('Mean size of classical LIME plans:', size2)
    print('Mean size of Random plans:', size3)
