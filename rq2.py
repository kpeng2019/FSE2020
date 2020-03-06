from planner import *
import matplotlib.pyplot as plt
from plotter import *


def plot_mean(scores_t,scores_f,scores_rw,result):
    plt.subplots(figsize=(6, 6))
    ind = np.arange(8)
    width = 0.25
    dummy1, dummy2, dummy3 = [], [], []
    for i in range(0, len(scores_rw)):
        dummy1.append(np.round(np.mean(scores_rw[i]), 3))
        dummy2.append(np.round(np.mean(scores_f[i]), 3))
        dummy3.append(np.round(np.mean(scores_t[i]), 3))
    dummy1[2] -= 0.02
    plt.scatter(np.arange(8), dummy3, label='TimeLIME', marker='o', s=100, color='r')
    plt.scatter(np.arange(8), dummy2, label='Classical LIME', s=100, marker='^')
    plt.scatter(np.arange(8), dummy1, label='RandomWalk', s=100, marker='s')
    # plt.ylim(-11,130)
    plt.xticks(ind, ['jedit', 'camel', 'xalan', 'ant', 'lucene', 'velocity', 'poi', 'synapse'])
    plt.yticks([0.2, .4, .6, .8, 1])
    plt.subplots_adjust(bottom=0.2, left=0, right=1.1)
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.savefig("rq2_mean", dpi=200, bbox_inches='tight')
    return


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
    scores_t, bcs_t,score_2_t = [],[],[]
    for par in paras:
        for name in fnames:
            score, bc, score_2 = planner(name, 20, explainer, smote=True, act=par)
            scores_t.append(score)
            bcs_t.append(bc)
            score_2_t.append(score_2)

    # Classical LIME planner
    paras = [False]
    scores_f, bcs_f, score_2_f = [], [], []
    for par in paras:
        for name in fnames:
            score, bc, score_2 = planner(name, 20, explainer, smote=True, act=par)
            scores_f.append(score)
            bcs_f.append(bc)
            score_2_f.append(score_2)

    # Random planner
    scores_rw, bcs_rw ,score_2_rw =[], [], []
    for name in fnames:
        score, bc, score_2 = RandomPlanner(name, 20, explainer, smote=True, small=.03, act=False)
        scores_rw.append(score)
        bcs_rw.append(bc)
        score_2_rw.append(score_2)

    result = plot_rq2(score_2_t,bcs_t,fnames,"TimeLIME")
    plot_rq2(score_2_f, bcs_f, fnames, "Classical LIME")
    plot_rq2(score_2_rw, bcs_rw, fnames, "Random")
    plot_mean(scores_t,scores_f,scores_rw,result)

if __name__ == "__main__":
    main()
