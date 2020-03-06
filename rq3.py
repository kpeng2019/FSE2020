from planner import *
import matplotlib.pyplot as plt
from plotter import *


def plot_mean(t,f,r):
    plt.subplots(figsize=(6, 6))
    plt.rcParams.update({'font.size': 14})
    plt.scatter(np.arange(8), t, label='TimeLIME', marker='o', s=100, color='r')
    plt.scatter(np.arange(8), f, label='Classical LIME', s=100, marker='^')
    plt.scatter(np.arange(8), r, label='RandomWalk', s=100, marker='s')
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.xticks(np.arange(8), ['jedit', 'camel', 'xalan', 'ant', 'lucene', 'velocity', 'xerces', 'synapse'])
    plt.yticks([0, 0.2, .4, .6, .8, 1])
    plt.subplots_adjust(bottom=0.2, left=0, right=1.1)
    plt.grid(axis='y')
    plt.savefig("scatter", dpi=200, bbox_inches='tight')
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
        score, bc, score_2 = RandomPlanner(name, 20, explainer, smote=True, act=False)
        scores_rw.append(score)
        bcs_rw.append(bc)
        score_2_rw.append(score_2)

    result1 = plot_rq3(score_2_t,bcs_t,fnames,"TimeLIME")
    result2 = plot_rq3(score_2_f, bcs_f, fnames, "Classical LIME")
    result3 = plot_rq3(score_2_rw, bcs_rw, fnames, "Random")

    ws1 = []
    for i in range(0, len(scores_t)):
        temp = 0
        for j in range(0, len(scores_t[i])):
            temp -= (bcs_t[i][j] * scores_t[i][j])
        ws1.append(np.round(temp /np.sum(result1[i]), 3))

    ws2 = []
    for i in range(0, len(scores_f)):
        temp = 0
        for j in range(0, len(scores_f[i])):
            temp -= (bcs_f[i][j] * scores_t[i][j])
        ws2.append(np.round(temp / np.sum(result2[i]), 3))

    ws3 = []
    for i in range(0, len(scores_rw)):
        temp = 0
        for j in range(0, len(scores_rw[i])):
            temp -= (bcs_rw[i][j] * scores_rw[i][j])
        ws3.append(np.round(temp / np.sum(result3[i]), 3))

    plot_mean(ws1,ws2,ws3)
    return ws1,ws2,ws3
if __name__ == "__main__":
    ws1,ws2,ws3 = main()
    print('The weighted score gained by TimeLIME in 8 projects:', ws1)
    print('The weighted score gained by classical LIME in 8 projects:', ws2)
    print('The weighted score gained by RandomWalk in 8 projects:', ws3)