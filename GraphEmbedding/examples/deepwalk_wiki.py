
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings):
    #导入每个点的lable
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    #使用embedding构建一个分类器来预测最终输出效果
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('../data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    #使用TSNE将128维的embedding降低到2维
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])#使用networkx将图导入，无向图，结点无类型，权重
    nx.draw(G,node_size=10,font_size=10,font_color="blue",font_weight="bold")#获取原始图
    plt.show()
    print(G)
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)#随机游走长度：10，次数：80，进程数：1
    model.train(window_size=5, iter=3,embed_size=128)
    embeddings = model.get_embeddings()
    print(embeddings)
    evaluate_embeddings(embeddings)#评估函数
    plot_embeddings(embeddings)#画图函数
