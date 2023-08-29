import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from sklearn import preprocessing
import umap


def zadanie_1():
    zoo = pd.read_csv('C:\\Users\\Maksim\\Desktop\\learning\\7_semestr\\bigData\\pr_5\\fashion-mnist_test.csv')
    Z = zoo.drop(['label'], axis=1)
    print(Z)
    T = TSNE(n_components=2,perplexity=25,random_state=123)
    TSNE_features = T.fit_transform(Z)
    print(TSNE_features[1:4,:])
    ZOO = Z.copy()
    ZOO['x'] = TSNE_features[:,0]
    ZOO['y'] = TSNE_features[:,1]

    fig = plt.figure()
    sns.scatterplot(x='x', y='y', hue=zoo['label'], data=ZOO, palette='bright')
    plt.show()




def zadanie_2():
    zoo = pd.read_csv('C:\\Users\\Maksim\\Desktop\\learning\\7_semestr\\bigData\\pr_5\\fashion-mnist_test.csv')
    Z = zoo.drop(['label'], axis=1)
    scaler = preprocessing.MinMaxScaler()
    Z = pd.DataFrame(scaler.fit_transform(Z), columns=Z.columns)
    print(Z)
    T = TSNE(n_components=2, perplexity=5, random_state=123)
    TSNE_features = T.fit_transform(Z)

    TSNE_features[1:4, :]
    ZOO = Z.copy()
    ZOO['x'] = TSNE_features[:, 0]
    ZOO['y'] = TSNE_features[:, 1]

    n_n = (5, 25, 50)
    m_d = (0.1, 0.6)

    um = dict()
    for i in range(len(n_n)):
        for j in range(len(m_d)):
            um[(n_n[i], m_d[j])] = (umap.UMAP(n_neighbors=n_n[i], min_dist=m_d[j], random_state=123).fit_transform(ZOO))




if __name__ == '__main__':
    z = int(input("Введите номер задания 1(t-SNE) или 2(UMAP): "))
    if z == 1:
        zadanie_1()
    elif z == 2:
        zadanie_2()
    else:
        print("ERROR! Не верный ввод! Доступный ввод '1' или '2'!")
