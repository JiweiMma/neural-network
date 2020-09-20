# 生成数据集并绘制出来
import sklearn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    return X, y

def visualize(X, y):
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

def main():
    X, y = generate_data()
    visualize(X, y)

if __name__ == '__main__':
     main()