import sklearn
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    return X, y

def classify(X, y):
    #创建分类器对象
    clf = linear_model.LogisticRegressionCV()
    #用训练数据拟合分类器模型
    clf.fit(X, y)
    return clf

def plot_decision_boundary(pred_func, X, y):
    # 设置最小值和最大值，并给它一些填充
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    #生成一个网格，其中点之间的距离为h
    #把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 预测整个函数值
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 填充等高线
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def visualize(X, y, clf):
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.title("Logistic Regression")

def main():
    X, y = generate_data()
    clf = classify(X, y)
    visualize(X, y, clf)

if __name__ == '__main__':
     main()