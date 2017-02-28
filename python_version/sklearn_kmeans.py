#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]

iris = datasets.load_iris()#导入iris数据集
X = iris.data   #得到数据
y = iris.target #得到数据的分类标签(整数0、1、2)
'''参考：
>>>y
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
#得到分类器
estimators = {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
                                              init='random')}

for name, est in estimators.items():
    fig = plt.figure(name, figsize=(6, 5)) #figsize指定图像的纵向高度和横向宽度
    #plt.clf()      #清空当前图像操作，此处可以不加
    ax = Axes3D(fig)#返回3D图形对象
    #plt.cla()      #清空当前坐标操作，此处可以不加
    est.fit(X)      #用数据对算法进行拟合操作
    labels = est.labels_#得到每一数据点的分类结果
                    #绘制散点图
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels)
                    #scatter是绘制散点图的函数，前面3个参数对应数据在x,y,z轴的坐标，c代表
                    #更多可参考：http://blog.csdn.net/u013634684/article/details/49646311


    #设置x,y,z轴的刻度标签，[]代表不描绘刻度
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    #设置x,y,z轴的标签
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')

# Plot the ground truth
fig = plt.figure('real_iris', figsize=(6, 5))
#plt.clf()
ax = Axes3D(fig)
#plt.cla()

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    #在数据集中心绘制 分类标签的名字
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',#center代表text向中间水平对齐
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))#bbox用于设置ext背景框
                       #alpha为透明度，edgecolor为边框颜色(w为white之意),facecolor为背景框内部颜色
#绘制散点图
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

#设置x,y,z轴的刻度标签，[]代表不描绘刻度
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
#设置x,y,z轴的标签
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
#显示图像，没有这句图像就显示不出来
plt.show()
