#-*-coding:utf-8-*-
from numpy import *

def distEclud(vecA, vecB):
    '''
    输入：向量A，向量B
    输出：两个向量的欧式距离
    描述：计算两个向量的欧式距离
    '''      
    return sqrt(sum(power(vecA - vecB, 2)))     

def randCent(dataSet, k):
    '''
    输入：数据集、K
    输出：包含K个随机质心（centroid）的集合
    描述：为给定数据集生成一个包含K个随机质心的集合
    '''       
    n = shape(dataSet)[1]          #得到数据集的列数
    centroids = mat(zeros((k,n)))  #得到一个K*N的空矩阵
    for j in range(n):             #对于每一列（对于每一个数据维度）
        minJ = min(dataSet[:,j])   #得到最小值
        rangeJ = float(max(dataSet[:,j]) - minJ) #得到当前列的范围
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #在最小值和最大值之间取值
        						   #random模块的rand(a,b)函数返回a行b列的随机数（范围：0-1）
        						   #在这里mat()加不加都无所谓，但是为了避免赋值右边的变量类
        						   #型不是matrix，还是加上比较好
    return centroids

'''randCent函数的粗暴实现
 def randCent(dataSet, k):
   
    输入：数据集、K
    输出：包含K个随机质心（centroid）的集合
    描述：为给定数据集生成一个包含K个随机质心的集合
          
    n = shape(dataSet)[1]          #得到数据集的列数
    m = shape(dataSet)[0]          #得到数据集的行数
    centroids = mat(zeros((k,n)))  #得到一个K*N的空矩阵
    for i in range(k):
        for j in range(n):             #对于每一列（对于每一个数据维度）
            minJ = min(dataSet[:,j])   #得到当前列最小值
            rangeJ = float(max(dataSet[:,j]) - minJ) #得到当前列的范围
            centroids[i,j] = minJ + rangeJ * random.rand(1) #在最小值和最大值之间取值
                                   #random模块的rand(a,b)函数返回a行b列的随机数（范围：0-1）
    return centroids

 '''
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    输入：数据集，k,计算向量间距离的函数名，随机生成k个随机质心的函数名
    输出：包含质心的集合，簇分配结果矩阵
    描述：kmeans算法实现
    '''   
    m = shape(dataSet)[0]             #数据集的行数，即数据的个数
    clusterAssment = mat(zeros((m,2)))#簇分配结果矩阵
                                      #第一列储存簇索引值
                                      #第二列储存数据与对应质心的误差
    centroids = createCent(dataSet, k)#先随机生成k个随机质心的集合
    clusterChanged = True
    while clusterChanged:             #当任意一个点的簇分配结果改变时
        clusterChanged = False
        for i in range(m):            #对数据集中的每一个数据
            minDist = inf; minIndex = -1
            for j in range(k):        #对于每一质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])#得到数据与质心间的距离
                if distJI < minDist:  #更新最小值
                    minDist = distJI; minIndex = j
            #若该点的簇分配结果改变
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):         #对于每一个簇
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#通过数组过滤得到簇中所有数据
                                                                         #.A 方法将matrix类型元素转化为array类型
                                                                         #在这里也可以不加
            centroids[cent,:] = mean(ptsInClust, axis=0) #将质心更新为簇中所有数据的均值
                                                         #axis=0表示沿矩阵的列方向计算均值
    return centroids, clusterAssment


def loadDataSet(fileName):
    '''
    输入：文件名
    输出：包含数据的列表
    描述：从以tab键分隔的txt文件中提取数据
    '''      
    dataMat = []               
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)  #将所有数据转换为float类型
        dataMat.append(fltLine)
    return dataMat


def distSLC(vecA, vecB):
    '''
    输入：两点的经纬度
    输出：两点在地球表面的距离（单位为 英里）
    描述：使用球面余弦定理计算地球表面两点间的距离
    '''   
    #pi为圆周率，在导入numpy时就会导入的了
    #sin(),cos()函数输出的是弧度为单位的数据
    #由于输入的经纬度是以角度为单位的，故要将其除以180再乘以pi转换为弧度
    #设所求点A ，纬度β1 ，经度α1 ；点B ，纬度β2 ，经度α2。则距离
    #距离 S=R·arc cos[cosβ1cosβ2cos（α1-α2）+sinβ1sinβ2]
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #6371.0为地球半径

import matplotlib
import matplotlib.pyplot as plt
def clusterPlaces(numClust=5):
    '''
    输入：希望得到的簇数目
    输出：绘制的图像
    描述：对数据进行聚类并绘制相关图像
    '''   
    datList = []
    for line in open('places.txt').readlines():               #读取文件的每一行
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])#保存经纬度到列表中
    datMat = mat(datList)
                                                              #进行聚类
    myCentroids, clustAssing = kMeans(datMat, numClust, distMeas=distSLC)
    figure_name=str(numClust)+'-cluster'
    fig = plt.figure(figure_name)                                        #创建一幅图
    rect=[0.1,0.1,0.8,0.8]                                    #创建一个矩形
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']                  #用来标识簇的标记
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')                         #基于图像创建矩阵
    ax0.imshow(imgP)                                          #绘制图像矩阵
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):                                 #在图像上绘制每一簇
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    #print(myCentroids[:,0].flatten().A[0])
    #print(myCentroids[:,1].flatten().A[0])
    plt.show()
    
if __name__ == '__main__':
	for i in range(3,7):
		clusterPlaces(i)
	
    
