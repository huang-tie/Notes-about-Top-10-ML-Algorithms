
-------------------

## **1.关于分类和聚类**

kmeans属于聚类算法中的一种。分类和聚类是不同的概念。虽然两者的目的都是**对数据进行分类**，但是却有一定的区别。

 - 分类是按照某种标准给对象贴标签，再根据标签来区分归类；
 - 聚类是事先没有给出标签，刚开始并不知道如何对数据分类，完全是算法自己来判断各条数据之间的相似性，相似的就放在一起。

在聚类的结论出来之前，不能知道每一类有什么特点，最后一定要根据聚类的结果通过人的经验来分析才能知道聚成的这一类大概有什么特点。简言之，聚类就是“物以类聚、人以群分”的原理。

## **2.基本概念**

简单来说，kmeans算法的原理就是：**给定K的值，K代表要将数据分成的类别数，然后根据数据间的相似度将数据分成K个类，也称为K个簇(cluster)**。这就是kmeans算法名字中k的由来。

度量数据相似度的方法一般是用数据点间的距离来衡量，比如欧式距离、汉明距离、曼哈顿距离等等。

**一般来说，我们使用欧式距离来度量数据间的相似性。**所谓的欧式距离便是我们日常使用的距离度量方法。比如对于二维平面上的两个点A（x1,y1）和B(x2,y2)，两者间的欧式距离就为![二维平面两点间欧式距离公式](http://img.blog.csdn.net/20170226170747944?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)。

**而对于每一个簇，我们用簇中所有点的中心来描述，该中心也称为质心（centroid）**。我们通过对簇中的所有数据点取**均值（mean）**的方法来计算质心，这也是kmeans算法名字中mean的由来。


## **3.算法描述**

### **3.1.自然语言描述**

- 1.创建K个点作为初始质心（通常是随机选择）

- 2.当任意一个点的簇分类结果发生改变时

    - 2.1对数据的每一个点，计算每一个质心与该数据点的距离，将数据点分配到距其最近的簇

    - 2.2对于每一个簇，计算簇中所有点的均值并将均值作为质心

### **3.2.伪代码描述**
 

kmeans(dataset, k)

输入：数据集dataset, k的值（大于0的整数）

输出：包含质心的集合，簇分配结果

（1）	选择k个数据点作为质心（通常是随机选取）

（2）	当任意一个点的簇分类结果发生改变时

    a)	For each 数据集中的点

            For each 质心集合的质心

                计算数据点与质心的距离

            将数据点分配到距其最近的簇，更新簇分配结果

    b)	For each簇
    
            计算簇中所有点的均值并将均值作为质心


## **4.算法实现**

### **4.1.python实现及相关解释**

注：以下代码参考自《机器学习实战》

``` python
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
```

以上便是kmeans的python实现。可以看出，代码写得是相当的简洁朴实，有许多地方是值得思考和借鉴的。

比如对于上面的randCent函数的实现，该函数的作用是：为给定数据集生成一个包含K个随机质心的集合。

那么具体要怎么实现该函数呢？对于一般的数据集，我们一般是以每一行代表一个数据样本，每一列代表数据维度（或数据属性）。K个随机质心的集合就需要K*N（N为数据维度数目，也是数据集的列数）的矩阵来存储。那么要如何要随机选取呢？

一种比较简单粗暴的方法就是：对于每个质心的N个维度的值依次进行随机选取，这样需要两个for循环来实现了，当然选取的值要保证在数据集的范围之内（若取在数据集范围之外，比如取在一个距离数据集所有样本都很远的点，那计算欧式距离还有什么意义？）。实现如下：

```python
def randCent(dataSet, k):
    '''
    输入：数据集、K
    输出：包含K个随机质心（centroid）的集合
    描述：为给定数据集生成一个包含K个随机质心的集合
    '''       
    n = shape(dataSet)[1]          #得到数据集的列数
    m = shape(dataSet)[0]  		   #得到数据集的行数
    centroids = mat(zeros((k,n)))  #得到一个K*N的空矩阵
    for i in range(k):
    	for j in range(n):             #对于每一列（对于每一个数据维度）
        	minJ = min(dataSet[:,j])   #得到当前列最小值
        	rangeJ = float(max(dataSet[:,j]) - minJ) #得到当前列的范围
        	centroids[i,j] = minJ + rangeJ * random.rand(1) #在最小值和最大值之间取值
        						   #random模块的rand(a,b)函数返回a行b列的随机数（范围：0-1）
    return centroids
```

而借用numpy矩阵的方便性，我们可以节省掉一个for循环：

```python
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
    return centroids
```

其次，在kMeans函数中，下面的这行代码可能有点难以理解，其所在的for循环的作用是：对于每一个簇，将质心更新为簇中所有数据的均值。该行代码的作用是：通过数组过滤得到下标为cent的簇中的所有数据。

```python
ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]    
```

我们可以从外到内来看这行代码。首先，最外层是：`ptsInClust =dataSet[something]`,那么我们可以反推最外层的[]符号里面的代码所起的作用便是：找到被划分在下标为cent的簇的那些数据点（样本）的下标，更为准确的，应该是这些数据点在dataSet中的行下标。那么，我们来看这行代码：

`nonzero(clusterAssment[:,0].A==cent)[0]`

注：代码中的`.A`是将matrix类型的变量转换为array变量的方法，在这里有没有都没有影响。

对于nonzero函数，其输入值为：数组或矩阵，返回输入值中非零元素的信息，这些信息中包括 两个array， 包含了相应维度上非零元素所在的行标号，与列标标号。下面举两个例子：

```python
>>> a=mat([ [1,0,0],[0,0,0],[0,0,0]])
>>> nonzero(a)
(array([0], dtype=int64), array([0], dtype=int64))
```
输出结果表示：输入矩阵a只有1个非零值。 第一个array([0], dtype=int64)表示非零元素在第0行， 第二个array([0], dtype=int64)表示在第0行的第0列。（dtype代表数据类型）

```python
>>> a=mat([[1,0,0],[1,0,0],[0,0,1]])
>>> nonzero(a)
(array([0, 1, 2], dtype=int64), array([0, 0, 2], dtype=int64))
```
输出结果表示：矩阵a只有3个非零值。 返回值行维度数据array([[0, 1, 2]], dtype=int64)表示非零元素出现在a中的第0行、第1行和第2行；返回值列维度数据array([[0, 0, 2]], dtype=int64)对应表示每行中第几列，即第0行第0列，第1行第0例，第2行第2列。

那么，``nonzero(something)[0]``的作用就是：找出something中的非零值的行维度下标组成的array。而`clusterAssment[:,0].A==cent`就是判断`clusterAssment[:,0]`中元素是否与cent相等的bool表达式，相等时对应位置为True，反之为False。举个例子：

```python
>>> a
array([3, 0, 1, 4, 5, 1])
>>> a==1
array([False, False,  True, False, False,  True], dtype=bool)
```
至此，问题解决！

#### **4.1.1对应应用(使用kmeans对地图上的点进行聚类)**
注：该例参考自《机器学习实战》
##### **一、背景介绍**
假设现在给出俄勒冈洲的波特兰地区附件的70个地点的名字和对应所在的经纬度，具体分布如下图所示。现在要求给出将这些地方进行聚类的最佳策略，从而可以安排交通工具到达这些簇的质心，然后步行到每个簇内的地点。
![这里写图片描述](http://img.blog.csdn.net/20170228085339878?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##### **二、使用数据说明**
1.70个地点的数据存放在places.txt文件中，部分截图如下。其中每一行数据的第一列代表地点的名字，第4列和第5列分别代表对应地点的纬度、经度。
![这里写图片描述](http://img.blog.csdn.net/20170228085523848?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

2.未标注地点的俄勒冈洲的波特兰地区的图片如下，命名为Portland.png：
![这里写图片描述](http://img.blog.csdn.net/20170228085622942?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##### **三、代码及相关解释**
代码如下：

```python
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
	
    
```

该份代码在上面kmeans的python实现代码的基础之上，添加了如下函数：
| 函数名| 作用  |
| ------------- |:-------------:|
| loadDataSet(fileName) | 从以tab键分隔的txt文件中提取数据 |
| distSLC(vecA, vecB) |（由于给出的地点是位于地球上（球体））使用球面余弦定理计算地球表面两点间的距离，代替原来用于计算两个向量的欧式距离函数distEclud(vecA, vecB) | 
| clusterPlaces(numClust=5) | 对数据进行聚类并绘制相关图像 （可视化结果）|

##### **运行结果**
不同簇的数据点用不同的形状标记，+号所标注的就是对应簇的质心。K=3、4、5、6时的输出图像依次为：
![k=3](http://img.blog.csdn.net/20170228093023363?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![k=4](http://img.blog.csdn.net/20170228093050676?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![k=5](http://img.blog.csdn.net/20170228093106690?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![k=6](http://img.blog.csdn.net/20170228093123411?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
仔细观察发现，聚类的效果还算可以，不同区域块的地点都大致各自被划分在一起了。但是，细心的朋友可能会发现：上面的图“3-cluster”本应显示3个“+”号（3个不同形状的“小部落”），但是缺只显示了2个。图“5-cluster”、图“6-cluster”也有出现类似的情况。随后，笔者再次运行上面的代码，由于算法中的初始质心是随机选取的缘故，每次的结果的不一样，将每次聚类的结果所得的质心都输出来：

```python
#k=3时质心的经度
[-122.74941346 -122.54432282 -122.66436586]
#k=3时质心的纬度
[ 45.545862    45.52042118  45.48861296]

#k=4时质心的经度
[-122.7718838  -122.52181039 -122.69274678 -122.63248475]
#k=4时质心的纬度
[ 45.5345341   45.50142783  45.43529156  45.5331405 ]

#k=5时质心的经度
[-122.72072414 -122.62813707 -122.4009285  -122.7288018  -122.53692062]
#k=5时质心的纬度
[ 45.59011757  45.52791831  45.46897     45.45886713  45.50548506]

D:\Program Files\Anaconda3\lib\site-packages\numpy\core\_methods.py:59: RuntimeWarning: Mean of empty slice.
  warnings.warn("Mean of empty slice.", RuntimeWarning)
D:\Program Files\Anaconda3\lib\site-packages\numpy\core\_methods.py:68: RuntimeWarning: invalid value encountered in true_divide
  ret, rcount, out=ret, casting='unsafe', subok=False)

#k=6时质心的经度
[-122.59320258 -122.52181039 -122.7003585  -122.842918             nan
 -122.68842409]
 #k=6时质心的纬度
[ 45.55155133  45.50142783  45.58066533  45.646831            nan
  45.48668819]
```
[注]：为了方便理解数据，上面的注释是人工添加上的
可以看到，在这次运行中，当K=3、4、5时，运行结果正常，质心的坐标数值没出现如K=6时的nan值。当K=6时，运行报错，第五个质心的经纬度的计算出错 (详见上面的报错信息)。

所以，由于初始随机质心选取的缘故，在使用上面的代码时，可通过多次测试来找到没有报错的情况。

说到测试，我这里再运行一次代码，运行后质心的结果如下，注释我就不打了，参考上面的就行了。

```python
[-122.7255474  -122.61159983 -122.50706986]
[ 45.51717564  45.5133606   45.50135379]
[-122.55327667 -122.4009285  -122.7288018  -122.66332621]
[ 45.515949    45.46897     45.45886713  45.54090854]
[-122.761804   -122.49685582 -122.62341067 -122.5784446  -122.68897476]
[ 45.46639582  45.49891482  45.4787138   45.5447184   45.55172129]
[-122.76690133 -122.78288433 -122.65587515 -122.52181039 -122.60946453
 -122.70508429]
[ 45.612314    45.4942305   45.5080271   45.50142783  45.5579654
  45.42730186]
```
这次测试便没有报错，所以还是可能找到没有报错的情况的，可放心使用这份代码。

### **4.2.基于python-sklearn的应用**
scikit-learn的kmeans函数参数的解释，看[这里](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
以下代码参考自：[K-means Clustering](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#)，这份代码所做的事情是：将sklearn的kmeans函数应用于iris数据集上。关于该数据集，可以参考如下图片：
![这里写图片描述](http://img.blog.csdn.net/20170227215507902?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
由于数据集中数据有花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性，为了演示数据在3维空间的聚类效果，代码中只选择分析其中3个属性。

```python
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
```

### **4.3.matlab实现及相关解释**
关于kmeans的matlab实现，网上已经有很多人给出了。本文的kmeans的matlab实现为上述的python实现的matlab版本。

distEclud函数为：
```matlab
function dist = distEclud(vecA, vecB)
    dist = norm(vecA-vecB,2);
```
randCent函数为：

```matlab
function centroids = randCent(dataSet, k)
    [~, col] = size(dataSet);
    centroids = zeros(k,col);
    for j = 1:col
        minJ = min(dataSet(:,j));
        rangeJ = double(max(dataSet(:,j))-minJ);%使用float()会出错
        centroids(:,j) = minJ + rangeJ*rand(k,1);
    end
```
kmeans函数（命名为mykmeans）为：

```matlab
function [centroids,clusterAssment]=mykmeans(dataSet,k)
    if (nargin==2)
        distMeas=@distEclud;
        createCent=@randCent;
    elseif (nargin==3)
        createCent=@randCent;
    end
    [m,~] = size(dataSet);
    clusterAssment = zeros(m,2);
    centroids = createCent(dataSet,k);
    clusterChanged = true;
    while clusterChanged
        clusterChanged = false;
        for i = 1:m
            minDist = inf; minIndex = -1;
            for j = 1:k
                distJI = distMeas(centroids(j,:),dataSet(i,:));
                if distJI < minDist
                    minDist = distJI; minIndex = j;
                end
            end
            if clusterAssment(i,1) ~= minIndex %使用clusterAssment(i,0)会出错，matlab中下标从1开始
                clusterChanged = true;
            end
            clusterAssment(i,:) = [minIndex,minDist.*minDist];
        end
        for cent = 1:k
            ptsInClust = dataSet(clusterAssment(:,1)==cent,:);
            centroids(cent,:) = mean(ptsInClust, 1);
        end
    end

```
#### **4.3.1对应应用**
参考[这篇文章](http://www.cnblogs.com/tiandsp/archive/2013/04/24/3040883.html)的测试代码，上述matlab程序的测试代码如下：

```matlab
close all;
clc;
%第一类数据
mu1=[0 0 0];  %均值
S1=[0.3 0 0;0 0.35 0;0 0 0.3];  %协方差
data1=mvnrnd(mu1,S1,100);   %产生高斯分布数据

%%第二类数据
mu2=[1.25 1.25 1.25];
S2=[0.3 0 0;0 0.35 0;0 0 0.3];
data2=mvnrnd(mu2,S2,100);

%第三个类数据
mu3=[-1.25 1.25 -1.25];
S3=[0.3 0 0;0 0.35 0;0 0 0.3];
data3=mvnrnd(mu3,S3,100);

%显示数据
plot3(data1(:,1),data1(:,2),data1(:,3),'+');
hold on;
plot3(data2(:,1),data2(:,2),data2(:,3),'r+');
plot3(data3(:,1),data3(:,2),data3(:,3),'g+');
grid on;

%三类数据合成一个不带标号的数据类
data=[data1;data2;data3];   %这里的data是不带标号的
%k-means聚类
[centroids, clusterAssment]=mykmeans(data,3);  %最后产生带标号的数据，标号在所有数据的最后，意思就是数据再加一维度

[m, ~]=size(clusterAssment);

%最后显示聚类后的数据
figure;
hold on;
%绘制 质心的位置
plot3(centroids(:,1),centroids(:,2),centroids(:,3),'kd','MarkerSize',14);
for i=1:m 
    if clusterAssment(i,1)==1   
         plot3(data(i,1),data(i,2),data(i,3),'ro');
    elseif clusterAssment(i,1)==2
         plot3(data(i,1),data(i,2),data(i,3),'go');
    else 
         plot3(data(i,1),data(i,2),data(i,3),'bo');
    end
end
grid on;

```matlab
输出结果：
![这里写图片描述](http://img.blog.csdn.net/20170228140713212?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
左边为原数据的分布（由不同颜色区分），右图是使用kmeans划分得到的数据分布（中间的黑色菱形代表簇的质心）。

右边那张图实际上是三维的，实际上我们可以通过点击图像上方的“Rotate 3D”按钮将图像旋转至3D视图![这里写图片描述](http://img.blog.csdn.net/20170228141045789?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

效果如下：
![这里写图片描述](http://img.blog.csdn.net/20170228141250025?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

可以看到，聚类的效果是相当不错的。
### **4.4基于SPSS MODELER的应用**

软件版本：14.1(破解版)

这里以“Demos”文件夹的“DRUGIn”文件为例，使用modeler自带的kmeans算法进行分析。

DRUGIn”文件的导入：

 1. 点击软件界面下方的“ 源”
 2. 点击可变文件
 3. 选择DRUGIn文件（路径位于: IBM\SPSS\Modeler\14\Demos）

随后，双击“建模”界面下的K-measns，如下图：

![这里写图片描述](http://img.blog.csdn.net/20170227161620072?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

点击![这里写图片描述](http://img.blog.csdn.net/20170227161710838?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)运行，可得到如下结果：
![这里写图片描述](http://img.blog.csdn.net/20170227161950034?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
结果提示：字段的指定类型不足，这就要求在可变文件之后修改字段类型，这里选用“字段类型”的“类型”。
![这里写图片描述](http://img.blog.csdn.net/20170227162251715?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
运行后，结果如下：
![这里写图片描述](http://img.blog.csdn.net/20170227162416696?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
右键![这里写图片描述](http://img.blog.csdn.net/20170227162508216?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)，选择“编辑”，可得聚类结果：
![这里写图片描述](http://img.blog.csdn.net/20170227162634748?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
点击“视图”，可更改视图类型：
![这里写图片描述](http://img.blog.csdn.net/20170227162908796?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGluemNoMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
参考自：http://www.docin.com/p-598480949.html


## **5.算法分析**

优点：算法简单，容易理解和实现

缺点：需要提前确定k值，算法根据k的取值会有不同的表现；无法确定哪个属性对聚类的贡献更大；由于使用均值来计算质心	，故对于异常值敏感。

适用数据范围： 数值型数据，这是由于算法涉及计算数据间距离的操作，而只有数值型数据可以进行该操作。而对于只在有限目标集中取值的标称型数据（如true与false）需要通过一些手段映射为数值型数据才可使用该算法。


---------

