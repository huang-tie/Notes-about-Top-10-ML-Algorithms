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