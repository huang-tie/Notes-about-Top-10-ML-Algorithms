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
