function centroids = randCent(dataSet, k)
    [~, col] = size(dataSet);
    centroids = zeros(k,col);
    for j = 1:col
        minJ = min(dataSet(:,j));
        rangeJ = double(max(dataSet(:,j))-minJ);%ʹ��float()�����
        centroids(:,j) = minJ + rangeJ*rand(k,1);
    end