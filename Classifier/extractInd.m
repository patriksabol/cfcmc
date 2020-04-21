% trainset - input training data
% clusterInd - index of cluster
% return - indexes of clusters and corrensponding vetro of indexes of
% classes
function [clusters, classes] = extractInd(trainset,clusterInd)
% vector of clusters indexes
clusters = unique(clusterInd);
% vector of classes indexes for clusters vector
classes = zeros(length(clusters),1);

for i=1:length(clusters)
    % find first index in vector of clusters
    indCl = find(clusterInd==clusters(i),1);
    % get classes
    classes(i) = trainset(indCl,end);
end

end
