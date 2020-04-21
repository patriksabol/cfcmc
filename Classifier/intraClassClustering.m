function intraClassClusterVector = intraClassClustering(X,method,klist)
% intraClassClustering creates clusters of input data for each class.
%
% intraClassClustering(X,method,klist) for each class in the dataset X,
% function produces clusters using kmeans returned as vector with cluster
% identifiers.
%
% X must be an N-by-P matrix of data with one row per observation and one
%   column per variable . All rows besides last are input for the
%   classifier, last row represent corresponding class.
%
% method  is parameter used for the detection of optimal K. The value of method 
%   could be 'CalinskiHarabasz', 'Silhouette', 'gap' or 'DaviesBouldin'.
%
% klist range of parameters used for the search of optimal K. f.e [1:10]
%
intraClassClusterVector=zeros(size(X,1),1);

classes = unique(X(:,end));
clusterIncrement=0;
for i=1:length(unique(X(:,end)))
    disp(['Clustering class',num2str(i)]);
    classIdx=X(:,end)==classes(i);
    evaluation = evalclusters(X(classIdx,1:end-1),'kmeans',method,'klist',klist);
    
    if(evaluation.OptimalK>1)
        idx = kmeans(X(classIdx,1:end-1),evaluation.OptimalK);
        intraClassClusterVector(classIdx) = idx+clusterIncrement;
    else
        intraClassClusterVector(classIdx) = clusterIncrement+1;
    end
    
    clusterIncrement = clusterIncrement+evaluation.OptimalK;
end

end

