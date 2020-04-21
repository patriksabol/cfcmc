function [trainset, validset, testset, trainInd] = splitDataset(X,trainPerc,validPerc,testPerc)

% generate indexes
[trainInd,valInd,testInd] = dividerand(size(X,1),trainPerc,validPerc,testPerc);

% create zero matrices
dim = size(X,2);
trainset = zeros(length(trainInd),dim);
validset = zeros(length(valInd),dim);
testset  = zeros(length(testInd),dim);

% fill in input vector based on indexes
% training set
for i=1:length(trainInd)
    trainset(i,:) = X(trainInd(i),:);
end
% validation set
for i=1:length(valInd)
    validset(i,:) = X(valInd(i),:);
end
% testing set
for i=1:length(testInd)
    testset(i,:) = X(testInd(i),:);
end

end