function [certaintyThreshold, maxFoM] = certainityThresholdCFCMC(classifier, sim)
% compute certainty threshold defined in Chapter 7.1.2 in the dissertation
% first extract validation set saved in classifier object
validX = classifier.ValidationSet(:,1:end-1);
validGT = classifier.ValidationSet(:,end);

% classify validation patterns
[Y, ~, membVector, ~] = classifier.classify(validX, classifier.Awidth, classifier.K);

% compute factors of misclassification for each validation pattern for each
% cluster except clusters of the winning class
% size of numOfValidation x numOfClusters
factorsOfMisc = zeros(size(validX,1),length(classifier.ReviewClusterVector));
for i=1:size(validX,1)
    factorsOfMisc(i,:) = factorOfMisclassification(membVector(i,:), sim, ...
        classifier.ReviewClusterVector', classifier.ReviewClassesVector');
    %disp(num2str(i));
end

% following code is based on equation 7.5 in dissertation
% pick misclassified samples
W = Y ~= validGT;

% create array of the size numOfValidation
ff = zeros(size(Y,1),1);
% for each validation pattern compute FoM to the grount truth label
for i=1:size(Y,1)
    ff(i) = factorsOfMisc(i,validGT(i));
end
% choose only patterns that were misclassified
misFF = ff(W);

% choose minimum FoM
certaintyThreshold = min(misFF);
maxFoM = max(factorsOfMisc(:));

end