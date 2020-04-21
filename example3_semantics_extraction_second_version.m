% semantic extraction based factor of misclassification
% this approach was used in Chapter 7 in dissertation
% we have to have trained CFCMC classifier
%% compute similarity between clusters
disp('Computing max memberships');
classifier.MaxClusterMemberships = classifier.maxClusterMembership;
disp('Computing sim');
sim = classifier.computeSimilarity;

%% compute certainty threshold and maximum FoM
[certaintyThreshold, maxFoM] = certainityThresholdCFCMC(classifier, sim);

%% generate plausability explanation 
testIdx = 1;
testPattern = testSet(testIdx,1:end-1);
testY = testSet(testIdx,end);
explanation_plausability(testPattern, classifier, sim, certaintyThreshold, maxFoM)
