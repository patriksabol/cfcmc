% semantic extraction based on only similarity between clusters
% this approach was used in Chapter 4 and Chapter 6 in dissertation
% we have to have trained CFCMC classifier
%% compute similarity between clusters
disp('Computing max memberships');
classifier.MaxClusterMemberships = classifier.maxClusterMembership;
disp('Computing sim');
sim = classifier.computeSimilarity;

%% convert similarity values to semantics
classifier.ReviewClusterVector'
classifier.ReviewClassesVector'
simSemantics = classifier.similaritySemantic(sim)