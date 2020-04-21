function factors = factorOfMisclassification(membVect, sim, reviewCluster, reviewClasses)
% function for computing Factor of Misclassification
% inputs: membVect - memberships of the input to all clusters, sim -
% similarity values

% get winner cluster and label
[~,maxIdx] = max(membVect);
winCluster = reviewCluster(maxIdx);
winClass = reviewClasses(maxIdx);
% extract similarities of the winner cluster to the rest clusters
simWin_Rest = sim(winCluster,:);

% create vector with the value of membership of winner cluster
membWinVect = zeros(1,length(simWin_Rest))+membVect(winCluster);

% compute FoM baed on equation 7.1 in dissertation
FoM = (membVect./membWinVect)+(simWin_Rest);
% compute only for clusters belongig to the other than winner class
factors = FoM.*(reviewClasses~=winClass);

end