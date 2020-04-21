function explanation_plausability(X, classifier, sim, certaintyThreshold, maxFoM)

% classify input
[Y, ~, membVector, ~] = classifier.classify(X, classifier.Awidth, classifier.K);
% compute FoM to other clusters
FoM = factorOfMisclassification(membVector, sim, ... 
    classifier.ReviewClusterVector', classifier.ReviewClassesVector');

disp(['Input is class ', num2str(Y)]);
labels = unique(classifier.ReviewClassesVector');

for i=1:length(labels)
    % compute max FoM for each label of not winner label 
    % based on equation 7.2 in dissertation
    if(labels(i)~=Y)
        maxLabelFoM = max(FoM(classifier.ReviewClassesVector'==labels(i)));
        semPosibility = classifier.FoMSemantic(maxLabelFoM, certaintyThreshold, maxFoM);
        disp(['There is ', semPosibility, ' possibility that it could be class ', num2str(labels(i))])
    end
end