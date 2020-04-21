%% load data
data = csvread('bupa.csv');
data = normalizeData(data);
%% split data
[trainSet, validSet, testSet] = splitDataset(data,0.6,0.2,0.2);

%% training CFCMC
% create object with trainSet
classifier = CFCMC(trainSet);
% training classifier
[bestSolution, accValid] = classifier.trainClassifier(validSet);
classifier.ValidationSet = validSet;
% we have to set object parameters
classifier.Awidth = bestSolution(1:end/2);
classifier.K = bestSolution(end/2+1:end);

%% evaluate classifier
[cor, unknown, incor ,expOutTest,compOutTest] = classifier.evaluateClassifier(testSet);
[corVal, ~, ~,expOutValid,compOutValid] = classifier.evaluateClassifier(validSet);
[corTr, ~, ~,expOutTrain,compOutTrain] = classifier.evaluateClassifier(trainSet);
 
[expOutTest,compOutTest] =  convert_classes_confusion_mat(expOutTest,compOutTest);
[expOutValid,compOutValid] =  convert_classes_confusion_mat(expOutValid,compOutValid);
[expOutTrain,compOutTrain] =  convert_classes_confusion_mat(expOutTrain,compOutTrain);

plotconfusion(expOutTrain, compOutTrain, 'Training set',...
              expOutValid, compOutValid, 'Validation set',...
              expOutTest, compOutTest, 'Testing set',...
              [expOutTrain expOutValid expOutTest],...
              [compOutTrain compOutValid compOutTest],...
              'Whole data');
          

