classdef CFCMC
    %CFCMC Summary
    %   class for the CFCMC classifier
    
    properties
        TrainPatterns
        TrainImgPaths
        Awidth
        ReviewClassesVector
        ReviewClusterVector
        ClusterIndexes
        ValidationSet
        ValidTrainDistances
        Threshold
        K
        %%% properties for semnatic extraction %%%
        Centroids
        Radius
        MaxClusterMemberships
        TrainMemb
        TrainClustersIndexes
    end
    
    methods
        %% Constructor
        function obj = CFCMC(trainPatterns)
            % Construct an instance of this class
            % input - trainPattern - matrix of data, where rows are number
            % of patterns, while columns are number of features plus
            % labels, thus numOfPatterns x numOfFeatures + 1
            % class labels must be integers starting on label 1
            
            % initial phase of the algorithm
            % set training patterns to class variable
            obj.TrainPatterns = trainPatterns;
            % clustering each class and find optimal number of cluster for
            % each class, could be 'CalinskiHarabasz', 'Silhouette', 'gap' or 'DaviesBouldin'.
            
            disp('Clustering data');
            obj.ClusterIndexes = intraClassClustering(trainPatterns, 'Silhouette', [1:3]);
            
            %obj.ClusterIndexes = trainPatterns(:,end); % if you want to
            %set for each class one cluster
            
            % extract indexes of all clusters and their corensponding
            % classes/labels, so f.e.
            % ReviewClusterVector = [1 2 3 4 5 6]
            % ReviewClassesVector = [1 1 2 3 3 3]
            [obj.ReviewClusterVector, obj.ReviewClassesVector] = extractInd(obj.TrainPatterns, obj.ClusterIndexes);
            % compute cetroids for each of the clusters
            obj.Centroids = obj.computeCentroids;
            disp(obj.ReviewClusterVector');
            disp(obj.ReviewClassesVector');
            % initialize vector of parameters of membership function to
            % optimize, length is the nuber of clusters
            obj.Awidth = ones(1,length(obj.ReviewClusterVector));
            obj.K = ones(1,length(obj.ReviewClusterVector));
            % threshold, below which the input os labeled as "not
            % classified"
            obj.Threshold = 0.01;
        end
        %% Setters
        function obj = set.TrainPatterns(obj,trainPatterns)
            obj.TrainPatterns=trainPatterns;
        end
        function obj = set.TrainImgPaths(obj,trainImgPaths)
            obj.TrainImgPaths=trainImgPaths;
        end
        function obj = set.Awidth(obj,eparams)
            obj.Awidth=eparams;
        end
        function obj = set.ReviewClassesVector(obj,reviewClassesVector)
            obj.ReviewClassesVector=reviewClassesVector;
        end
        function obj = set.ReviewClusterVector(obj,reviewClusterVector)
            obj.ReviewClusterVector=reviewClusterVector;
        end
        function obj = set.ClusterIndexes(obj,clusterIndexes)
            obj.ClusterIndexes=clusterIndexes;
        end
        function obj = set.ValidationSet(obj,validationSet)
            obj.ValidationSet=validationSet;
        end
        function obj = set.ValidTrainDistances(obj,distances)
            obj.ValidTrainDistances=distances;
        end
        function obj = set.Threshold(obj,threshold)
            obj.Threshold=threshold;
        end
        function obj = set.K(obj,k)
            obj.K=k;
        end
        function obj = set.Centroids(obj,centroids)
            obj.Centroids=centroids;
        end
        function obj = set.Radius(obj,radius)
            obj.Radius=radius;
        end
        function obj = set.MaxClusterMemberships(obj,maxmemb)
            obj.MaxClusterMemberships=maxmemb;
        end
        function obj = set.TrainMemb(obj,trainmemb)
            obj.TrainMemb=trainmemb;
        end
        function obj = set.TrainClustersIndexes(obj,trainClInd)
            obj.TrainClustersIndexes=trainClInd;
        end
        %% Public functions
        function [cor, unknown, incor, expOut,compOut] = evaluateClassifier(obj, testSet)
            % evaluate classifier with testing set
            % testing set - matrix of the size numOfPatterns x
            % numOfFeatures + 1 (expected labels/classes)
            testPattterns = testSet(:,1:end-1);
            % expected labels
            expOut = testSet(:,end);
            % classify testing patterns
            % [computed labels, membership values to winner label]
            [compOut,memb] = obj.classify(testPattterns,obj.Awidth, obj.K);
            % compute accuracy
            % any other metric could be used
            [cor, unknown, incor] = accuracy(expOut, compOut, memb, obj.Threshold);
        end
        
        function [solution, acc] = trainClassifier(obj, set)
            % training classifier with validation set
            obj.ValidationSet = set;
            % pre-computing distances between validation and training set
            disp('Computing distances between validation and training set....');
            obj.ValidTrainDistances = pdist2(obj.ValidationSet(:,1:end-1), obj.TrainPatterns(:,1:end-1));
            % initial value of the a - ainit, in dissertation - equation 5.4
            startPoint = mean(min(obj.ValidTrainDistances));
            maxPoint = max(obj.ValidTrainDistances(:));
            % low bound - [ainit,k = 0]
            lb = [startPoint*ones(1,length(obj.ReviewClusterVector)) zeros(1,length(obj.ReviewClusterVector))];
            % upper bound - [amax, k = 1]
            ub = [maxPoint*ones(1,length(obj.ReviewClusterVector)) ones(1,length(obj.ReviewClusterVector))];
            % genetic algorithm
            disp('Optimizing');
            options = optimoptions('ga', 'MaxGenerations', 50,...
                'MaxStallGenerations', 30, ...
                'PlotFcn' , 'gaplotbestf',... 'InitialPopulationMatrix', best,...
                'CrossoverFcn',@crossoverarithmetic,...
                'MutationFcn',@mutationadaptfeasible,...
                'PopulationSize', 20,...
                'display', 'off');%,...);
            % size of individual
            % for N position for A parameters, the rest for K
            % paramteres
            individualSize = length(obj.ReviewClusterVector)*2;
            % this give us best solution and corresponding fitness value
            [solution, acc] = ga(@obj.fitnessFun, individualSize ,[],[],[],[],lb, ub,[],[],options);
        end
        
        % X - input vectors - size = numOfVectors x numOfFeatures
        % dist - precomputed distance between X and train
        function [Y, memb, membVector, Ycluster] = classify(obj,X,Awidth,K,distances)
            % classify input pattern
            % whether precomputed distances are passed as argument or not
            % precomputed distances are used during training
            if(nargin==5)
                membVector = obj.membership(X,Awidth,K,distances);
            elseif(nargin == 4)
                membVector = obj.membership(X,Awidth,K);
            end
            % membVector - size - numOfVectors x numOfClusters
            % matrix of memberships of each input vetors to each cluster
            % choose winner cluster index
            [memb,membInd] = max(membVector,[],2);
            % winner class label
            Y = obj.ReviewClassesVector(membInd);
            % winner cluster
            Ycluster = obj.ReviewClusterVector(membInd);
        end
        
    end
    %% Private functions
    methods (Hidden)
        function acc = fitnessFun(obj,individual)
            % unfold individual
            As = individual(1:end/2);
            Ks = individual(end/2+1:end);
            % classify validation set
            % compute output labels
            [compOut, memb] = obj.classify(obj.ValidationSet, As,Ks, obj.ValidTrainDistances);
            % choose expected output
            expOut = obj.ValidationSet(:,end);
            
            % compute kappa
            %conMat = confusionmat(expOut, compOut);
            %acc = 1-kappa(conMat);
            
            % or accuracy
            [acc,~,~] = accuracy(expOut, compOut, memb, obj.Threshold);
            % because of minimalization
            acc = 1 - acc;
        end
        function err = fitnessFunError(obj,individual)
            % fitness based on error, which is not documented
            % only testing version
            % classify validation set
            As = individual(1:end/2);
            Ks = individual(end/2+1:end);
            [compOut, memb, membVectors] = obj.classify(obj.ValidationSet,As,Ks, obj.ValidTrainDistances);
            err = 0;
            % choose expected output
            expOut = obj.ValidationSet(:,end);
            clasInd = obj.ReviewClassesVector;
            for i = 1:length(compOut)
                thresholdError = obj.Threshold - memb(i);
                thresholdError(thresholdError<0) = 0;
                wrongClassifiedError = 0;
                if(compOut(i) ~= expOut(i))
                    compMemb = memb(i);
                    ind = clasInd==expOut(i);
                    expClasMemb = max(membVectors(i,ind));
                    wrongClassifiedError = compMemb/expClasMemb;
                end
                err = err + thresholdError + wrongClassifiedError;
            end
            %err = err / length(compOut);
        end
        
        % compute membership for input vectors
        % output - vector of membership to all clusters
        % X - input vectors
        % dist - precomputed distance between X and train, only for the
        % case of training, where X is validation set
        function memb = membership(obj,X,Awidth,K,distances)
            % number of arguments
            args = nargin;
            % creating matrix of memberships for each input pattern to each
            % cluster
            memb = zeros(size(X,1),length(obj.ReviewClusterVector));
            % compute cluster membership for each cluster
            for i=1:size(memb,2)
                if(args==5)
                    memb(:,i) = obj.clusterMembership(X,obj.ReviewClusterVector(i),Awidth,K,distances);
                elseif(args == 4)
                    memb(:,i) = obj.clusterMembership(X,obj.ReviewClusterVector(i),Awidth,K);
                end
            end
        end
        
        % compute membership of input vector to specific cluster clIndex
        % X - input vectors
        % clIndex - index of cluster
        % dist - precomputed distance between X and train, only for the
        % case of training, where X is validation set
        function clMemb = clusterMembership(obj, X, clIndex,Awidth,K,distances)
            % get vector of indexes of training vector belonging to cluster
            % with index clIndex
            trainVectorsForClusterInd = obj.ClusterIndexes==clIndex;
            % whether distances are passed as argument or not
            if(nargin==5)
                % if there are no pre-computed distances
                % create matrix of train patterns for clIndex cluster
                trainPat = obj.TrainPatterns(trainVectorsForClusterInd,1:end-1);
                % compute matrix of distances between X and trainPat
                distances = pdist2(X, trainPat);
                % compute cummulative membership for each input vector in X
                clMemb = obj.membershipFunction(distances,Awidth(clIndex),K(clIndex));
            elseif(nargin == 6)
                % compute cummulative membership for each input vector in X
                % with precomputed distance matrix
                clMemb = obj.membershipFunction(distances(:,trainVectorsForClusterInd),Awidth(clIndex),K(clIndex));
            end
        end
        % triangle Membership Function
        % dist - norm distance function
        % E - parameter of the function - width of the triangle
        function memb = membershipFunction(obj, dist, A, K)
            memb = 0;
            
            % triangle function
            % in dissertation equation 5.1
            memb = 1 - dist/A;
            memb(memb<0) = 0;
            
            % sum of membership K nearest training patterns / number of
            % training patterns
            % get in from K, which is from interval 0;1
            K = round(size(memb,2)*K);
            if(K == 0)
                K = 1;
            end
            % get k max elements
            maxKelements = maxk(memb,K,2);
            if(sum(memb>1)>0)
                a = 1;
            end
            % compute membership based on equation 5.2 in dissertation
            memb = sum(maxKelements,2)/size(maxKelements,2);
        end
        
        %% semantics functions
        % compute centroids for all clusters
        function centroids = computeCentroids(obj)
            clIndexes = obj.ClusterIndexes;
            X = obj.TrainPatterns(:,1:end-1);
            centroids = zeros(length(obj.ReviewClusterVector), size(X,2));
            for i = 1:size(centroids,1)
                ind = clIndexes == obj.ReviewClusterVector(i);
                centroids(i,:) = mean(X(ind,:));
            end
        end
        
        % compute max cluster membership based on equation 5.9 in the
        % dissertation
        function maxMembs = maxClusterMembership(obj)
            clInd = obj.ClusterIndexes;
            centroids = obj.Centroids;
            maxMembs = zeros(1,length(obj.ReviewClusterVector));
            for i = 1:size(centroids,1)
                ind = clInd == obj.ReviewClusterVector(i);
                trPat = obj.TrainPatterns(ind,1:end-1);
                % use distance defined by equation 5.17 in the dissertation
                distances = pdist2(centroids(i,:), trPat,'cityblock');
                distances = distances/size(trPat,2);
                maxMembs(i) = obj.membershipFunction(distances,obj.Awidth(i),obj.K(i));
            end
        end
        
        % compute estimated radiuses for all clusters based on equation
        % 5.16 in the dissertation
        function radiuses = computeRadius(obj)
            radiuses = zeros(1,length(obj.ReviewClusterVector));
            maxMembs = obj.MaxClusterMemberships;
            dim = size(obj.TrainPatterns,2) - 1;
            for i = 1:length(radiuses)
                a1 = -0.7621; b1 = -0.2799; c1 = 0.07459;
                a2 = 0.8372; b2 = -0.3729; c2 = 0.1758;
                p1 = a1*dim^b1+c1;
                p2 = a2*dim^b2+c2;
                kConstant = p1*obj.K(i) + p2;
                mLow = 0.7;
                % sometimes you have to play with these values
                % not always data follow the assumptions
                %mHigh = 0.4;
                mHigh = 2.5;
                if(maxMembs(i)<kConstant)
                    radiuses(i) = obj.Awidth(i)*(kConstant/maxMembs(i))^mLow;
                else
                    radiuses(i) = obj.Awidth(i)*(kConstant/maxMembs(i))^mHigh;
                end
            end
        end
        
        % compute similarity between clusters
        % matrix of the size numOfClusters x numOfClusters
        % values from 0;1
        function similarity = computeSimilarity(obj)
            r = obj.computeRadius;
            d = squareform(pdist(obj.Centroids));
            similarity = zeros(size(d));
            for i = 1:size(d,1)
                for j = 1:size(d,2)
                    similarity(i,j) = obj.computeRatio(r(i),r(j),d(i,j));
                end
            end
        end
        
        % compute overlap between clusters described with circle with
        % radius
        function ratio = computeRatio(obj, r1, r2, d)
            Sr1 = pi*r1^2;
            Sr2 = pi*r2^2;
            if(r1+r2<=d)
                ratio = 0;
            elseif(r1>r2&&r1-r2>=d)
                ratio = Sr2/Sr1;
            elseif(r2>r1&&r2-r1>=d)
                ratio = Sr1/Sr2;
            else
                t = sqrt((d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(-d+r1+r2));
                A = r1^2*atan2(t,d^2+r1^2-r2^2)+r2^2*atan2(t,d^2-r1^2+r2^2)-t/2;
                ratio = A/Sr1;
            end
        end
        
        % transform similarity to semantics
        % if you want to use only similarity
        function similarity = similaritySemantic(obj, value)
            similarity = strings(size(value));
            for i = 1:size(value,1)
                for j = 1:size(value,2)
                    if(value(i,j) == 0)
                        similarity(i,j) = "no";
                    elseif(value(i,j) >0 && value(i,j)<0.7)
                        similarity(i,j) = "low";
                    elseif(value(i,j) >=0.7)
                        similarity(i,j) = "high";
                  % elseif(value(i,j) >0 && value(i,j)<0.33)
                  %     similarity(i,j) = "low";
                  % elseif(value(i,j) >=0.33 && value(i,j)<0.66)
                  %     similarity(i,j) = "medium";
                  % elseif(value(i,j) >=0.66)
                  %     similarity(i,j) = "high";
                    end
                end
            end
        end
        
    end
    
end

