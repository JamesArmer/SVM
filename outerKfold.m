function [accuracy, confusion] = outerKfold(m, t, size, class)
    %Inputs:
    %m = model to run, 1=linear, 2=gaussian, 3=polynomial
    %t = type of model to run, 1=classifier, 2=regression
    %size = data size
    %class = class
    
    [features, labels] = loadInput(size, class);
    foldSize = round(size/10);
    
    accuracy = zeros(1,10);
    confusion = zeros(1,10);
    
    for fold = 0:9
        disp("Fold " + fold);
        %get indexes of training section
        if(fold == 0)
            trainF = features(1:9*foldSize,:);
            trainL = labels(1:9*foldSize,:);
            testF = features(9*foldSize + 1:size,:);
            testL = labels(9*foldSize + 1:size,:);
        elseif(fold == 1)
            trainF = features(foldSize + 1:size,:);
            trainL = labels(foldSize + 1:size,:);
            testF = features(1:foldSize,:);
            testL = labels(1:foldSize,:);
        else
            trainStart = foldSize*fold + 1;
            trainEnd = (trainStart + 9*foldSize) - size - 1;
            trainF = cat(1,features(1:trainEnd,:),features(trainStart:size,:));
            trainL = cat(1,labels(1:trainEnd,:),labels(trainStart:size,:));
            testF = features(trainEnd+1:trainStart-1,:);
            testL = labels(trainEnd+1:trainStart-1,:);
        end
            
        params = []; %placeholder
        if(t == 1)
            %Classification
            %params = NewClassificationCrossValidation(10000,1,10); %no m
            %hard code results for speed
            if(m == 1) params = [1]; end %was 0.1
            if(m == 2) params = [186, 49]; end
            if(m == 3) params = [150, 2]; end
            model = buildModel(m, t, params, trainF, trainL);
            disp("Number of support vectors in model: " + length(model.Alpha));
            disp("Testing model")
            [r, conf] = testModel(model, testF, testL);
            tp = conf(2,2);
            fp = conf(1,2);
            fn = conf(2,1);
            precision = tp/(tp+fp);
            recall = tp/(tp+fn);
            f1Score = 2*((precision*recall)/(precision+recall));
            elseif(t == 2)
            %Regression
            %params = InnerCrossfoldRegression(10000,1,10); %no m
            %hard code results for speed
            if(m == 1) params = [0.05, 0.1]; end
            if(m == 2) params = [1, 0.25, 0.05]; end
            if(m == 3) params = [1, 2, 0.25]; end
            model = buildModel(m, t, params, trainF, trainL);
            disp("Number of support vectors in model: " + length(model.Alpha));
            disp("Testing model")
            r = testModelRegression(model, testF, testL);
            f1Score = 0; %not needed for regression
        else
            disp("Model invalid");
        end
        
        accuracy(fold + 1) = r;
        confusion(fold + 1) = f1Score;
    end
end