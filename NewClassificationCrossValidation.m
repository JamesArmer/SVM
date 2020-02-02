function bestParams = NewClassificationCrossValidation(dataSize, feature, numOfFolds)

    %OUTPUT IN THE FOLLOWING FORMAT:
    %bestParams = [Kernel , PolynomialOrder/KernelScale , BoxConstraint]
    %Where: 
%     kernel = 1 is linear
%     kernel = 2 is polynomail
%     kernel = 3 is RBF

    [features, labels] = loadInput(dataSize, feature);
    foldSize = round(dataSize / numOfFolds);
    
    bestModelSoFar = [];
    bestAccuracy = 0;

    
   
    
    for kernel = 3:3
        hyperparamPoly = [2];
        %hyperparamBox = [0.1 0.2 0.3 0.4 0.5];
        hyperparamBox = [1 5 10 15 20];
        %hyperparamKernelScale = [1 30 60];
        hyperparamKernelScale = [40 43 46 49];
        i = length(hyperparamBox);
        j = 1;
        
        if kernel == 2
            j = length(hyperparamPoly);
        end
        if kernel ==3 
            j = length(hyperparamKernelScale);
        end 
        
        
        
        for hyperParam = 1:j
            for hyperBox = 1:i
                averageAccuracy = 0;
                averageF1 = 0;

                disp("Hyperparameters:");
                if kernel == 1
                    disp("Kernel = linear");
                end
                if kernel == 2
                    disp("Kernel = polynomial");
                    disp("PolynomialOrder = " + hyperparamPoly(hyperParam));
                end
                if kernel == 3
                    disp("Kernel = RBF");
                    disp("KernelScale = " + hyperparamKernelScale(hyperParam));
                end
                
                disp("Boxconstraint = " + hyperparamBox(hyperBox));

                for foldNumber = 0:numOfFolds-1
                    %Get index of start and end of outer fold
                    foldIndexStart = (foldNumber*foldSize +1);
                    foldIndexEnd = ((foldNumber + 1)*foldSize);

                    %Create Trainging Data
                    %TODO - FIX THE BELOW CODE TO SPLIT THE DATA CORRECTLY
                    if foldNumber == 0
                            train_X = features(foldIndexEnd+1:dataSize,:);
                            train_Y = labels(foldIndexEnd+1:dataSize);
                            test_X = features(1:foldIndexEnd,:);
                            test_Y = labels(1:foldIndexEnd);
                            %disp("train = " + (foldIndexEnd+1) + " to " + dataSize);
                            %disp("test = " + 1 + " to " + foldIndexEnd);
                        elseif foldNumber == numOfFolds - 1
                            train_X = features(1:foldIndexStart-1,:);
                            train_Y = labels(1:foldIndexStart-1,:);
                            test_X = features(foldIndexStart:dataSize,:);
                            test_Y = labels(foldIndexStart:dataSize);
                            %disp("train = " + 1 + " to " + (foldIndexStart-1));
                            %disp("test = " + foldIndexStart + " to " + dataSize);
                        else
                            train_X = [features(1:foldIndexStart-1,:);features(foldIndexEnd+1:dataSize,:)];
                            train_Y = [labels(1:foldIndexStart-1);labels(foldIndexEnd+1:dataSize)];
                            test_X = features(foldIndexStart:foldIndexEnd,:);
                            test_Y = labels(foldIndexStart:foldIndexEnd);
                            %disp("train = " + 1 + " to " + (foldIndexStart-1) + " | " + (foldIndexEnd+1) + " to " + dataSize);
                            %disp("test = " + foldIndexStart + " to " + foldIndexEnd);
                    end

                   if kernel == 1
                       Mdl = fitcsvm(train_X, train_Y, 'Standardize',true,  'KernelFunction','linear', 'BoxConstraint',hyperparamBox(hyperBox));
                   elseif kernel ==2 
                       Mdl = fitcsvm(train_X, train_Y, 'Standardize',true,  'KernelFunction','polynomial','PolynomialOrder',hyperparamPoly(hyperParam),'BoxConstraint',hyperparamBox(hyperBox));
                   else
                       Mdl = fitcsvm(train_X, train_Y, 'Standardize',true,  'KernelFunction','RBF','KernelScale',hyperparamKernelScale(hyperParam),'BoxConstraint',hyperparamBox(hyperBox));
                   end


                    [accuracy,confusionMatrix] = testModel(Mdl,test_X,test_Y);
                    tp = confusionMatrix(2,2);
                    fp = confusionMatrix(1,2);
                    fn = confusionMatrix(2,1);
                    precision = tp/(tp+fp);
                    recall = tp/(tp+fn);
                    f1Score = 2*((precision*recall)/(precision+recall));
                    %disp("accuracy within fold: " + accuracy);
                    %disp(confusionMatrix)
                    averageF1 = averageF1 + f1Score;

                    averageAccuracy = averageAccuracy + accuracy;


                end

                averageAccuracy = averageAccuracy / numOfFolds;
                averageF1 = averageF1/numOfFolds;

                disp("Average Accuracy Across Folds: " + averageAccuracy);
                disp("Average F1 Score: " + averageF1);
                disp(" ");
                
                if(averageAccuracy > bestAccuracy)
                    bestAccuracy = averageAccuracy;

                    if kernel == 1
                        bestParams = [kernel, hyperparamBox(hyperBox)];
                    end
                    if kernel == 2
                        bestParams = [kernel , hyperparamPoly(hyperParam), hyperparamBox(hyperBox)];
                    end
                    if kernel == 3
                        bestParams = [kernel, hyperparamKernelScale(hyperParam), hyperparamBox(hyperBox)];
                    end
                end
                        


            end
        end
    end
    disp("BEST PARAMETERS: " + bestParams);
    disp("BEST ACCURACY = " + bestAccuracy);

    