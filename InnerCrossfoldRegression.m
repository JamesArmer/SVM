%Regression
%Tune BoxConstraint + Epsilon + (Polynomial/KernelScale)

function bestValues = InnerCrossfoldRegression(dataSize, feature, innerFolds)

    %Load Data
    [features, labels] = loadDataRegression(dataSize, feature);
    
    %Declare Kernels
    kernels = ["linear";"polynomial";"RBF"];
    
    %Declare Dependent Hyperparameters
    hyperparam = ["PolynomialOrder";"KernelScale"];
    
    epsilonValue = 0.05:0.05:0.4;
  
    %HP Dependent Value, Epsilon Value, Box Constraint Value
    bestValues = [-1,-1,-1];
    bestRMSE = inf;
    
    RMSEarray = zeros(innerFolds,1);
    
    %Values of hyperparameters
    for kernel = 3:3
        if kernel == 1
            hyperparamValue = [];
            boxconstraint = [183 186 189 192];
        elseif kernel == 2
            hyperparamValue = [2,3,4];
            boxconstraint = [];
        else
            hyperparamValue = [0.25:0.25:1 1.5:0.5:3 4 5];
            boxconstraint = [183 186 189 192];
        end

        for hyperparamIndex = 1:size(hyperparamValue,2)

            for epsilonIndex = 1:size(epsilonValue,2)
                
                for boxIndex = 1:size(boxconstraint,2)
                    
                    %Inner fold
                    for innerFold = 0:innerFolds-1

                        
                        innerFoldSize = dataSize/innerFolds;

                        %Get index of start and end of inner fold
                        innerFoldIndexStart = (innerFold*innerFoldSize +1);
                        innerFoldIndexEnd = ((innerFold+1)*innerFoldSize);

                        %Create Subset and Validation Data
                        if innerFold == 0
                            train_X = features(innerFoldIndexEnd+1:dataSize,:);
                            train_Y = labels(innerFoldIndexEnd+1:dataSize);
                            test_X = features(1:innerFoldIndexEnd,:);
                            test_Y = labels(1:innerFoldIndexEnd);
                            %disp("train = " + (innerFoldIndexEnd+1) + " to " + dataSize);
                            %disp("test = " + 1 + " to " + innerFoldIndexEnd);
                        elseif innerFold == innerFolds - 1
                            train_X = features(1:innerFoldIndexStart-1,:);
                            train_Y = labels(1:innerFoldIndexStart-1);
                            test_X = features(innerFoldIndexStart:dataSize,:);
                            test_Y = labels(innerFoldIndexStart:dataSize);
                            %disp("train = " + 1 + " to " + (innerFoldIndexStart-1));
                            %disp("test = " + innerFoldIndexStart + " to " + dataSize);
                        else
                            train_X = [features(1:innerFoldIndexStart-1,:);features(innerFoldIndexEnd+1:dataSize,:)];
                            train_Y = [train_Y(1:innerFoldIndexStart-1);labels(innerFoldIndexEnd+1:dataSize)];
                            test_X = features(innerFoldIndexStart:innerFoldIndexEnd,:);
                            test_Y = labels(innerFoldIndexStart:innerFoldIndexEnd);
                            %disp("train = " + 1 + " to " + (innerFoldIndexStart-1) + " | " + (innerFoldIndexEnd+1) + " to " + dataSize);
                            %disp("test = " + innerFoldIndexStart + " to " + innerFoldIndexEnd);
                        end

                        if(kernel == 1)
                            Mdl = fitrsvm(train_X, tain_Y, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', boxconstraint(boxIndex), 'Epsilon', epsilonValue(epsilonIndex));
                        elseif(kernel == 2)
                            Mdl = fitrsvm(train_X, train_Y, 'Standardize', true, 'KernelFunction', 'polynomial', hyperparam(kernel-1), hyperparamValue(hyperparamIndex), 'BoxConstraint', 1, 'Epsilon', epsilonValue(epsilonIndex));
                        else
                            Mdl = fitrsvm(train_X, train_Y, 'Standardize', true, 'KernelFunction', 'RBF', hyperparam(kernel-1), hyperparamValue(hyperparamIndex), 'BoxConstraint', boxconstraint(boxIndex), 'Epsilon', epsilonValue(epsilonIndex));
                        end
                        RMSE = testModelRegression(Mdl, test_X, test_Y);
                        RMSEarray(innerFold+1) = RMSE;
                    end

                    RMSE = sum(RMSEarray) / innerFolds;

                    if RMSE < bestRMSE
                        bestValues = [boxconstraint(boxIndex),hyperparamValue(hyperparamIndex),epsilonValue(epsilonIndex)];
                        bestRMSE = RMSE;
                        disp("  New best!");
                    end
                    disp("  Average RMSE value: " + RMSE);
                end
            end
        end
    end
    
    disp("Best Result = RMSE " + bestRMSE + ": " + bestValues(1) + " = " + bestValues(2) + ", Epsilon = " + bestValues(3));
end    

