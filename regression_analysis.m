function regression_analysis()
    % Take [size] samples from class [class]
    % Perform 10-fold cross-validation on every model with these samples
    % Return the RMSE (root mean squared error) of every fold
    
    size = 30000;
    class = 1;
    
    % Run ANN Regression
    % Run manually
    % Batch size 32
    % Hidden layers 4
    % Epochs 1000 with early stopping
    ann_a = [13.87583, 15.644047, 16.410091, 17.15539, 15.30368, 15.086084, 15.353704, 18.778374, 13.9448, 18.185783];
    
    % Run SVM Linear Regression
    [svml_a, svml_f] = outerKfold(1,2,size,class);
    %svml_a = [0,0,0,0,0,0,0,0,0,0];
    %svml_f = [0,0,0,0,0,0,0,0,0,0];
    % Run SVM Gaussian Regression
    [svmg_a, svmg_f] = outerKfold(2,2,size,class);
    %svmg_a = [0,0,0,0,0,0,0,0,0,0];
    %svmg_f = [0,0,0,0,0,0,0,0,0,0];
    % Run SVM Polynomial Regression
    [svmp_a, svmp_f] = outerKfold(3,2,size,class);
    %svmp_a = [0,0,0,0,0,0,0,0,0,0];
    %svmp_f = [0,0,0,0,0,0,0,0,0,0];
    
    % Compare losses
    disp("ANN has a mean RMSE of " + mean(ann_a) + " and  a standard deviation of " + std(ann_a) + ".")
    disp("SVM Linear has a mean RMSE of " + mean(svml_k) + " and  a standard deviation of " + std(svml_k) + ".")
    disp("SVM Gaussian has a mean RMSE of " + mean(svmg_k) + " and  a standard deviation of " + std(svmg_k) + ".")
    disp("SVM Polynomial has a mean RMSE of " + mean(svmp_k) + " and  a standard deviation of " + std(svmp_k) + ".")
    accuracy = [ann_a;svml_a;svmg_a;svmp_a];
    names = ["ANN","SVM Linear","SVM Gaussian","SVM Polynomial"];
    % Box plot takes 4x10 matrix of accuracies
    boxplot(accuracy', names)
    ylabel("Root Mean Squared Error")
    xlabel("Algorithm")
    title("Comparisson of Regression Algorithms")
    disp(" ");
    
    %Perform t-tests
    disp("Performing t-tests. Null hypothesis is that both models perform the same");
    set = [ann_a; svml_a; svmg_a; svmp_a];
    printname = ["Artifical Neural Network", "Linear SVM", "Gaussian SVM", "Polynomial SVM"];
    for i = 1:3
        for j = (i+1):4
            %Maybe change some variables
            [h, p] = ttest2(set(i,:), set(j,:));
            disp("2-sample t-test between " + printname(i) + " and " + printname(j) + ".");
            disp("H = " + h + ". P = " + p + ".");
            disp(" ");
        end
    end
end