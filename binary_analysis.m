function binary_analysis()
    % Take [size] samples from class [class]
    % Perform k-fold cross-validation on every model with these samples
    % Return the accuracy and f1 score of every fold
    
    size = 30000;
    class = 1;
    
    % Run DT Binary
    % Tree Depth 5
    dt_a = [0.7757, 0.6327, 0.6580, 0.7403, 0.6480, 0.5963, 0.4630, 0.6733, 0.4637, 0.7647];
    dt_f = [0.7378, 0.4506, 0.5976, 0.8003, 0.6825, 0.5595, 0.4596, 0.6673, 0.5372, 0.7974];
    dt_a = dt_a * 100;
    %dt_f = dt f1 scores
    
    % Run SVM Linear Binary
    svml_a = [0,0,0,0,0,0,0,0,0,0];
    svml_f = [0,0,0,0,0,0,0,0,0,0];
    %[svml_a, svml_f] = outerKfold(1,1,size,class);
    svml_a = svml_a * 100;
    % Run SVM Gaussian Binary
    svmg_a = [0,0,0,0,0,0,0,0,0,0];
    svmg_f = [0,0,0,0,0,0,0,0,0,0];
    %[svmg_a, svmg_f] = outerKfold(2,1,size,class);
    svmg_a = svmg_a * 100;
    % Run SVM Polynomial Binary
    svmp_a = [0,0,0,0,0,0,0,0,0,0];
    svmp_f = [0,0,0,0,0,0,0,0,0,0];
    %[svmp_a, svmp_f] = outerKfold(3,1,size,class);
    svmp_a = svmp_a * 100;
    
    % Compare accuracies
    disp("DT has a mean accuracy of " + mean(dt_a) + " and  a standard deviation of " + std(dt_a) + ".");
    disp("SVM Linear has a mean accuracy of " + mean(svml_a) + " and  a standard deviation of " + std(svml_a) + ".");
    disp("SVM Gaussian has a mean accuracy of " + mean(svmg_a) + " and  a standard deviation of " + std(svmg_a) + ".");
    disp("SVM Polynomial has a mean accuracy of " + mean(svmp_a) + " and  a standard deviation of " + std(svmp_a) + ".");
    accuracy = [dt_a; svml_a; svmg_a; svmp_a];
    names = ["DT", "SVM Linear", "SVM Gaussian", "SVM Polynomial"];
    % Box plot takes 4x10 matrix of accuracies
    boxplot(accuracy', names)
    ylabel("Classification Accuracy")
    xlabel("Algorithm")
    title("Comparisson of Binary Classification Algorithms")
    disp(" ");
    
    % Compare f1 scores
    disp("DT has an average f1 score of " + mean(dt_f) + ".");
    disp("SVM Linear has an average f1 score of " + mean(svml_f) + ".");
    disp("SVM Gaussian has an average f1 score of " + mean(svmg_f) + ".");
    disp("SVM Polynomial has an average f1 score of " + mean(svmp_f) + ".");
    disp(" ");
    
    %Perform t-tests
    disp("Performing t-tests. Null hypothesis is that both models perform the same");
    set = [dt_a; svml_a; svmg_a; svmp_a];
    printname = ["Decision Tree", "Linear SVM", "Gaussian SVM", "Polynomial SVM"];
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