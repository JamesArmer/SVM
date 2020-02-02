function model = buildModel(m, t, params, trainF, trainL)
    %Inputs:
    %m = model to build
    %params = params to build model with
    %if m = lc, params = [box constraint]
    %if m = lr, params = [box constraint, epsilon]
    %if m = gc, params = [box constraint, kernel scale]
    %if m = gr, params = [box constraint, kernel scale, epsilon]
    %if m = pc, params = [box constraint, polynomial]
    %if m = pr, params = [box constraint, polynomial, epsilon]
    
    if(m == 1 && t == 1)
        %Build linear classifier
        disp("Builing linear classifier");
        model = fitcsvm(trainF, trainL, 'Standardize', true, 'KernelFunction','linear', 'BoxConstraint', params(1));
    elseif(m == 1 && t == 2)
        %Build linear regression
        disp("Building linear regression");
        model = fitrsvm(trainF, trainL, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', params(1), 'Epsilon', params(2));
    elseif(m == 2 && t == 1)
        %Build gaussian classifier
        disp("Building gaussian classifier");
        model = fitcsvm(trainF, trainL, 'Standardize', true, 'KernelFunction','RBF', 'BoxConstraint', params(1), 'KernelScale', params(2));
    elseif(m == 2 && t == 2)
        %Build gaussian regression
        disp("Building gaussian regression");
        model = fitrsvm(trainF, trainL, 'Standardize', true, 'KernelFunction','RBF', 'BoxConstraint', params(1), 'KernelScale', params(2), 'Epsilon', params(3));
    elseif(m == 3 && t == 1)
        %Build polynomial classifier
        disp("Building polynomial classifier");
        model = fitcsvm(trainF, trainL, 'Standardize', true, 'KernelFunction', 'polynomial', 'BoxConstraint', params(1), 'PolynomialOrder', params(2));
    elseif(m == 3 && t == 2)
        %Build polynomial regression
        disp("Building polynomial regression");
        model = fitrsvm(trainF, trainL, 'Standardize', true, 'KernelFunction', 'polynomial', 'BoxConstraint', params(1), 'PolynomialOrder', params(2), 'Epsilon', params(3));
    else
        disp("Model invalid");
    end
end