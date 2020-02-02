function RMSE = testModelRegression(model,features,labels)
    
    pred = zeros(size(features,1),1);
    squaredErrorTotal = 0;
    
    for i = 1:size(features,1)
        pred(i) = predict(model,features(i,:));
        squaredErrorTotal = squaredErrorTotal + ((labels(i) - pred(i))^2);
    end
    
    RMSE = sqrt(squaredErrorTotal / size(features,1));

end

