function [accuracy,confusionMatrix] = testModel(model,features,labels)
%Tests the given model on given feautres, 
% returning accuracy and a confusion matrix

pred = [];
%disp(size(features,1));

for i = 1:size(features,1)
    pred = [pred;predict(model,features(i,:))];
end

[accuracy,confusionMatrix] = confusion(labels,pred);
end

