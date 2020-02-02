function [accuracy, confusionMatrix] = confusion(preds,labels)
%Returns the accuracy and confusion matrix of a model

    tp = 0;
    tn = 0;
    fp = 0;
    fn = 0;
    %disp(size(preds,1));

    for i = 1:size(preds,1)
        if (preds(i) == 1)
            if (labels(i) == 1)
                tp = tp + 1;
            else
                fp = fp + 1;
            end
        else
            if (labels(i) == 0)
                tn = tn + 1;
            else
                fn = fn + 1;
            end
        end
    end
    
    accuracy = (tp + tn) / (size(preds,1));
    confusionMatrix = [tn, fp; fn, tp];
    
    %plotconfusion(labels',preds');
end

