function [features, labels] = loadInput(dataSize, class)
%loadInput : Loads size rows of features and size labels of column class
%   use [a, b] = loadInput(x, y)

x = table2array(readtable("predx_for_classification.csv"));
y = table2array(readtable("predy_for_classification.csv"));
l = table2array(readtable("label.csv"));

a = [x y];
features = a(1:dataSize,:);
labels = l(1:dataSize,class);
end