function [features, labels] = loadDataRegression(dataSize, class)
%loadInput : Loads size rows of features and size labels of column class
%   use [a, b] = loadInput(x, y)

x = table2array(readtable("predx_for_regression.csv"));
y = table2array(readtable("predy_for_regression.csv"));
l = table2array(readtable("angle.csv"));

a = [x y];
features = a(1:dataSize,:);
labels = l(1:dataSize,class);
end