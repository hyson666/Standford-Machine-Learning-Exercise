function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% 注意向量化的本质是让两个数批量相乘。
% X中每一行都是特征量，向量化的时候转化为列计算，而theta是列向量，转化为行。
sm = theta' * X';
% 1*3矩阵和3*100矩阵相乘得到1*100矩阵h行向量,转化一下变成列向量为下面做准备。
h = (1./(1+e.^-sm))';

% 两种办法，一种点乘法用sum相加，第二种直接向量化相乘得到和。
J = (-y' * log(h) - (1-y)' * log(1-h))/m;
grad = ((h - y)' * X)'/m;


% =============================================================

end
