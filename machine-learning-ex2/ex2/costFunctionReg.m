function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h = (1./(1+e.^(-theta'*X')))';
% 注意这里也是严格从1开始！
theta(1)=0;
J = (-y'*log(h)-(1-y)'*log(1-h))/m + lambda*sum(theta.^2)/(2*m);
% (h-y)'*X属于列乘行，得到导数值的横向向量组，所以记得化成列。
% 注意修正值一定是从1开始的,1对应的index是2，所以令tmp（1）=0。
tmp = lambda*theta/m;
grad = ((h-y)'*X)'/m + tmp;

% =============================================================

end
