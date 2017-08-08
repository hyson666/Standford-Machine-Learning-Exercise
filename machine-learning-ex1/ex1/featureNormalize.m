function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% It's always a good way to mark the size as m,n.
n = size(X,2);
m = size(X,1);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

%记得正态分布，N(miu,sigma)通过（N-μ）/segma化为N(0,1)吗，就是这个公式。

mu = mean(X);   % 1 x n;
sigma = std(X); % 1 x n;

mu_temp = ones(m,1)*mu;  % (m x 1) * (1 x n) => (n x m) , 得到m行mean(X);
sigma_temp = ones(m,1)*sigma;

X_norm = (X-mu_temp)./sigma_temp;

% ============================================================

end
