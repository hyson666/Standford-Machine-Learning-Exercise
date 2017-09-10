function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

error = 10;
temp = 0;
trylist = [0.01 0.03 0.1 0.3 1 3 10 30];

% 直接在for中指定要遍历的表
for C_temp = trylist
    for sigma_temp = trylist
        % svmPredict中的model指的是svmTrain训练返回的参数，要传送给定核函数的未知数与常数作为函数句柄
        model= svmTrain(X, y, C_temp, @(X, y) gaussianKernel(X, y, sigma_temp));
        predictions = svmPredict(model, Xval);
        temp = mean(double(predictions ~= yval));   %准度等于错误数除总数，刚好用mean实现
        if(temp < error)
            C = C_temp;
            sigma = sigma_temp;
            error = temp;
        end
    end
end


% =========================================================================

end
