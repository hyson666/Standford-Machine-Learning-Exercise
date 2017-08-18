function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%

% 经过公式计算选取合适的随机范围值进而非对称化
epsilon_init = 0.12;
% 上一层为m个单元，下一层为n个单元，加上bias生成一个n*(m+1)的矩阵对应随机代价矩阵，复习前面可以确认这点
% rand生成[0,1]范围内的随机数，把其置换到[-epsilon_init,+epsilon_init]的区间内
W = rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init;



% =========================================================================

end
