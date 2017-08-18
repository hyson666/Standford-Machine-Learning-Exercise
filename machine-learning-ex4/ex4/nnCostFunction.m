function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 输入层a1,处理添加1
a1 = [ones(m,1),X];
% 根据a1计算隐藏层a2
z2 = a1*Theta1';
a2 = sigmoid(z2);
% 根据a2计算隐藏层a3，映射过来之后成为25个隐藏单元，还是继续添加1
a2 = [ones(size(a2,1),1),a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
% 得到a3的每一列都是输出层的值，根据此计算j
% 利用上次练习使用到的“==”技巧，对应位置相同的话变成1，不相同的话变成0
labels = 1:num_labels;
Y = (y==labels);
% importan:sum在只有一行或者一列的时候会自动计算列和行的总和,如果是矩阵，则计算每一列的和先。
h = a3;
% 注意这里用点乘法而不是向量化来计算累计误差
J = sum(sum(-Y.*log(h)-(1-Y).*log(1-h)))/m;
% 进行正则化
J += lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/(2*m);

% 进行神经网络求梯度,实际上可以利用向量化，但是第一次用普通for来实现，以便于理解整体过程
for t = 1:m
    % 每次循环把一组样例输入输入层，除了正则化的时候都要记得加上bias
    % 区别上面计算代价的部分
    % 首先进行FP算法推出输出
    a_1 = [1;X(t,:)'];
    z_2 = Theta1 * a_1;
    a_2 = [1;sigmoid(z_2)];
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
    % 这里只需要计算隐藏层（输出层的计算是无意义的）
    % 计算输出层误差（向量）,要把Y先提取出来
    y_new = zeros(1,num_labels);
    y_new(1,y(t)) = 1;
    delta_3 = a_3 - y_new';
    % 计算隐藏层误差
    delta_2 = Theta2' * delta_3 .* sigmoidGradient([1;Theta1*a_1]) ;
    % 逐步修正Delta的值，第二层隐藏层存在bias，输出层不存在
    Theta1_grad = Theta1_grad + delta_2(2:end) * a_1';
    % 这一层是没有bias的
    Theta2_grad = Theta2_grad + delta_3 * a_2';
endfor

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

    



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
