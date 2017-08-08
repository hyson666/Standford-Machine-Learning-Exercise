function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% 观察数据集知道数据分为两类，y为他们的分类，分别为0，1，而X数据集对应他们的坐标。
% 利用find函数来构建两个集合，一个为分类0集合，一个为分类1集合。
pos = find(y==1);
neg = find(y==0);

% plot examples，留意plot函数的参数。
% X为二维坐标集，X(pos,d)中，d=1代表第一维（X坐标），d=2代表第二位（Y坐标）。
% 指定pos、neg为已经区分的坐标集。
% plot用点集输入，plot（X坐标，Y坐标，options），这里options指定画图线宽、颜色、大小、形状等。
% MakerFaceColor指定颜色，LineWidth指定线宽。
plot(X(pos,1),X(pos,2),'LineWidth',2,'MarkerSize', 7 ,'k+');
plot(X(neg,1),X(neg,2),'MarkerSize',7,'MarkerFaceColor','y','ko');



% =========================================================================



hold off;

end
