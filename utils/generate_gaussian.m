% 生成数据
n = 5000;
w = 2000;
data0 = zeros(n,w);
data1 = randn(n,w);
data2 = randn(n,w);
data = [[data1,data0];[data0,data2]];
class = ones(2*n,1);
class(n+1:end) = 2;

% 生成文件名，包含 w 的值
filename = sprintf('Data/w%dGaussians.mat', w);

% 保存数据
save(filename, 'data', 'class');
