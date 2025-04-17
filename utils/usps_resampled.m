% 加载数据集
load('../../Data/usps_resampled.mat');  % 假设你的文件名为 'mydata.mat'

% 转置数据
train_patterns_transposed = train_patterns';  % 变为 4649 x 256
test_patterns_transposed = test_patterns';    % 变为 4649 x 256

% 拼接模式数据
data = [train_patterns_transposed; test_patterns_transposed];  % 结果为 9298 x 256

% 转置和拼接标签数据
train_labels_transposed = train_labels';  % 确保标签是列向量
test_labels_transposed = test_labels';    % 确保标签是列向量

% 将 train_labels 和 test_labels 转换为 9298 x 1 的标签向量
% 假设 train_labels 和 test_labels 是 10 x 4649 的标签矩阵
train_labels_vector = zeros(4649, 1);  % 初始化向量
test_labels_vector = zeros(4649, 1);   % 初始化向量

% 将标签转换为 1D 向量
for i = 1:4649
    train_labels_vector(i) = find(train_labels(:, i) == 1);  % 找到值为 1 的行数
end

for i = 1:4649
    test_labels_vector(i) = find(test_labels(:, i) == 1);  % 找到值为 1 的行数
end

% 拼接标签
labels = [train_labels_vector; test_labels_vector];  % 结果为 9298 x 1

% 保存为新的 .mat 文件
save('../../Data/usps_resampled.mat', 'data', 'labels');
