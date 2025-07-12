% 读取原始数据
load('../../Data/AC.mat');  % 假设包含变量 data 和 labels
data = data;
labels = class;

% 确保 labels 是列向量
if isrow(labels)
    labels = labels';
end

% 初始化新数据容器
all_data = [];
all_labels = [];

% 获取类别标签
classes = unique(labels);
[D, n_features] = size(data);

for i = 1:length(classes)
    cls = classes(i);
    
    % 提取当前类的数据
    class_data = data(labels == cls, :);
    n = size(class_data, 1);
    target_n = 10 * n;
    
    % 估计高斯分布参数
    mu = mean(class_data, 1);
    sigma = cov(class_data) + 1e-6 * eye(n_features);  % 加正则项防止奇异
    
    % 多元正态分布采样
    generated_data = mvnrnd(mu, sigma, target_n);
    generated_labels = repmat(cls, target_n, 1);
    
    % 累加
    all_data = [all_data; generated_data];
    all_labels = [all_labels; generated_labels];
    
    fprintf('类 %d: 原始 %d -> 扩展为 %d 样本。\n', cls, n, target_n);
end

% 保存为新的数据集
save('../../Data/AC_10.mat', 'all_data', 'all_labels');
fprintf('已保存为 AC10.mat，总样本数：%d\n', size(all_data, 1));
