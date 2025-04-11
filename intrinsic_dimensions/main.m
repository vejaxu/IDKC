datasets = {
    '../../Data/AC.mat', 'data';
    '../../Data/spam.mat', 'data';
    '../../Data/a9a.mat', 'X';
    '../../Data/ImageNet-10.mat', 'data';
    '../../Data/Cifar-10.mat', 'data';
    '../../Data/aloi.mat', 'data';
    '../../Data/USPS.mat', 'data';
    '../../Data/mnist100000.mat', 'data';
    '../../Data/COIL20.mat', 'X';
    '../../Data/RCV1.mat', 'X';
    '../../Data/gisette.mat', 'data';
    '../../Data/YaleB.mat', 'X';
};

dataset_names = {'AC', 'spam', 'a9a', 'ImageNet-10', 'Cifar-10', ...
                 'aloi', 'USPS', 'mnist100000', 'COIL20', ...
                 'RCV1', 'gisette', 'YaleB'};

k = 50;
all_ids = [];
all_labels = [];

for i = 1:size(datasets, 1)
    matfile_path = datasets{i, 1};
    variable_name = datasets{i, 2};
    fprintf('\n====== Processing %s (%s) ======\n', matfile_path, variable_name);

    % 调用函数（优化过的版本，避免 pdist）
    ids = estimate_id(matfile_path, variable_name, k);

    % 收集 ID 和标签
    all_ids = [all_ids; ids];
    all_labels = [all_labels; repmat(dataset_names(i), numel(ids), 1)];

    % 释放中间变量以节省内存
    clear ids;
end

% 绘制合并箱线图
figure;
boxplot(all_ids, all_labels, 'LabelOrientation', 'inline');
set(gcf, 'color', 'w');
set(gca, 'linewidth', 1, 'fontsize', 14, 'fontname', 'Times');
ylabel('ID');
title('Estimated Intrinsic Dimensions Across Datasets');
