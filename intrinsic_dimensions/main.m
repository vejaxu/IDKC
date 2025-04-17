datasets = {
    '../../Data/AC.mat', 'data';
    '../../Data/spam.mat', 'data';
    '../../Data/a9a.mat', 'X';
    '../../Data/ImageNet-10.mat', 'data';
    '../../Data/Cifar-10.mat', 'data';
    '../../Data/aloi.mat', 'data';
    '../../Data/usps_resampled.mat', 'data';
    '../../Data/USPS.mat', 'data';
    '../../Data/mnist.mat', 'data';
    '../../Data/mnist100000.mat', 'data';
    '../../Data/COIL20.mat', 'X';
    '../../Data/RCV1.mat', 'X';
    '../../Data/gisette.mat', 'data';
    '../../Data/YaleB.mat', 'X';
};
% datasets = {
%     '../../Data/one_to_five/AC.mat', 'data';
%     '../../Data/one_to_five/spam.mat', 'data';
%     '../../Data/one_to_five/a9a.mat', 'data';
%     '../../Data/one_to_five/ImageNet-10.mat', 'data';
%     '../../Data/one_to_five/Cifar-10.mat', 'data';
%     '../../Data/one_to_five/aloi.mat', 'data';
%     '../../Data/one_to_five/usps_resampled.mat', 'data';
%     '../../Data/one_to_five/USPS.mat', 'data';
%     '../../Data/one_to_five/mnist.mat', 'data';
%     '../../Data/one_to_five/mnist100000.mat', 'data';
%     '../../Data/one_to_five/COIL20.mat', 'data';
%     '../../Data/one_to_five/RCV1.mat', 'data';
%     '../../Data/one_to_five/gisette.mat', 'data';
%     '../../Data/one_to_five/YaleB.mat', 'data';
% };


dataset_names = {'AC', 'spam', 'a9a', 'ImageNet-10', 'Cifar-10', ...
                 'aloi', 'usps_resampled' 'USPS', 'mnist', 'mnist100000', 'COIL20', ...
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


% 获取所有数据集的中位数并进行标注
hold on;
% 获取箱线图的中位数
medians = grpstats(all_ids, all_labels, {@median});

% 标注每个数据集的中位数
positions = 1:numel(dataset_names); % 位置对应每个数据集
for i = 1:numel(medians)
    text(positions(i), medians(i) + 0.05, sprintf('%.2f', medians(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'k');
end
hold off;