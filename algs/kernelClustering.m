clc; clear; close all;
addpath E:\XWJ_code\personal\IDKC\metrics

datafile = 'E:\XWJ_code\personal\Data\AC.mat';
load(datafile);
data  = data;
data_row = data;
label = class(:);
data  = (data - min(data)) ./ range(data);
k     = numel(unique(label));
%% 2. 距离矩阵
dist = squareform(pdist(data));
%% 3. KNN 网格搜索
K_list   = 5:2:15;              % 网格：5,7,9,11,13,15
rounds   = 10;
NMI_res  = zeros(rounds, numel(K_list));
ARI_res  = zeros(rounds, numel(K_list));

for r = 1:rounds
    for ki = 1:numel(K_list)
        K = K_list(ki);
        % 3.1 计算 K-th 近邻距离 R_K
        [~, R_K] = knnsearch(data, data, 'k', K+1);  % +1 去掉自身
        R_K = R_K(:, end);                           % n×1
        % 3.2 自适应带宽核矩阵
        Sigma = R_K * R_K.';                         % σp·σq
        S = exp(-0.5 * (dist.^2) ./ Sigma);
        % 3.3 谱聚类
        pred = spectralcluster(S, k, ...
                               'Distance', 'precomputed', ...
                               'LaplacianNormalization', 'symmetric');
        NMI_res(r, ki) = nmi(label, pred);
        ARI_res(r, ki) = ari(label, pred);
    end
end
%% 4. 选最佳 K（平均 NMI 最大）
[~, bestIdx] = max(mean(NMI_res,1));
bestK = K_list(bestIdx);
%% 5. 用最佳 K 重新聚类并可视化
[~, R_K] = knnsearch(data, data, 'k', bestK+1);
R_K = R_K(:, end);
Sigma = R_K * R_K.';
S_best = exp(-0.5 * (dist.^2) ./ Sigma);
T_best = spectralcluster(S_best, k, ...
                         'Distance', 'precomputed', ...
                         'LaplacianNormalization', 'symmetric');
NMI_best = nmi(label, T_best);
ARI_best = ari(label, T_best);
%% 6. 绘图
figure;
gscatter(data_row(:,1), data_row(:,2), T_best, lines(k), [], 15);
axis equal
title(sprintf('自适应核谱聚类  best K=%d  NMI=%.3f  ARI=%.3f', bestK, NMI_best, ARI_best));
%% 7. 打印网格结果
fprintf('K-grid: %s\n', mat2str(K_list));
fprintf('Avg NMI: %s\n', mat2str(mean(NMI_res,1)));
fprintf('Avg ARI: %s\n', mat2str(mean(ARI_res,1)));
fprintf('Best K = %d\n', bestK);