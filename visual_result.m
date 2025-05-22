addpath("IDKC")
addpath("utils")
addpath("metrics")
load(['../Data/w3Gaussians.mat'])

data = double(data);    
class = double(class);

data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; % Data normalization
k = size(unique(class), 1);

v = 0.9;
s = min(size(data, 1), 10000);
t = 100;

Kn = 50;
% rng('default')
rng(1)

best_psi_nmi = 128;

% 使用 best_psi_nmi 再次计算聚类结果
[ndata_best] = iNNEspace(data, data, best_psi_nmi, t);
[Tclass_best, ~] = DKC(ndata_best, k, Kn, v, s);

% 使用 t-SNE 将数据降到2D
Y = tsne(data, 'NumDimensions', 2, 'Perplexity', 30, 'Standardize', true);

% 绘制聚类结果
figure;
gscatter(Y(:,1), Y(:,2), Tclass_best);
title(sprintf('Clustering result with best psi (psi = %d), NMI = %.4f', best_psi_nmi, best_NMI));
xlabel('t-SNE 1'); ylabel('t-SNE 2');

% 可选：绘制真实类别标签
figure;
gscatter(Y(:,1), Y(:,2), class);
title('Ground truth classes (t-SNE visualization)');
xlabel('t-SNE 1'); ylabel('t-SNE 2');

% 使用 best_psi_nmi 再次计算聚类结果
% [ndata_best] = iNNEspace(data, data, best_psi_nmi, t);
% [Tclass_best, ~] = DKC(ndata_best, k, Kn, v, s);
% 
% 可视化聚类结果
% figure;
% gscatter(data(:,1), data(:,2), Tclass_best);
% title(sprintf('Clustering result with best psi (psi = %d), NMI = %.4f', best_psi_nmi, best_NMI));
% xlabel('Feature 1'); ylabel('Feature 2');
% 
% 可选：可视化真实标签
% figure;
% gscatter(data(:,1), data(:,2), class);
% title('Ground truth classes');
% xlabel('Feature 1'); ylabel('Feature 2');
