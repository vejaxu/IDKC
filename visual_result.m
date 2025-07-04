addpath("IDKC")
addpath("utils")
addpath("metrics")
load(['../Data/one_gaussian_10_one_line_5_2.mat'])

data = double(data);    
class = double(class);

data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; % Data normalization
k = size(unique(class), 1);

v = 0.9;
s = min(size(data, 1), 10000);
t = 100;

Kn = 490;
% rng('default')
rng(1)

best_psi_nmi = 6;

% 使用 best_psi_nmi 再次计算聚类结果
[ndata_best] = iNNEspace(data, data, best_psi_nmi, t);
[Tclass_best, ~] = DKC(ndata_best, k, Kn, v, s);

if size(data, 2) == 2
    % 如果是二维数据，直接绘图
    figure;
    gscatter(data(:,1), data(:,2), Tclass_best);
    title(sprintf('Clustering result with best psi (psi = %d)', best_psi_nmi));
    xlabel('X'); ylabel('Y');

    % 可选：绘制真实类别标签
    figure;
    gscatter(data(:,1), data(:,2), class);
    title('Ground truth classes');
    xlabel('X'); ylabel('Y');

else
    % 如果不是二维数据，使用 t-SNE 可视化
    Y = tsne(data, 'NumDimensions', 2, 'Perplexity', 30, 'Standardize', true);

    figure;
    gscatter(Y(:,1), Y(:,2), Tclass_best);
    title(sprintf('Clustering result with best psi (psi = %d)', best_psi_nmi));
    xlabel('t-SNE 1'); ylabel('t-SNE 2');

    % 可选：绘制真实类别标签
    figure;
    gscatter(Y(:,1), Y(:,2), class);
    title('Ground truth classes (t-SNE visualization)');
    xlabel('t-SNE 1'); ylabel('t-SNE 2');
end