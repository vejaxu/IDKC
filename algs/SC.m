clear
clc
addpath E:\XWJ_code\personal\IDKC\metrics
datasets = ["E:\XWJ_code\personal\Data\one_gaussian_10_one_line_5_2.mat"];

Res = []; % 保存每个数据集的最优聚类指标结果

for datai = 1: length(datasets)
    datanow = char(datasets(datai));
    load(datanow);
    if size(class, 1) < size(class, 2)
        class = class';
    end
    dataA = data;
    data = (data - min(data)).*((max(data) - min(data)).^-1);
    data(isnan(data)) = 0.5; % data normalisation
    class = class - min(class) + 1;


siglist = [2.^[-5: 1: 5] 2.^[-5: 1: 5].*size(data, 2)]; % 构造超参列表
siglist = unique(siglist);

k = size(unique(class), 1) % 自动识别聚类数
t = min(400, size(data, 1));

rounds = 10; % 每个参数跑10次


class = double(class);


MA = zeros(rounds, 2);
AA = zeros(rounds, 2);


for i = 1: 1: rounds

    dist_temp = pdist(data);
    dist = squareform(dist_temp); % 构造相似度矩阵

    for pp = 1: length(siglist) % 对于每个参数而言
        sig = siglist(pp);
        S = exp(-0.5 * (dist.^2)./(2 * sig ^ 2));
        Tclass = spectralcluster(S, k, 'Distance', 'precomputed', 'LaplacianNormalization', 'symmetric');

        [NMI] = nmi(class, Tclass);
        [ARI] = ari(class, Tclass);
        % [f1]=fmeasure(class,Tclass);
        MA(i, pp) = NMI;
        AA(i, pp) = ARI;
        % FA(i,pp) = f1;
    end
end


[maxNMI, pp] = max(mean(MA, 1));
[maxARI, pp] = max(mean(AA, 1));
% [maxFme, ~] = max(mean(FA, 1));
Res = [Res; maxNMI; maxARI];
end

% 开始画图
sig = siglist(pp);
S = exp(-0.5 * (dist.^2)./(2 * sig ^ 2));

Tclass = spectralcluster(S, k, 'Distance', 'precomputed', 'LaplacianNormalization', 'symmetric');
% Tclass = spectralcluster(S,k,'Distance','precomputed','LaplacianNormalization','none');
% [Tclass] = BestMapping(class, Tclass);
[NMI] = nmi(class, Tclass);

% color = ['r','b','b','c','m','y'];
color = ['r','g','b','c','m','y','k',[0.5 0.5 0.5],'b','r'];

gscatter(data(:, 1), data(:, 2), Tclass, color, [], 15);
xlim([0 1]);     % 设置 x 轴范围
ylim([0 1]);     % 设置 y 轴范围
axis equal
axis off
% axis([0 1 0 1])
set(gcf, 'InvertHardCopy', 'off');
