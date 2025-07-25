clear;      % 清除工作区中的变量
clc;        % 清除命令行窗口内容
close all;  % 关闭所有打开的图窗
addpath E:\XWJ_code\personal\IDKC\metrics
datasets = ["E:\XWJ_code\personal\Data\4C.mat"];

Res = []; % 保存每个数据集的最优聚类指标结果

for datai = 1: length(datasets)
    datanow = char(datasets(datai));
    load(datanow);
    data = data;
    class = class;
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

n = size(data, 1)
class = double(class);


MA = zeros(rounds, 2);
AA = zeros(rounds, 2);


for i = 1: 1: rounds

    dist_temp = pdist(data);
    dist = squareform(dist_temp); % 构造相似度矩阵

    for pp = 1: length(siglist) % 对于每个参数而言
        sig = siglist(pp);
        S = exp(-0.5 * (dist.^2)./(2 * sig ^ 2));
        S = double(S);
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


sig = siglist(pp);
S = exp(-0.5 * (dist.^2) ./ (2 * sig^2));

Tclass = spectralcluster(S, k, 'Distance', 'precomputed', 'LaplacianNormalization', 'symmetric');
[NMI] = nmi(class, Tclass);

color = lines(k);
gscatter(dataA(:, 1), dataA(:, 2), Tclass, color, [], 15);
axis equal 