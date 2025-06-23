clear
clc
addpath E:\XWJ_code\personal\IDKC\metrics
datasets = ["E:\XWJ_code\personal\Data\AC.mat"];

Res = [];

for datai = 1: length(datasets)
    datanow = char(datasets(datai));
    load(datanow);

    if size(class,1) < size(class,2)
        class = class';
    end

    dataA = data;
    data = (data - min(data)).*((max(data) - min(data)).^-1);
    data(isnan(data)) = 0.5; % data normalisation
    class = class - min(class) + 1;


siglist = [2.^[-5: 1: 5] 2.^[-5: 1: 5].*size(data, 2)];
% siglist=[0.01];
siglist = unique(siglist);

k = size(unique(class), 1);
t = min(400, size(data, 1));
rounds = 1;


class = double(class);


AA = zeros(rounds,2);

for i = 1: 1: rounds

dist_temp = pdist(data);
dist = squareform(dist_temp);

for pp = 1: length(siglist)    
    sig = siglist(pp);
    S = exp(-0.5 * (dist.^2)./(2 * sig ^ 2));
    Tclass = spectralcluster(S, k, 'Distance', 'precomputed', 'LaplacianNormalization', 'symmetric');
    [NMI] = nmi(class,Tclass);
    % [f1]=fmeasure(class,Tclass);
    AA(i,pp) = NMI;
    % FA(i,pp) = f1;
end
end
[maxNMI, pp] = max(mean(AA, 1));
% [maxFme, ~] = max(mean(FA, 1));
Res = [Res;maxNMI];
end

sig = siglist(pp);
S = exp(-0.5 * (dist.^2)./(2 * sig ^ 2));

Tclass = spectralcluster(S, k, 'Distance', 'precomputed', 'LaplacianNormalization', 'symmetric');
% Tclass = spectralcluster(S,k,'Distance','precomputed','LaplacianNormalization','none');
% [Tclass] = BestMapping(class, Tclass);
[NMI] = nmi(class, Tclass);

color = ['r','b','b','c','m','y'];


gscatter(data(:, 1), data(:, 2), Tclass, color, [], 15);
axis off
axis([0 1 0 1])
set(gcf, 'InvertHardCopy', 'off');
