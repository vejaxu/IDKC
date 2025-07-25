% load('data/s3.mat') % load dateset
% 
% data = (data - min(data)).*((max(data) - min(data)).^-1);
% data(isnan(data)) = 0.5; % data normalisation
% k=size(unique(class),1);
% 
% v=0.9;
% s=min(size(data,1),10000);
% t=100;
% 
% psi=64;
% Kn=50;
% rng(1)
% 
% [ndata] = iNNEspace(data,data, psi, t); 
% [Tclass,Centre] =DKC(ndata,k,Kn,v,s);
% [NMI]=nmi(class,Tclass+1)

% visualResults(data,Tclass,Centre,NMI)

addpath("IDKC")
addpath("utils")
addpath("metrics")
load(['../Data/4C.mat'])

data = double(data);    
class = double(class);

data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; % Data normalization
k = size(unique(class), 1);

v = 0.9;
s = min(size(data, 1), 10000);
t = 100;
rng(1)

n = size(data, 1);
Kn_values = round(n * (0.05: 0.05: 0.5));
% psi_values = [2, 4, 6, 8, 16, 24, 32, 48, 64, 80, 100, 200, 250, 500, 750, 1000, 2000, 2500];
% psi_values = [2, 4, 6, 8, 16, 24, 32, 48, 64, 80, 100, 200, 250, 500, 750, 1000, 2000]; % YaleB
% psi_values = [2, 4, 6, 8, 16, 24, 32, 48, 64, 80, 100, 200, 250, 500, 750]; % DSSS
psi_values = [2, 4, 6, 8, 16, 24, 32, 48, 64, 80, 100, 200, 250, 500, 750]; % AC

best_NMI = -inf;
best_ARI = -inf;
best_params_nmi = [NaN, NaN];
best_params_ari = [NaN, NaN];

for psi = psi_values
    for Kn = Kn_values
        NMI_values = zeros(1, 10);
        ARI_values = zeros(1, 10);
        
        for iter = 1:10
            [ndata] = iNNEspace(data, data, psi, t);
            [Tclass, Centre] = DKC(ndata, k, Kn, v, s);
            NMI_values(iter) = nmi(class + 1, Tclass + 1);
            ARI_values(iter) = ari(class + 1, Tclass + 1);
        end
        
        mean_NMI = mean(NMI_values);
        mean_ARI = mean(ARI_values);

        if mean_NMI > best_NMI
            best_NMI = mean_NMI;
            best_params_nmi = [psi, Kn];
        end
        if mean_ARI > best_ARI
            best_ARI = mean_ARI;
            best_params_ari = [psi, Kn];
        end

        fprintf('psi = %d, Kn = %d, NMI = %.4f, ARI = %.4f\n', psi, Kn, mean_NMI, mean_ARI);
    end
end

fprintf('Best (psi, Kn) for NMI: (%d, %d) with NMI = %.4f\n', best_params_nmi(1), best_params_nmi(2), best_NMI);
fprintf('Best (psi, Kn) for ARI: (%d, %d) with ARI = %.4f\n', best_params_ari(1), best_params_ari(2), best_ARI);
