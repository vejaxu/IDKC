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
load(['../Data/one_to_five/isolet.mat'])

class = class';
class = str2double(class);

data = double(data);    
class = double(class);

data = (data - min(data)).*((max(data) - min(data)).^-1);
data(isnan(data)) = 0.5; % Data normalization
k = size(unique(class), 1);

v = 0.9;
s = min(size(data, 1), 10000);
t = 100;
 
psi_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
Kn = 50;
% rng('default')
rng(1)

best_NMI = -inf;
best_ARI = -inf;
best_psi_nmi = NaN;
best_psi_ari = NaN;

for psi = psi_values
    NMI_values = zeros(1, 10);
    ARI_values = zeros(1, 10);

    for iter = 1: 10
        [ndata] = iNNEspace(data, data, psi, t);
        [Tclass, Centre] = DKC(ndata, k, Kn, v, s);
        NMI_values(iter) = nmi(class + 1, Tclass + 1);
        [ARI_values(iter), ~, ~, ~] = ari(class + 1, Tclass + 1);
    end

    mean_NMI = mean(NMI_values);
    mean_ARI = mean(ARI_values);
     if mean_NMI > best_NMI
         best_NMI = mean_NMI;
         best_psi_nmi = psi;
     end

     if mean_ARI > best_ARI
         best_ARI = mean_ARI;
         best_psi_ari = psi;
     end
    fprintf('psi = %d, NMI = %.4f\n', psi, mean_NMI);
    fprintf('psi = %d, ARI = %.4f\n', psi, mean_ARI);
end

fprintf('Best psi = %d with NMI = %.4f\n', best_psi_nmi, best_NMI);
fprintf('Best psi = %d with ARI = %.4f\n', best_psi_ari, best_ARI);
