rng(42);
k = 50;

disp("Loading...");
tic;
mF = matfile("./featuremaps/NYC/NYC_GDK_1600_train.mat");
data = mF.featureMaps;
n = size(data, 1);
data = data(randperm(n, 100000), :);
data = unique(data, 'rows');
n = size(data, 1);
toc;

disp("Computing...");
tic; d = squareform(pdist(data)); toc;

disp("Sorting...");
tic; [dist, indices] = mink(d, k+1, 2); toc;

indices(:, 1) = [];
dist(:, 1) = [];

tic;
ids = zeros(n, 1);
for i = 1 : n
    if mod(i, 1000) == 0
        fprintf('i = %d / %d\n', i, n);
    end
    KNN = data(indices(i,:), :);
    ids(i) = idtle(KNN, dist(i,:));
end
toc;

% boxplot(ids);
% set(gcf, 'color', 'w');
% set(gca,'linewidth', 1, 'fontsize', 14, 'fontname', 'Times');
% ylabel('ID');
% title('Estimated intrinsic dimensions');
disp(quantile(ids, [0.03, 0.25, 0.5, 0.75, 0.97]));
% % IDK:
% % T-Drive:  2.1951,  4.6151,  7.6759,  9.6981, 13.3389
% %   Porto:  3.2883,  4.7809,  5.8347,  7.1153, 10.5933
% %     NYC: 12.2849, 16.1424, 18.0505, 20.0885, 24.2509
% % GDK: 
% % T-Drive: 0.8921, 1.5931, 2.2857, 2.8013,  4.3701
% %   Porto: 2.3681, 3.1962, 3.7799, 4.5201,  6.4909
% %     NYC: 0.0000, 0.0000, 0.0000, 0.0000, 26.4541
