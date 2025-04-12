addpath("../metrics/")
load("../../Data/AC.mat")
data;
class;
num_clusters = length(unique(class));
sigma = 0.5;
n = size(data, 1);
W = zeros(n, n);
epsilon = 1e-5;  % 加入小的正数以避免奇异矩阵
for i = 1:n
    for j = 1:n
        W(i, j) = exp(-norm(data(i, :) - data(j, :))^2 / (2 * sigma^2)) + epsilon;
        % W(i, j) = exp(-norm(data(i, :) - data(j, :))^2 / (2 * sigma^2));
    end
end
D = diag(sum(W, 2));
L = D - W;

D_inv_sqrt = D^(-0.5);
L_sym = D_inv_sqrt * L * D_inv_sqrt;

[eig_vec, ~] = eigs(L_sym, num_clusters, 'smallestabs');

Y = bsxfun(@rdivide, eig_vec, sqrt(sum(eig_vec.^2, 2)));

pred_labels = kmeans(Y, num_clusters);

nmi_value = nmi(Y, pred_labels + 1);

fprintf('NMI 值为: %.4f\n', nmi_value);