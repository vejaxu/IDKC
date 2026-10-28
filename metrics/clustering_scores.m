function [scores, aligned_labels] = clustering_scores(true_labels, cluster_labels)
% Match clusters to classes by maximum overlap, then compute macro-F1.
% Label 0 is unassigned in DKC: retain it in NMI/ARI, but never match it.
true_labels = true_labels(:);
cluster_labels = cluster_labels(:);
assert(numel(true_labels) == numel(cluster_labels), 'Label lengths differ.');
assert(all(isfinite(true_labels)) && all(isfinite(cluster_labels)), ...
    'Labels must be finite.');
[class_values, ~, truth] = unique(true_labels);
[cluster_values, ~, prediction] = unique(cluster_labels);
k = numel(class_values);
overlap = accumarray([truth, prediction], 1, [k, numel(cluster_values)]);
assignable = find(cluster_values ~= 0);
cost = zeros(max(k, numel(assignable)));
cost(1:k, 1:numel(assignable)) = -overlap(:, assignable);
matching = hungarian(cost);

% Separate unmatched clusters preserve the original partition for NMI/ARI.
mapping = k + (1:numel(cluster_values))';
for j = 1:numel(assignable)
    class_id = find(matching(1:k, j), 1);
    if ~isempty(class_id)
        mapping(assignable(j)) = class_id;
    end
end
aligned_ids = mapping(prediction);
aligned_labels = NaN(size(truth));
matched = aligned_ids <= k;
aligned_labels(matched) = class_values(aligned_ids(matched));

% Average class F1 equally over all ground-truth classes, including misses.
class_f1 = zeros(k, 1);
for c = 1:k
    tp = sum(truth == c & aligned_ids == c);
    class_f1(c) = 2 * tp / (sum(truth == c) + sum(aligned_ids == c));
end
scores.NMI = nmi(truth, aligned_ids);
scores.ARI = ari(truth, aligned_ids);
scores.MacroF1 = mean(class_f1);
scores.Unassigned = sum(cluster_labels == 0);
end
