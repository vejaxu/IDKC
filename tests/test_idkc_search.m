function test_idkc_search(output_root)
% Verify alignment, macro averaging, seed/q selection and CSV exports.
project_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_dir, 'IDKC'), fullfile(project_dir, 'experiments'), ...
    fullfile(project_dir, 'metrics'));
if ~isfolder(output_root)
    mkdir(output_root);
end

[scores, labels] = clustering_scores([10;10;20;20], [2;2;1;1]);
assert(isequal(labels, [10;10;20;20]));
assert(abs(scores.NMI - 1) < 1e-12 && abs(scores.ARI - 1) < 1e-12);
assert(scores.MacroF1 == 1);
scores = clustering_scores([ones(9,1);2], ones(10,1));
assert(abs(scores.MacroF1 - 9/19) < 1e-12);
[scores, labels] = clustering_scores([1;1;2;2], [1;1;0;0]);
assert(scores.MacroF1 == 0.5 && scores.Unassigned == 2);
assert(all(isnan(labels(3:4))));
[scores, labels] = clustering_scores([0;0;10;10], [2;2;1;1]);
assert(scores.MacroF1 == 1 && isequal(labels, [0;0;10;10]));
% More predicted clusters than classes: do not silently merge extras.
truth = [1;1;1;2;2;2];
raw = [1;1;3;2;2;4];
[scores, labels] = clustering_scores(truth, raw);
assert(sum(isnan(labels)) == 2 && abs(scores.MacroF1 - 0.8) < 1e-12);
assert(abs(scores.NMI - nmi(truth,raw)) < 1e-12);
assert(abs(scores.ARI - ari(truth,raw)) < 1e-12);

rng(123, 'twister');
data = [randn(20,2)*0.1; randn(20,2)*0.1+2; randn(20,2)*0.1+4];
data(:,3) = 1; % Constant attribute exercises normalization.
class = [repmat(10,20,1); repmat(20,20,1); repmat(30,20,1)]'; %#ok<NASGU>
data_file = fullfile(output_root, 'synthetic.mat');
save(data_file, 'data', 'class');
options = struct('psi_values', [2,4], 'seeds', [42,3407,4079,2024,0], ...
    'q_fractions', [0.1,0.2], 't', 20, 'v', 0.9, 'sample_size', 10000);
result = run_idkc_search(data_file, fullfile(output_root, 'first'), options);
assert(height(result.QSearch) == 20 && height(result.SeedBest) == 10);
assert(height(result.PsiSummary) == 2);
for i = 1:height(result.SeedBest)
    row = result.SeedBest(i,:);
    candidates = result.QSearch(result.QSearch.Psi == row.Psi & ...
        result.QSearch.Seed == row.Seed,:);
    [~, index] = max(candidates.NMI);
    assert(isequal(table2array(row), table2array(candidates(index,:))));
end
for i = 1:height(result.PsiSummary)
    selected = result.SeedBest(result.SeedBest.Psi == result.PsiSummary.Psi(i),:);
    assert(isequal(table2array(result.PsiSummary(i,2:5)), ...
        mean(table2array(selected(:,4:7)),1)));
end
[~, index] = max(result.PsiSummary.MeanNMI);
assert(result.BestPsi.Psi == result.PsiSummary.Psi(index));
assert(result.BestRun.NMI == max(result.BestPsiSeeds.NMI));
assert(all(result.QSearch.LoadToLabelSeconds > 0));
saved_labels = readtable(fullfile(result.OutputDirectory, 'best_labels.csv'));
assert(isequaln(table2array(saved_labels), table2array(result.BestLabels)));
[scores, aligned] = clustering_scores(saved_labels.TrueLabel, saved_labels.RawLabel);
assert(isequaln(aligned, saved_labels.AlignedLabel));
assert(isequal([scores.NMI,scores.ARI,scores.MacroF1], ...
    table2array(result.BestRun(:,4:6))));
csv_lines = splitlines(string(fileread(fullfile(result.OutputDirectory, 'q_search.csv'))));
for i = 2:numel(csv_lines)
    if strlength(csv_lines(i)) == 0
        continue;
    end
    fields = split(csv_lines(i), ',');
    for j = 4:7
        assert(~isempty(regexp(char(fields(j)), '^[-]?\d+\.\d{4}$', 'once')));
    end
end
assert(contains(fileread(fullfile(result.OutputDirectory, 'best_psi.csv')), ...
    sprintf('%.4f', result.BestPsi.MeanNMI)));
repeat = run_idkc_search(data_file, fullfile(output_root, 'repeat'), options);
assert(isequal(table2array(result.QSearch(:,[1:6,8])), ...
    table2array(repeat.QSearch(:,[1:6,8]))));
assert(isequaln(result.BestLabels, repeat.BestLabels));
statistics = jsondecode(fileread(repeat.StatisticsFile));
assert(statistics.Psi == repeat.BestPsi.Psi);
assert(statistics.BestSeed == repeat.BestRun.Seed && statistics.BestQ == repeat.BestRun.Q);
assert(max(abs(statistics.Metrics - table2array(repeat.BestPsiSeeds(:,4:7))), [], 'all') < 1e-12);
test_idkc_summary(output_root, repeat);
fprintf('All IDKC search checks passed.\n');
end

function test_idkc_summary(output_root, result)
root = tempname(output_root);
mkdir(root);
info = struct('dataset', 'AC', 'points', 60, 'dimensions', 3, 'clusters', 3);
filename = update_idkc_summary(info, fullfile(root, 'AC'), result);
first = read_summary(filename);
assert(height(first) == 1 && first.dataset == "AC");
assert(first.psi == result.BestPsi.Psi && first.best_seed == result.BestRun.Seed);
assert(first.best_q == result.BestRun.Q && first.dimensions == 3);
expected_std = round(std(table2array(result.BestPsiSeeds(:,4:7)), 0, 1), 4);
assert(max(abs(table2array(first(:,{'nmi_std','ari_std','f1_std','time_std'})) - expected_std)) < 1e-12);
info.dataset = '123';
update_idkc_summary(info, fullfile(root, '123'), result);
second = read_summary(filename);
assert(height(second) == 2 && second.dataset(2) == "123");
assert(isequaln(first, second(1,:)));
info.dataset = 'ac';
info.dimensions = 4;
update_idkc_summary(info, fullfile(root, 'ac'), result);
third = read_summary(filename);
assert(height(third) == 2 && third.dataset(1) == "ac" && third.dimensions(1) == 4);
assert(isequaln(second(2,:), third(2,:)));
info.dataset = 'comma,name';
update_idkc_summary(info, fullfile(root, 'comma,name'), result);
fourth = read_summary(filename);
assert(height(fourth) == 3 && fourth.dataset(3) == "comma,name");
end

function result = read_summary(filename)
options = detectImportOptions(filename, 'TextType', 'string', 'VariableNamingRule', 'preserve');
options = setvartype(options, 'dataset', 'string');
result = readtable(filename, options);
end
