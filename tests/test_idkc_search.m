function test_idkc_search(output_root)
% Verify alignment, macro averaging, seed/q selection and CSV exports.
project_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_dir, 'IDKC'), fullfile(project_dir, 'experiments'), ...
    fullfile(project_dir, 'utils'), ...
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
summary = readtable(repeat.SummaryFile, 'TextType', 'string');
assert(height(summary) == 1 && summary.Dataset == "synthetic");
test_idkc_summary(output_root, repeat, options);
fprintf('All IDKC search checks passed.\n');
end

function test_idkc_summary(output_root, result, options)
summary_root = tempname(output_root);
output_dir = fullfile(summary_root, 'AC');
filename = update_idkc_summary('AC.mat', output_dir, result, options);
first = read_summary_for_test(filename);
assert(height(first) == 1 && first.Dataset == "AC");
assert(first.BestPsi == result.BestPsi.Psi);
assert(first.BestRunSeed == result.BestRun.Seed);
assert(first.Seeds == "42;3407;4079;2024;0");
assert(first.SeedBestQs == strjoin(string(result.BestPsiSeeds.Q'), ';'));
assert(abs(first.MeanNMI - round(result.BestPsi.MeanNMI, 4)) < 1e-12);

% Append a numeric-looking dataset name without changing the previous row.
update_idkc_summary('123.mat', fullfile(summary_root, '123'), result, options);
second = read_summary_for_test(filename);
assert(height(second) == 2 && second.Dataset(2) == "123");
assert(isequaln(first, second(1,:)));

% Replace an existing dataset in place (case insensitive), preserving others.
replacement = result;
replacement.BestPsi.MeanNMI = 0.123456;
replacement.BestRun.Seed = 2024;
replacement.BestRun.Q = 12;
replacement.SearchSeconds = 12.345678;
update_idkc_summary('ac.mat', output_dir, replacement, options);
third = read_summary_for_test(filename);
assert(height(third) == 2 && third.Dataset(1) == "ac");
assert(third.MeanNMI(1) == 0.1235 && third.BestRunSeed(1) == 2024);
assert(third.BestRunQ(1) == 12 && third.SearchSeconds(1) == 12.3457);
assert(isequaln(second(2,:), third(2,:)));

% CSV quoting survives a dataset name containing a comma.
update_idkc_summary('comma,name.mat', fullfile(summary_root, 'comma,name'), ...
    result, options);
fourth = read_summary_for_test(filename);
assert(height(fourth) == 3 && fourth.Dataset(3) == "comma,name");
assert(contains(fileread(filename), ',0.1235,'));
assert(isempty(dir(fullfile(summary_root, 'tp*.csv'))));
end

function value = read_summary_for_test(filename)
options = detectImportOptions(filename, 'TextType', 'string');
options = setvartype(options, {'Dataset', 'Seeds', 'SeedBestQs'}, 'string');
value = readtable(filename, options);
end
