function results = run_idkc_search(data_file, output_dir, options)
% Each candidate is independently timed from load() through DKC labels.
% Resetting RNG for every q gives each (psi, seed) the same kernel/sample.
assert(isrow(options.psi_values) && isrow(options.seeds) && isrow(options.q_fractions), ...
    'Search candidates must be row vectors.');
assert(~isempty(options.seeds) && numel(unique(options.seeds)) == numel(options.seeds) ...
    && all(options.seeds >= 0 & options.seeds == floor(options.seeds)), 'Invalid seeds.');
assert(all(isfinite(options.psi_values)) && all(options.psi_values == floor(options.psi_values)) ...
    && numel(unique(options.psi_values)) == numel(options.psi_values), 'Invalid psi values.');
assert(all(isfinite(options.q_fractions)) && all(options.q_fractions > 0 & options.q_fractions < 1), ...
    'Invalid q fractions.');
assert(options.t >= 1 && options.t == floor(options.t) && options.v > 0 && options.v < 1 ...
    && options.sample_size >= 2 && options.sample_size == floor(options.sample_size), ...
    'Invalid t, v or sample size.');
metadata = load(data_file, 'data', 'class');
assert(isfield(metadata, 'data') && isfield(metadata, 'class'), ...
    'Dataset must contain data and class.');
n = size(metadata.data, 1);
assert(n >= 2 && numel(metadata.class) == n, 'Invalid dataset dimensions.');
assert(all(isfinite(metadata.data(:))) && all(isfinite(metadata.class(:))), ...
    'Dataset must contain finite numeric values.');
s = min(n, options.sample_size);
k = numel(unique(metadata.class(:)));
assert(k <= s, 'The seed sample must contain at least k points.');
psi_values = options.psi_values(options.psi_values >= 2 & options.psi_values <= n);
q_values = unique(round(n * options.q_fractions));
q_values = q_values(q_values >= 1 & q_values < s);
assert(~isempty(psi_values) && ~isempty(q_values), 'No valid psi/q candidates.');
if numel(psi_values) < numel(options.psi_values)
    warning('IDKC:SkippedPsi', 'Skipping psi values outside [2, n].');
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end

names = {'Psi', 'Seed', 'Q', 'NMI', 'ARI', 'MacroF1', 'LoadToLabelSeconds', 'Unassigned'};
grid_rows = zeros(0, numel(names));
seed_rows = zeros(0, numel(names));
mean_rows = zeros(0, 5);
best_mean_nmi = -Inf;
best_labels = [];
best_psi_seed_rows = [];
search_start = tic;

for psi = psi_values
    psi_seed_rows = zeros(numel(options.seeds), numel(names));
    psi_labels = cell(numel(options.seeds), 1);
    for seed_index = 1:numel(options.seeds)
        seed = options.seeds(seed_index);
        best_nmi = -Inf;
        for q = q_values
            rng(seed, 'twister');
            run_start = tic;
            loaded = load(data_file, 'data', 'class');
            data = double(loaded.data);
            true_labels = double(loaded.class(:));
            data = (data - min(data, [], 1)) ./ ...
                (max(data, [], 1) - min(data, [], 1));
            data(isnan(data)) = 0.5;
            ndata = iNNEspace(data, data, psi, options.t);
            labels = DKC(ndata, k, q, options.v, s);
            load_to_label_seconds = toc(run_start);
            [scores, aligned_labels] = clustering_scores(true_labels, labels);
            row = [psi, seed, q, scores.NMI, scores.ARI, scores.MacroF1, ...
                load_to_label_seconds, scores.Unassigned];
            grid_rows(end + 1, :) = row; %#ok<AGROW>
            % Compare full precision; tied NMI retains the smallest q.
            if scores.NMI > best_nmi
                best_nmi = scores.NMI;
                psi_seed_rows(seed_index, :) = row;
                psi_labels{seed_index} = table((1:n)', true_labels, labels, ...
                    aligned_labels, 'VariableNames', ...
                    {'SampleIndex', 'TrueLabel', 'RawLabel', 'AlignedLabel'});
            end
        end
        row = psi_seed_rows(seed_index, :);
        fprintf('psi=%d seed=%d best_q=%d NMI=%.4f ARI=%.4f macro-F1=%.4f time=%.4fs\n', ...
            row(1:7));
    end
    mean_row = [psi, mean(psi_seed_rows(:, 4:7), 1)];
    mean_rows(end + 1, :) = mean_row; %#ok<AGROW>
    seed_rows = [seed_rows; psi_seed_rows]; %#ok<AGROW>
    fprintf('psi=%d mean NMI=%.4f ARI=%.4f macro-F1=%.4f time=%.4fs\n', mean_row);
    if mean_row(2) > best_mean_nmi
        best_mean_nmi = mean_row(2);
        best_mean_row = mean_row;
        best_psi_seed_rows = psi_seed_rows;
        [~, best_seed_index] = max(psi_seed_rows(:, 4));
        best_run_row = psi_seed_rows(best_seed_index, :);
        best_labels = psi_labels{best_seed_index};
        best_psi_labels = psi_labels;
    end
    % Checkpoints keep completed psi results if a later candidate fails.
    write_numeric_csv(fullfile(output_dir, 'q_search.csv'), array2table(grid_rows, 'VariableNames', names));
    write_numeric_csv(fullfile(output_dir, 'seed_best.csv'), array2table(seed_rows, 'VariableNames', names));
    write_numeric_csv(fullfile(output_dir, 'psi_summary.csv'), mean_table(mean_rows));
end

results.QSearch = array2table(grid_rows, 'VariableNames', names);
results.SeedBest = array2table(seed_rows, 'VariableNames', names);
results.PsiSummary = mean_table(mean_rows);
results.BestPsi = mean_table(best_mean_row);
results.BestPsiSeeds = array2table(best_psi_seed_rows, 'VariableNames', names);
results.BestRun = array2table(best_run_row, 'VariableNames', names);
results.BestLabels = best_labels;
results.OutputDirectory = output_dir;
results.SearchSeconds = toc(search_start);
write_numeric_csv(fullfile(output_dir, 'best_psi.csv'), results.BestPsi);
write_numeric_csv(fullfile(output_dir, 'best_psi_seeds.csv'), results.BestPsiSeeds);
write_numeric_csv(fullfile(output_dir, 'best_run.csv'), results.BestRun);
write_numeric_csv(fullfile(output_dir, 'best_labels.csv'), best_labels);
for i = 1:numel(options.seeds)
    write_numeric_csv(fullfile(output_dir, sprintf('best_psi_labels_seed_%d.csv', ...
        options.seeds(i))), best_psi_labels{i});
end
write_numeric_csv(fullfile(output_dir, 'search_time.csv'), ...
    table(results.SearchSeconds, 'VariableNames', {'SearchSeconds'}));
% Preserve full precision for the shared MATLAB mean/std summary and redraws.
statistics.Psi = best_run_row(1);
statistics.Seeds = best_psi_seed_rows(:, 2)';
statistics.Qs = best_psi_seed_rows(:, 3)';
statistics.Metrics = best_psi_seed_rows(:, 4:7);
statistics.BestSeed = best_run_row(2);
statistics.BestQ = best_run_row(3);
statistics.BestMetrics = best_run_row(4:7);
statistics.SearchSeconds = results.SearchSeconds;
results.StatisticsFile = fullfile(output_dir, 'search_statistics.json');
fid = fopen(results.StatisticsFile, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'Cannot write search statistics.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(statistics));
fprintf('\nBest psi by mean NMI: %d; NMI=%.4f ARI=%.4f macro-F1=%.4f time=%.4fs\n', best_mean_row);
fprintf('Saved label run: psi=%d seed=%d q=%d NMI=%.4f ARI=%.4f macro-F1=%.4f time=%.4fs\n', ...
    best_run_row(1:7));
fprintf('Total search time: %.4fs\nResults: %s\n', results.SearchSeconds, output_dir);
end

function value = mean_table(rows)
value = array2table(rows, 'VariableNames', ...
    {'Psi', 'MeanNMI', 'MeanARI', 'MeanMacroF1', 'MeanLoadToLabelSeconds'});
end

function write_numeric_csv(filename, values)
% Explicit formatting preserves trailing zeros (e.g. 1.0000) in CSV files.
fid = fopen(filename, 'w');
assert(fid >= 0, 'Cannot write %s.', filename);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', strjoin(values.Properties.VariableNames, ','));
integer_names = {'Psi', 'Seed', 'Q', 'Unassigned', 'SampleIndex'};
formats = repmat({'%.4f'}, 1, width(values));
formats(ismember(values.Properties.VariableNames, integer_names)) = {'%.0f'};
% Original class values must be preserved, even if they are non-integers.
formats(ismember(values.Properties.VariableNames, ...
    {'TrueLabel', 'RawLabel', 'AlignedLabel'})) = {'%.17g'};
format = [strjoin(formats, ',') '\n'];
fprintf(fid, format, table2array(values)');
end
