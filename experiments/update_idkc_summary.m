function filename = update_idkc_summary(data_file, output_dir, results, options)
% Upsert one dataset into the shared CSV beside per-dataset directories.
[~, dataset] = fileparts(data_file);
summary_dir = fileparts(output_dir);
if isempty(summary_dir)
    summary_dir = '.';
end
if ~isfolder(summary_dir)
    mkdir(summary_dir);
end
filename = fullfile(summary_dir, 'idkc.csv');
average = results.BestPsi;
best = results.BestRun;
seeds = strjoin(string(results.BestPsiSeeds.Seed'), ';');
seed_qs = strjoin(string(results.BestPsiSeeds.Q'), ';');
names = {'Dataset', 'BestPsi', 'MeanNMI', 'MeanARI', 'MeanMacroF1', ...
    'MeanLoadToLabelSeconds', 'Seeds', 'SeedBestQs', 'T', 'GrowthRate', ...
    'SeedSampleSize', 'BestRunPsi', 'BestRunSeed', 'BestRunQ', 'BestRunNMI', ...
    'BestRunARI', 'BestRunMacroF1', 'BestRunLoadToLabelSeconds', ...
    'BestRunUnassigned', 'SearchSeconds'};
row = table(string(dataset), average.Psi, average.MeanNMI, average.MeanARI, ...
    average.MeanMacroF1, average.MeanLoadToLabelSeconds, seeds, seed_qs, ...
    options.t, options.v, min(height(results.BestLabels), options.sample_size), ...
    best.Psi, best.Seed, best.Q, best.NMI, best.ARI, best.MacroF1, ...
    best.LoadToLabelSeconds, best.Unassigned, results.SearchSeconds, ...
    'VariableNames', names);
text_names = {'Dataset', 'Seeds', 'SeedBestQs'};
if isfile(filename)
    import_options = detectImportOptions(filename, 'TextType', 'string', ...
        'VariableNamingRule', 'preserve');
    assert(isequal(import_options.VariableNames, names), ...
        'IDKC:SummarySchema', 'Unexpected column layout in %s.', filename);
    import_options = setvartype(import_options, text_names, 'string');
    import_options = setvartype(import_options, setdiff(names, text_names), 'double');
    existing = readtable(filename, import_options);
    matches = find(strcmpi(existing.Dataset, dataset));
    if isempty(matches)
        existing = [existing; row];
    else
        existing(matches(1), :) = row;
        existing(matches(2:end), :) = []; % Collapse any duplicate records.
    end
else
    existing = row;
end

% Replace only after the complete CSV is written, retaining prior rows.
temporary_file = [tempname(summary_dir) '.csv'];
cleanup = onCleanup(@() remove_temporary_file(temporary_file)); %#ok<NASGU>
write_summary_csv(temporary_file, existing, text_names);
[success, message] = movefile(temporary_file, filename, 'f');
assert(success, 'IDKC:SummaryWrite', 'Cannot update %s: %s', filename, message);
end

function write_summary_csv(filename, values, text_names)
fid = fopen(filename, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'Cannot write %s.', filename);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
names = values.Properties.VariableNames;
fprintf(fid, '%s\n', strjoin(names, ','));
integer_names = {'BestPsi', 'T', 'SeedSampleSize', 'BestRunPsi', ...
    'BestRunSeed', 'BestRunQ', 'BestRunUnassigned'};
for i = 1:height(values)
    fields = cell(1, width(values));
    for j = 1:width(values)
        value = values.(names{j})(i);
        if ismember(names{j}, text_names)
            % Quote all strings, including dataset names with commas/quotes.
            fields{j} = ['"' strrep(char(value), '"', '""') '"'];
        elseif ismember(names{j}, integer_names)
            fields{j} = sprintf('%.0f', value);
        else
            fields{j} = sprintf('%.4f', value);
        end
    end
    fprintf(fid, '%s\n', strjoin(fields, ','));
end
end

function remove_temporary_file(filename)
if isfile(filename)
    delete(filename);
end
end
