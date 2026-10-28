function filename = update_idkc_summary(info, output_dir, results)
% Update one dataset in the only shared report, preserving all other rows.
names = {'dataset','points','dimensions','clusters','nmi','nmi_std', ...
    'ari','ari_std','f1','f1_std','time','time_std','psi','best_seed', ...
    'best_q','best_nmi','best_ari','best_f1','best_time'};
selected = results.BestPsiSeeds;
best = results.BestRun;
assert(height(selected) == 5 && isequal(sort(selected.Seed)', [0,42,2024,3407,4079]) ...
    && all(selected.Psi == results.BestPsi.Psi) && best.Psi == results.BestPsi.Psi ...
    && best.NMI == max(selected.NMI), 'Invalid best-psi/seed results.');
average = table2array(results.BestPsi(:,2:5));
deviation = std(table2array(selected(:,4:7)), 0, 1); % Sample SD, n-1.
row = table(string(info.dataset), info.points, info.dimensions, info.clusters, ...
    average(1), deviation(1), average(2), deviation(2), average(3), deviation(3), ...
    average(4), deviation(4), best.Psi, best.Seed, best.Q, ...
    best.NMI, best.ARI, best.MacroF1, best.LoadToLabelSeconds, 'VariableNames', names);
assert(all(isfinite(table2array(row(:,2:end)))), 'Nonfinite summary values.');
root = fileparts(output_dir);
filename = fullfile(root, 'idkc.csv');
if isfile(filename)
    import_options = detectImportOptions(filename, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    assert(isequal(import_options.VariableNames, names), 'Unexpected summary columns.');
    import_options = setvartype(import_options, 'dataset', 'string');
    import_options = setvartype(import_options, names(2:end), 'double');
    existing = readtable(filename, import_options);
    matches = find(strcmpi(existing.dataset, info.dataset));
    if isempty(matches)
        existing = [existing; row];
    else
        existing(matches(1), :) = row;
        existing(matches(2:end), :) = [];
    end
else
    existing = row;
end
temporary = [tempname(root) '.csv'];
cleanup = onCleanup(@() remove_temporary(temporary)); %#ok<NASGU>
write_summary(temporary, existing);
[ok, message] = movefile(temporary, filename, 'f');
assert(ok, '%s', message);
end

function write_summary(filename, values)
fid = fopen(filename, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'Cannot write summary CSV.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
names = values.Properties.VariableNames;
fprintf(fid, '%s\n', strjoin(names, ','));
integers = {'points','dimensions','clusters','psi','best_seed','best_q'};
for i = 1:height(values)
    fields = cell(1,width(values));
    fields{1} = ['"' strrep(char(values.dataset(i)), '"', '""') '"'];
    for j = 2:width(values)
        if ismember(names{j}, integers)
            fields{j} = sprintf('%.0f', values.(names{j})(i));
        else
            fields{j} = sprintf('%.4f', values.(names{j})(i));
        end
    end
    fprintf(fid, '%s\n', strjoin(fields, ','));
end
end

function remove_temporary(filename)
if isfile(filename)
    delete(filename);
end
end
