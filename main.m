function results = main(dataset, config)
% Complete one dataset: load -> normalize/search -> plot -> publish/summary.
% Example: main('AC', struct('data_root', 'E:\data\DC', 'python', 'python'))
% main('AC', struct('plot_only', true)) redraws saved labels without searching.
if nargin < 1
    dataset = 'dlpfc_151507';
end
if nargin < 2
    config = struct();
end
project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir, 'IDKC'), fullfile(project_dir, 'experiments'), ...
    fullfile(project_dir, 'metrics'));
defaults.data_root = fullfile(fileparts(fileparts(project_dir)), 'data', 'DC');
defaults.results_root = fullfile(project_dir, 'results');
defaults.python = getenv('IDKC_PYTHON');
if isempty(defaults.python)
    defaults.python = 'python';
end
defaults.input_file = '';
defaults.x_key = 'data';
defaults.y_key = 'class';
defaults.pos_key = '';
defaults.plot_only = false;
defaults.plot_sample_size = 5000; % Legacy option; plotting always uses all samples.
defaults.plot_seed = 42;
defaults.psi_values = [2,4,6,8,16,24,32,48,64,80,100,200,250,500,750];
defaults.q_fractions = 0.05:0.05:0.5;
defaults.t = 100;
defaults.v = 0.9;
defaults.sample_size = 10000;
fields = fieldnames(config);
assert(all(ismember(fields, fieldnames(defaults))), 'Unknown configuration field.');
for i = 1:numel(fields)
    defaults.(fields{i}) = config.(fields{i});
end
config = defaults;
dataset = char(dataset);
assert(~isempty(dataset) && ~ismember(dataset(1), '._') ...
    && isempty(regexp(dataset, '[\\/:*?"<>|\r\n]', 'once')), 'Invalid dataset name.');
assert(config.plot_seed >= 0 && config.plot_seed == floor(config.plot_seed), 'Invalid plot seed.');
options = struct('psi_values', config.psi_values(:)', ...
    'seeds', [42,3407,4079,2024,0], 'q_fractions', config.q_fractions(:)', ...
    't', config.t, 'v', config.v, 'sample_size', config.sample_size);
run_root = fullfile(project_dir, '.local', 'runs', dataset);
if ~isfolder(run_root)
    mkdir(run_root);
end
run_dir = tempname(run_root);
mkdir(run_dir);
helper = fullfile(project_dir, 'pipeline_helper.py');
args = {config.python, helper, 'prepare', '--work-dir', run_dir};
if isempty(config.input_file)
    args = [args, {'--key', dataset, '--data-root', config.data_root}];
else
    args = [args, {'--input-file', config.input_file, '--name', dataset, ...
        '--x-key', config.x_key, '--y-key', config.y_key}];
    if ~isempty(config.pos_key)
        args = [args, {'--pos-key', config.pos_key}];
    end
end
if config.plot_only
    args = [args, {'--skip-mat'}];
end
run_helper(args, fullfile(run_dir, 'prepare.log'));
info = jsondecode(fileread(fullfile(run_dir, 'dataset.json')));
staged = fullfile(run_dir, 'result');
output_dir = fullfile(config.results_root, dataset);
if config.plot_only
    assert(isfile(fullfile(output_dir, 'best_run.csv')), 'No saved result for this dataset.');
    [ok, message] = copyfile(output_dir, staged);
    assert(ok, '%s', message);
    results = read_saved_results(staged);
else
    results = run_idkc_search(fullfile(run_dir, 'input.mat'), staged, options);
end
metadata = info;
metadata.search_options = options;
if config.plot_only && isfile(fullfile(staged, 'run_metadata.json'))
    old = jsondecode(fileread(fullfile(staged, 'run_metadata.json')));
    if isfield(old, 'search_options')
        metadata.search_options = old.search_options;
    else
        metadata = rmfield(metadata, 'search_options');
    end
elseif config.plot_only
    metadata = rmfield(metadata, 'search_options');
end
metadata.plot_seed = config.plot_seed;
metadata.plot_sample_size = info.points;
metadata_file = fullfile(staged, 'run_metadata.json');
fid = fopen(metadata_file, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'Cannot write run metadata.');
fprintf(fid, '%s\n', jsonencode(metadata));
fclose(fid);
run_helper({config.python, helper, 'plot', '--work-dir', run_dir, ...
    '--seed', num2str(config.plot_seed)}, fullfile(run_dir, 'plot.log'));
% Publish only after search, label validation and both plots have succeeded.
if ~isfolder(config.results_root)
    mkdir(config.results_root);
end
previous = fullfile(run_dir, 'previous_result');
if isfolder(output_dir)
    [ok, message] = movefile(output_dir, previous);
    assert(ok, '%s', message);
end
try
    [ok, message] = movefile(staged, output_dir);
    assert(ok, '%s', message);
    results.SummaryFile = update_idkc_summary(info, output_dir, results);
catch error
    if isfolder(output_dir)
        % Keep partial output apart from any staging folder left by a failed move.
        [restored, restore_message] = movefile(output_dir, fullfile(run_dir, 'failed_result'));
        if ~restored
            warning('IDKC:Restore', 'Could not archive failed output: %s', restore_message);
        end
    end
    if isfolder(previous)
        if isfolder(output_dir)
            warning('IDKC:Restore', 'Previous result remains in %s; destination could not be cleared.', previous);
        else
            [restored, restore_message] = movefile(previous, output_dir);
            if ~restored
                warning('IDKC:Restore', 'Could not restore previous result: %s', restore_message);
            end
        end
    end
    rethrow(error);
end
results.OutputDirectory = output_dir;
if isfield(results, 'StatisticsFile')
    results.StatisticsFile = fullfile(output_dir, 'search_statistics.json');
end
fprintf('\nCompleted %s\nResults: %s\nSummary: %s\n', dataset, output_dir, results.SummaryFile);
end

function run_helper(args, logfile)
quoted = cell(size(args));
for i = 1:numel(args)
    value = char(args{i});
    if ispc
        assert(isempty(regexp(value, '["%%!\r\n]', 'once')), 'Unsupported command argument.');
        quoted{i} = ['"' value '"'];
    else
        % Single quotes prevent the shell from expanding paths or argument text.
        quoted{i} = ['''' strrep(value, '''', '''"''"''') ''''];
    end
end
command = strjoin(quoted, ' ');
if ispc
    % cmd.exe removes the outer pair, preserving each quoted path/argument.
    command = ['"' command '"'];
end
[status, output] = system(command);
fid = fopen(logfile, 'w', 'n', 'UTF-8');
if fid >= 0
    fprintf(fid, '%s', output);
    fclose(fid);
end
assert(status == 0, 'IDKC:PythonHelper', 'Data/plot helper failed; see %s\n%s', logfile, output);
end

function results = read_saved_results(directory)
results.BestPsi = readtable(fullfile(directory, 'best_psi.csv'));
results.BestPsiSeeds = readtable(fullfile(directory, 'best_psi_seeds.csv'));
results.BestRun = readtable(fullfile(directory, 'best_run.csv'));
filename = fullfile(directory, 'search_statistics.json');
if isfile(filename)
    statistics = jsondecode(fileread(filename));
    assert(statistics.Psi == results.BestPsi.Psi && statistics.Psi == results.BestRun.Psi ...
        && statistics.BestSeed == results.BestRun.Seed && statistics.BestQ == results.BestRun.Q ...
        && isequal(statistics.Seeds(:), results.BestPsiSeeds.Seed(:)) ...
        && isequal(statistics.Qs(:), results.BestPsiSeeds.Q(:)) ...
        && max(abs(statistics.Metrics - table2array(results.BestPsiSeeds(:,4:7))), [], 'all') < 0.000051, ...
        'Saved full-precision statistics do not match the selected run.');
    results.BestPsiSeeds{:,4:7} = statistics.Metrics;
    results.BestPsi{1,2:5} = mean(statistics.Metrics, 1);
    assert(max(abs(statistics.BestMetrics(:)' - table2array(results.BestRun(:,4:7)))) < 0.000051, ...
        'Saved full-precision best metrics do not match best_run.csv.');
    results.BestRun{1,4:7} = statistics.BestMetrics(:)';
end
end
