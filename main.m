% Search q for each (psi, seed), then select psi by mean NMI.
project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir, 'IDKC'));
addpath(fullfile(project_dir, 'experiments'));
addpath(fullfile(project_dir, 'utils'));
addpath(fullfile(project_dir, 'metrics'));

% Choose a MAT file containing data and class.
data_file = fullfile(project_dir, 'results', '_inputs', 'dlpfc_151507.mat');
options.psi_values = [2, 4, 6, 8, 16, 24, 32, 48, 64, 80, 100, 200, 250, 500, 750];
options.seeds = [42, 3407, 4079, 2024, 0];
options.q_fractions = 0.05:0.05:0.5;
options.t = 100;
options.v = 0.9;
options.sample_size = 10000;

[~, dataset_name] = fileparts(data_file);
output_dir = fullfile(project_dir, 'results', dataset_name);
results = run_idkc_search(data_file, output_dir, options);
