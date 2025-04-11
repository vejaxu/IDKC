function ids = estimate_id(matfile_path, variable_name, k)
    rng(42);  % 保证可重复

    fprintf("Loading %s...\n", matfile_path);
    tic;
    mF = matfile(matfile_path);
    data = mF.(variable_name);
    n = size(data, 1);

    % 可选：采样（避免大型数据集爆内存）
    max_samples = 20000;
    if n > max_samples
        idx = randperm(n, max_samples);
        data = data(idx, :);
        n = max_samples;
    else
        data = data(randperm(n, round(n)), :);  % 打乱
    end

    % 去重（防止重复样本影响 KNN）
    data = unique(data, 'rows');
    n = size(data, 1);
    toc;

    fprintf("Finding kNN (k = %d)...\n", k);
    tic;
    [indices, dist] = knnsearch(data, data, 'K', k+1);  % 包括自身
    indices(:, 1) = [];
    dist(:, 1) = [];
    toc;

    fprintf("Estimating IDs...\n");
    ids = zeros(n, 1);
    tic;
    for i = 1:n
        if mod(i, 1000) == 0
            fprintf('i = %d / %d\n', i, n);
        end
        KNN = data(indices(i,:), :);
        ids(i) = idtle(KNN, dist(i,:));  % 使用你已有的 idtle 函数
    end
    toc;
end
