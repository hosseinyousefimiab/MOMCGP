% =========================================================
% This script runs a high level parallelized optimization loop. It is designed for 
% high efficiency in evaluating many candidate configurations, only recalculating
% what's needed when parts of the design change.
% Key concepts:
% - Hybrid parallel/batched workflow with heavy use of caching.(memoization
%   is implemented liberaly)
% - Precomputing element/node data and all possible positional parameters.
% - Each "generation" consists of producing new test points, patching them,
%   and then evaluating them in batches.
% - Within each batch, only the *changing* part of the design is re-computed.
% - memory-efficient caches.
% Author: Hossein Yousefimiab | Date: 2024
% =======================================================================

num_cores = 20;
fprintf('%s - Initializing %d workers.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), num_cores);
parpool('local', num_cores);
pool = gcp();

% Ensure all workers have access to the DataHash
% IMPORTANT! DataHash.m is needed for caching, make sure it exists in the
% code folder, it is also available online
dhashPath = which('DataHash.m');
if isempty(dhashPath)
    error('DataHash.m not found. Please download it and add its location to the MATLAB path.');
end
addAttachedFiles(pool, {dhashPath});

% --- Load mesh connectivity and node locations ---
fprintf('%s - Loading connectivity and node data.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
load conn_case4     
load nodes_case4  

elements = conn;
nodes = nodes;
Lx_plate = max(nodes(:,1)) - min(nodes(:,1));  
delta = 0.1;                                   

% Compute index mappings and polynomial orders for the spectral element mesh
% this parameters are going to stay constant thoughout the optimization
[indA, indR, elementpoints, polynum_xi, polynum_eta] = ...
    element_division_index_conn(elements, nodes, Lx_plate, delta);

% Store a parallel constant of indA expensive to transfer repeatedly
% indA is a huge matrix , it is not logical for each worker to load it each
% time, we make it a constant variable for all workers in parallel pool
constIndA = parallel.pool.Constant(indA);

% Logging
log_file = 'optimization_parallel.log';
fprintf('%s - Starting diary logging in %s.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), log_file);
diary(log_file);
diary on;

% ======================= PRECOMPUTATION =======================
% Precompute all possible configurations of the "active/inactive" status 
% of the 7 groups (plate + 6 stiffeners). The results are cached in 
% posn_cache, a cell array, so that positional arrangements don't have to 
% be recomputed in the main loop. 
% i am extremely proud of this part!

num_configs = 2^7; 
posn_cache = cell(1, num_configs);
angle_fields = {'Angle_plate', 'Angle_stiffener1', 'Angle_stiffener2', ...
    'Angle_stiffener3', 'Angle_stiffener4', 'Angle_stiffener5', 'Angle_stiffener6'};

for config_idx = 0:(num_configs - 1)
    binary_str = dec2bin(config_idx, 7);
    is_active = logical(binary_str - '0'); % 1 = group is present; 0 = removed

    if ~is_active(1)
        continue;
    end

    % struct showing which groups are present
    angle_config = struct();
    for g = 1:7
        if is_active(g)
            angle_config.(angle_fields{g}) = rand(1, 10); 
        else
            angle_config.(angle_fields{g}) = -1 * ones(1, 10); % fill with -1 for removed
        end
    end

    fprintf('%s - Precompute posn | config_idx=%d | binary=%s | activeGroups=[%s]\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), config_idx, binary_str, sprintf('%d', is_active));

    % cache the positional arrangement
    posn_cache{config_idx + 1} = posn_calculator(elements, nodes, ...
        angle_config.Angle_plate, ...
        angle_config.Angle_stiffener1, ...
        angle_config.Angle_stiffener2, ...
        angle_config.Angle_stiffener3, ...
        angle_config.Angle_stiffener4, ...
        angle_config.Angle_stiffener5, ...
        angle_config.Angle_stiffener6);
end

fprintf('%s - Precomputation of posn completed!\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

% ======================= MAIN OPTIMIZATION LOOP =======================
% For each generation:
%   - Generate testpoints using CEGA.exe
%   - Patch test points to fix removed baseplate
%   - Load test points and process them in batches.
%   - For each batch, determine the design group that is changing.
%   - Re-use as much computation as possible from the previous configuration.
%   - Run parallel evaluations, only updating contribution of the changing group.

num_generations = 51;
batch_size = 375;

fprintf('%s - Starting optimization for %d generations.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), num_generations);

for generation = 1:num_generations
    tic_gen = tic;
    fprintf('\n%s - Starting generation %d.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);

    % Run CEGA.exe
    %___________________________________________________
    % IMPORTANT!
    % if the code is going to be run on a windows machine , using
    % system('CEGA.exe') is fine, BUT if using a LINUX machine use
    % system(python 'CEGA.py') and use the raw python code, LINUX cannot
    % run .exe files. dont forget to call python in the slurm
    % such is the art of shell scripting
    %___________________________________________________
    fprintf('%s - [Gen %d] Running CEGA.exe...\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);
    [status, cmdout] = system('CEGA.exe');
    if status ~= 0
        fprintf('%s - [Gen %d] ERROR: CEGA.exe failed with status %d. Output:\n%s\n', ...
            datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, status, cmdout);
        error('CEGA failed in generation %d', generation);
    else
        fprintf('%s - [Gen %d] CEGA.exe completed successfully.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);
    end

    fprintf('%s - [Gen %d] Running NoLayerPatchGA.exe...\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);
    [status, cmdout] = system('NoLayerPatchGA.exe');
    if status ~= 0
        fprintf('%s - [Gen %d] ERROR: NoLayerPatchGA.exe failed with status %d. Output:\n%s\n', ...
            datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, status, cmdout);
        error('NoLayerPatchGA failed in generation %d', generation);
    else
        fprintf('%s - [Gen %d] NoLayerPatchGA.exe completed successfully.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);
    end

    fprintf('%s - [Gen %d] Loading TestPoints.txt...\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);
    if ~isfile('TestPoints.txt')
        fprintf('%s - [Gen %d] ERROR: TestPoints.txt not found.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);
        error('TestPoints.txt not found in generation %d', generation);
    end
    test_points = load('TestPoints.txt');
    fprintf('%s - [Gen %d] TestPoints.txt loaded with %d points.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, size(test_points, 1));

    % --- Batch process for efficiency ---
    % Each batch contains test points where only one group changes.
    % The rest are  fixed and their variables are precomputed and cached.
    num_test_points = size(test_points, 1);
    results = zeros(num_test_points, 1);
    num_batches = ceil(num_test_points / batch_size);
    fprintf('%s - [Gen %d] Processing %d points in %d batches.\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, num_test_points, num_batches);

    for batch_idx = 1:num_batches
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = min(batch_idx * batch_size, num_test_points);
        current_batch = test_points(start_idx:end_idx, :);
        batch_length = size(current_batch, 1);

        %  find which group changes in this batch
        % Only the dynamic group's stacking sequence varies in a batch, everything else is fixed.
        changing_group_idx = find_changing_group_in_batch(current_batch);
        fprintf('%s - [Gen %d] [Batch %d/%d] Changing group: %d | Batch lines: %d-%d\n', ...
            datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, batch_idx, num_batches, ...
            changing_group_idx, start_idx, end_idx);

        % Map group index to corresponding element indices in the mesh
        switch changing_group_idx
            case 1, di_dynamic = 1:18;   % Plate
            case 2, di_dynamic = 19:23;  % Stiffener 1
            case 3, di_dynamic = 24:28;  % Stiffener 2
            case 4, di_dynamic = 29:31;  % Stiffener 3
            case 5, di_dynamic = 32:34;  % Stiffener 4
            case 6, di_dynamic = 35:37;  % Stiffener 5
            case 7, di_dynamic = 38:40;  % Stiffener 6
            otherwise, error('Unexpected changing_group_idx: %d', changing_group_idx);
        end

        % Indices of "static" elements (everything not in the dynamic group)
        di_static = setdiff(1:size(elements, 1), di_dynamic);

        % Precompute matrices for static parts of the design
        % it is only done once for each batch
        cached_config = update_angles_from_test_point(current_batch(1, :));
        [cached_Ka, cached_Ma] = calculate_wns3(elements, nodes, ...
            cached_config.Angle_plate, ...
            cached_config.Angle_stiffener1, ...
            cached_config.Angle_stiffener2, ...
            cached_config.Angle_stiffener3, ...
            cached_config.Angle_stiffener4, ...
            cached_config.Angle_stiffener5, ...
            cached_config.Angle_stiffener6, ...
            indR, elementpoints, polynum_xi, polynum_eta, di_static);

        % Make constants for parallel workers to access fast
        constCachedKa = parallel.pool.Constant(cached_Ka);
        constCachedMa = parallel.pool.Constant(cached_Ma);

        % Evaluate all test points
        batch_results = zeros(batch_length, 1);
        parfor line_in_batch = 1:batch_length
            % Limit threads to one per worker
            % This is to make the eigs function inside solver use only one
            % worker, avoiding overhead
            maxNumCompThreads(1);

            % Get angles from the testpoint, update only the dynamic group
            current_tp = update_angles_from_test_point(current_batch(line_in_batch, :));
            updated_config = cached_config;
            fields = fieldnames(cached_config);
            updated_config.(fields{changing_group_idx}) = current_tp.(fields{changing_group_idx});

            % Compute natural frequency
            freq = getFrequencyFromConfig(updated_config, elements, nodes, indR, elementpoints, ...
                polynum_xi, polynum_eta, constCachedKa, constCachedMa, constIndA, posn_cache, di_dynamic);
            batch_results(line_in_batch) = freq;
        end

        % Write results 
        results(start_idx:end_idx) = batch_results;
    end

    % Write all results for this generation to file 
    write_results_to_file(results);
    fprintf('%s - Generation %d completed in %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, toc(tic_gen));
end

delete(gcp('nocreate'));
diary off;

% ================= SUPPORTING FUNCTIONS =================

function angles = update_angles_from_test_point(test_point)
    % split test_point data into angle arrays for each group.
    % Removes -1  entries.
    angles.Angle_plate = test_point(61:80);
    angles.Angle_stiffener1 = test_point(1:10);
    angles.Angle_stiffener2 = test_point(11:20);
    angles.Angle_stiffener3 = test_point(21:30);
    angles.Angle_stiffener4 = test_point(31:40);
    angles.Angle_stiffener5 = test_point(41:50);
    angles.Angle_stiffener6 = test_point(51:60);
    fields = fieldnames(angles);
    for i = 1:numel(fields)
        f = fields{i};
        angles.(f)(angles.(f) == -1) = [];
    end
end

function changing_group_idx = identify_changed_group(first, second)
    fields = fieldnames(first);
    for i = 1:numel(fields)
        if ~isequal(first.(fields{i}), second.(fields{i}))
            changing_group_idx = i;
            return;
        end
    end
    error('No changing group found.');
end

function changing_group_idx = find_changing_group_in_batch(batch)
    ref = update_angles_from_test_point(batch(1, :));
    for line = 2:size(batch, 1)
        current = update_angles_from_test_point(batch(line, :));
        try
            changing_group_idx = identify_changed_group(ref, current);
            return;
        catch
        end
    end
    error('No changing group found in the batch.');
end

function config_idx = compute_config_idx(angle_struct)
    is_active = zeros(1, 7);
    fields_ordered = {'Angle_plate','Angle_stiffener1','Angle_stiffener2', ...
                      'Angle_stiffener3','Angle_stiffener4','Angle_stiffener5','Angle_stiffener6'};
    for i = 1:7
        is_active(i) = ~isempty(angle_struct.(fields_ordered{i}));
    end
    config_idx = bin2dec(num2str(is_active));
end

function w = solve_eigenvalue_problem(Ka, Ma, indA, posn)
    % Solves the eigenvalue problem.
    % If a solution for this (Ka, Ma, indA, posn) tuple has been seen before, re-use it.
    persistent cache_eig maxCacheSize usageOrder
    if isempty(cache_eig)
        cache_eig = containers.Map();
        maxCacheSize = 1000;
        usageOrder = {};
    end
    % Remove all rows/cols for unreferenced DOFs
    indF = find(sum(abs(Ma)) == 0);
    Ka(indF, :) = [];
    Ka(:, indF) = [];
    Ma(indF, :) = [];
    Ma(:, indF) = [];
    indA(indF) = [];
    posn(indF, :) = [];
    % Build cache key and check for previous solution
    key = DataHash({Ka, Ma, indA, posn});
    if isKey(cache_eig, key)
        w = cache_eig(key);
        usageOrder = updateUsage(usageOrder, key);
        return;
    end
    BCs = ['C', 'C', 'C', 'C'];
    [Ka, Ma, indA, posn] = Boundary_Conditions3_plate(Ka, Ma, indA, BCs, posn);
    shift = 0.01;
    OPTS.disp = 0;
    [~, eigVal] = eigs(Ka + shift * Ma, Ma, 7, 'smallestabs', OPTS);
    wns = sort(real(sqrt(diag(eigVal) - shift)));
    w = wns(1) / (2 * pi); 
    % Update cache
    cache_eig(key) = w;
    usageOrder{end+1} = key;
    if numel(usageOrder) > maxCacheSize
        lruKey = usageOrder{1};
        remove(cache_eig, lruKey);
        usageOrder(1) = [];
    end
end

function usage = updateUsage(usage, key)
    idx = find(strcmp(usage, key), 1);
    if ~isempty(idx)
        usage(idx) = [];
    end
    usage{end+1} = key;
end

function freq = getFrequencyFromConfig(updated_config, elements, nodes, indR, elementpoints, polynum_xi, polynum_eta, constCachedKa, constCachedMa, constIndA, posn_cache, di_range)
    % Returns eigenfrequency for a configuration.
    % Uses cached matrices for static groups, only recalculating the dynamic group.
    persistent configCache maxConfigCache configUsage
    if isempty(configCache)
        configCache = containers.Map();
        maxConfigCache = 10000;  
        configUsage = {};
    end
    % Use DataHash for fast, collision-resistant cache lookup.
    key = DataHash(updated_config);
    if isKey(configCache, key)
        freq = configCache(key);
        configUsage = updateUsage(configUsage, key);
        return;
    end
    [upd_Ka, upd_Ma] = calculate_wns3(elements, nodes, ...
        updated_config.Angle_plate, ...
        updated_config.Angle_stiffener1, ...
        updated_config.Angle_stiffener2, ...
        updated_config.Angle_stiffener3, ...
        updated_config.Angle_stiffener4, ...
        updated_config.Angle_stiffener5, ...
        updated_config.Angle_stiffener6, ...
        indR, elementpoints, polynum_xi, polynum_eta, di_range);
    Ka = constCachedKa.Value + upd_Ka;
    Ma = constCachedMa.Value + upd_Ma;
    final_config_idx = compute_config_idx(updated_config);
    posn_final = posn_cache{final_config_idx + 1};
    freq = solve_eigenvalue_problem(Ka, Ma, constIndA.Value, posn_final);
    % Update cache
    configCache(key) = freq;
    configUsage{end+1} = key;
    if numel(configUsage) > maxConfigCache
        lruKey = configUsage{1};
        remove(configCache, lruKey);
        configUsage(1) = [];
    end
end

function write_results_to_file(results)
    fid = fopen('Results.txt', 'w');
    fprintf(fid, '%.6f\n', results);
    fclose(fid);
end
