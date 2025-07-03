% =======================================================================
% This script runs a high level parallelized optimization loop. It is designed for 
% high efficiency in evaluating many candidate configurations, only recalculating
% what's needed when parts of the design change.
%
% Key workflow features:
%   - Hybrid parallel + batched design to maximize efficiency.
%   - Memoization and result caching at several levels for speed.
%   - Precomputes all possible (on/off) group layouts for fast access.
%   - Only recalculates matrix contributions for the groups ("plate" or each "stiffener") that 
%     change within a batch.
%   - Calculates both natural frequency and weight on the fly for each candidate.
%
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

% Load the element connectivity and nodal coordinate
fprintf('%s - Loading connectivity and node data.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
load conn_case1
load nodes_case1

elements = conn;
nodes = nodes;
Lx_plate = max(nodes(:,1)) - min(nodes(:,1));  % Plate x-length (meters)
Ly_plate = max(nodes(:,2)) - min(nodes(:,2));  % Plate y-length (meters)
delta = 0.1;

% Compute index mappings and polynomial orders for the spectral element mesh
% this parameters are going to stay constant thoughout the optimization
[indA, indR, elementpoints, polynum_xi, polynum_eta] = ...
    element_division_index_conn(elements, nodes, Lx_plate, delta);

% Store a parallel constant of indA expensive to transfer repeatedly
% indA is a huge matrix , it is not logical for each worker to load it each
% time, we make it a constant variable for all workers in parallel pool
constIndA = parallel.pool.Constant(indA);

% logging
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
    is_active = logical(binary_str - '0');
    if ~is_active(1)
        continue;  
    end

    % struct showing which groups are present
    angle_config = struct();
    for g = 1:7
        if is_active(g)
            angle_config.(angle_fields{g}) = rand(1, 10);
        else
            angle_config.(angle_fields{g}) = -1 * ones(1, 10);
        end
    end

    fprintf('%s - Precompute posn | config_idx=%d | binary=%s | activeGroups=[%s]\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), config_idx, binary_str, sprintf('%d', is_active));

    % Call posn_calculator 
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

% ------------------ MAIN OPTIMIZATION LOOP ------------------
% For each generation:
%   1. Generate new test points (candidate stacking sequences) with CEGA.
%   2. Patch any illegal configurations with NoLayerPatchGA.
%   3. If present, process "Optimum.txt" (log the best results for this generation).
%   4. Load the test points, process them in batches where only one group changes.
%   5. Within each batch, cache matrix factorizations for the static groups;
%      only recalculate what changes.
%   6. Evaluate each candidate in parallel for both frequency and weight.
%   7. Write the results for this generation.

num_generations = 101;
batch_size = 750;  

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

    if isfile('Optimum.txt')
        fprintf('%s - Optimum.txt found, processing optimum values.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fid = fopen('Optimum.txt', 'r');
        lines = {};
        while ~feof(fid)
            line = fgetl(fid);
            if ischar(line) && ~isempty(strtrim(line))
                lines{end+1} = line; %#ok<AGROW>
            end
        end
        fclose(fid);
        output_str = sprintf('Optimums for generation %d were:\n', generation);
        for j = 1:length(lines)
            tokens = strsplit(strtrim(lines{j}));
            if length(tokens) >= 2
                num1 = tokens{end-1};
                num2 = tokens{end};
            else
                num1 = 'NaN';
                num2 = 'NaN';
            end
            output_str = [output_str, sprintf('Line %d: %s %s\n', j, num1, num2)]; %#ok<AGROW>
        end
        new_filename = 'Optimum_processed.txt';
        fid_new = fopen(new_filename, 'a');
        fprintf(fid_new, '%s', output_str);
        fclose(fid_new);
        fprintf('%s', output_str);
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
    results = zeros(num_test_points, 2);  % Two objectives: freq and weight
    num_batches = ceil(num_test_points / batch_size);
    fprintf('%s - [Gen %d] Processing %d points in %d batches.\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, num_test_points, num_batches);

    for batch_idx = 1:num_batches
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = min(batch_idx * batch_size, num_test_points);
        current_batch = test_points(start_idx:end_idx, :);
        batch_length = size(current_batch, 1);

        % --- Find which group is changing in this batch ---
        changing_group_idx = find_changing_group_in_batch(current_batch);
        fprintf('%s - [Gen %d] [Batch %d/%d] Changing group: %d | Batch lines: %d-%d\n', ...
            datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, batch_idx, num_batches, ...
            changing_group_idx, start_idx, end_idx);

        % Indices of elements that are dynamic
        switch changing_group_idx
            case 1, di_dynamic = 1:18;
            case 2, di_dynamic = 19:23;
            case 3, di_dynamic = 24:28;
            case 4, di_dynamic = 29:31;
            case 5, di_dynamic = 32:34;
            case 6, di_dynamic = 35:37;
            case 7, di_dynamic = 38:40;
            otherwise, error('Unexpected changing_group_idx: %d', changing_group_idx);
        end

        % Indices of all static elements
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
        constCachedKa = parallel.pool.Constant(cached_Ka);
        constCachedMa = parallel.pool.Constant(cached_Ma);

        % Evaluate all test points in the batch
        batch_results = zeros(batch_length, 2);
        parfor line_in_batch = 1:batch_length
            maxNumCompThreads(1);  % Keep each worker to a single thread (important for eigs)

            % Only the changing group is updated for each testpoint
            current_tp = update_angles_from_test_point(current_batch(line_in_batch, :));
            updated_config = cached_config;
            fields = fieldnames(cached_config);
            updated_config.(fields{changing_group_idx}) = current_tp.(fields{changing_group_idx});

            % Calculate frequency using cached matrices
            freq = getFrequencyFromConfig(updated_config, elements, nodes, indR, elementpoints, ...
                polynum_xi, polynum_eta, constCachedKa, constCachedMa, constIndA, posn_cache, di_dynamic);

            % --- Calculate the weight of this configuration on-the-fly ---
            % (Geometry and material parameters are set at the top of this block.)
            h_t_p = 0.02;
            h_t_s = 0.01;
            h_n_p = h_t_p / 80;
            h_n_s = h_t_s / 40;
            Ly_stiffener = 0.1;
            rho = 1500;

            h_plate    = length(updated_config.Angle_plate)      * 2 * h_n_p;
            h_stiff_1  = length(updated_config.Angle_stiffener1) * 2 * h_n_s;
            h_stiff_2  = length(updated_config.Angle_stiffener2) * 2 * h_n_s;
            h_stiff_3  = length(updated_config.Angle_stiffener3) * 2 * h_n_s;
            h_stiff_4  = length(updated_config.Angle_stiffener4) * 2 * h_n_s;
            h_stiff_5  = length(updated_config.Angle_stiffener5) * 2 * h_n_s;
            h_stiff_6  = length(updated_config.Angle_stiffener6) * 2 * h_n_s;

            % Plate area and volume
            Lx_plate_local = Lx_plate;
            Ly_plate_local = Ly_plate;
            A_plate = Lx_plate_local * Ly_plate_local;
            V_plate = A_plate * h_plate;

            % Stiffener volumes (first two run in x, last four run in y)
            Lx_stiff_1to2 = Lx_plate_local;
            V_stiff_1to2 = Ly_stiffener * Lx_stiff_1to2 * [h_stiff_1, h_stiff_2];
            Lx_stiff_3to6 = Ly_plate_local;
            V_stiff_3to6 = Ly_stiffener * Lx_stiff_3to6 * [h_stiff_3, h_stiff_4, h_stiff_5, h_stiff_6];

            V_stiff_total = sum(V_stiff_1to2) + sum(V_stiff_3to6);
            V_total = V_plate + V_stiff_total;
            weight_val = V_total * rho;
            weight = -weight_val;  % Negative because algorithm is a maximizer

            batch_results(line_in_batch, :) = [freq, weight];
        end
        results(start_idx:end_idx, :) = batch_results;
    end

    % --- Write all results for this generation to file (freq, weight) ---
    write_results_to_file(results);
    fprintf('%s - Generation %d completed in %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, toc(tic_gen));
end

delete(gcp('nocreate'));
diary off;

% ================= SUPPORTING FUNCTIONS =================

function angles = update_angles_from_test_point(test_point)
    % Splits testpoints
    % Removes all -1 values.
    angles.Angle_plate = test_point(121:160);
    angles.Angle_stiffener1 = test_point(1:20);
    angles.Angle_stiffener2 = test_point(21:40);
    angles.Angle_stiffener3 = test_point(41:60);
    angles.Angle_stiffener4 = test_point(61:80);
    angles.Angle_stiffener5 = test_point(81:100);
    angles.Angle_stiffener6 = test_point(101:120);
    fields = fieldnames(angles);
    for i = 1:numel(fields)
        f = fields{i};
        angles.(f)(angles.(f) == -1) = [];
    end
end

function changing_group_idx = identify_changed_group(first, second)
    % Finds which group changes.
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
    % Detects which group varies in a batch of test points.
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
    % Returns integer config index based on which groups are present.
    is_active = zeros(1, 7);
    fields_ordered = {'Angle_plate','Angle_stiffener1','Angle_stiffener2', ...
                      'Angle_stiffener3','Angle_stiffener4','Angle_stiffener5','Angle_stiffener6'};
    for i = 1:7
        is_active(i) = ~isempty(angle_struct.(fields_ordered{i}));
    end
    config_idx = bin2dec(num2str(is_active));
end

function w = solve_eigenvalue_problem(Ka, Ma, indA, posn)
    % Solves the eigenvalue problem, caches results.
    persistent cache_eig maxCacheSize usageOrder
    if isempty(cache_eig)
        cache_eig = containers.Map();
        maxCacheSize = 1000;
        usageOrder = {};
    end
    indF = find(sum(abs(Ma)) == 0);
    Ka(indF, :) = [];
    Ka(:, indF) = [];
    Ma(indF, :) = [];
    Ma(:, indF) = [];
    indA(indF) = [];
    posn(indF, :) = [];
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
    % Computes eigenfrequency for  stacking sequence .
    % Uses cached static matrices, only recalculates what changes.
    persistent configCache maxCacheSize configUsage
    if isempty(configCache)
        configCache = containers.Map();
        maxCacheSize = 10000;
        configUsage = {};
    end
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
    configCache(key) = freq;
    configUsage{end+1} = key;
    if numel(configUsage) > maxCacheSize
        lruKey = configUsage{1};
        remove(configCache, lruKey);
        configUsage(1) = [];
    end
end

function write_results_to_file(results)
    fid = fopen('Results.txt', 'w');
    for i = 1:size(results,1)
        fprintf(fid, '%.6f %.6f\n', results(i,1), results(i,2));
    end
    fclose(fid);
end
