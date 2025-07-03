% Author: Hossein Yousefimiab | Date: 2024
% =======================================================================
num_cores = 20;
parpool('local', num_cores);

% logging
log_file = 'optimization_parallel.log';
diary(log_file);
diary on;

for generation = 1:101
    generation_start_time = tic;
    fprintf('%s - Starting generation %d\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);

    % Run CEGA.exe
    %___________________________________________________
    % IMPORTANT!
    % if the code is going to be run on a windows machine , using
    % system('CEGA.exe') is fine, BUT if using a LINUX machine use
    % system(python 'CEGA.py') and use the raw python code, LINUX cannot
    % run .exe files. dont forget to call python in the slurm
    % such is the art of shell scripting
    %___________________________________________________

    cega_start_time = tic;
    fprintf('%s - Running CEGA.exe to generate TestPoints.txt\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    [status, cmdout] = system('CEGA.exe');
    if status ~= 0
        error('%s - Error running CEGA.exe: %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'), cmdout);
    end
    fprintf('%s - CEGA.exe completed in %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), toc(cega_start_time));

    % Debugging CEGA
    if ~isfile('TestPoints.txt')
        error('%s - TestPoints.txt not found after running CEGA.', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    end

    % Patching testpoints.txt
    % this is to avoid the condition when the algorithm tries removing all layers from the base plate

    patch_start_time = tic;
    fprintf('%s - Running NoLayerPatchGA.exe to process TestPoints.txt\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    [status, cmdout] = system('NoLayerPatchGA.exe');
    if status ~= 0
        error('%s - Error running NoLayerPatchGA.exe: %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'), cmdout);
    end
    fprintf('%s - NoLayerPatchGA.exe completed in %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), toc(patch_start_time));

    % Load test points from file
    read_start_time = tic;
    fprintf('%s - Reading TestPoints.txt\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    test_points = load('TestPoints.txt');
    fprintf('%s - TestPoints.txt loaded.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf('%s - Reading TestPoints.txt took %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), toc(read_start_time));

    % Evaluate all test points in parallel
    num_test_points = size(test_points, 1);
    results = zeros(num_test_points, 2); % For two objectives: frequency and inverted weight

    eval_start_time = tic;
    parfor i = 1:num_test_points
        test_point = test_points(i, :);
        temp_result = evaluate_test_point(test_point);
        results(i, :) = temp_result;

        % Log the results for this test point
        fprintf('%s - Line %d: Evaluation results = [%f, %f]\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), i, temp_result(1), temp_result(2));
    end
    fprintf('%s - Parallel evaluation completed in %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), toc(eval_start_time));

    % Write results to output file
    results_file = fopen('Results.txt', 'w');
    for i = 1:num_test_points
        fprintf(results_file, '%f %f\n', results(i, 1), results(i, 2));
    end
    fclose(results_file);

    % Log the total duration
    generation_elapsed_time = toc(generation_start_time);
    fprintf('%s - Generation %d completed in %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, generation_elapsed_time);
end

diary off;
delete(gcp('nocreate'));

%% ---- Helper functions ----

function objectives = evaluate_test_point(test_point)
    % this part is to be used in conjunction with SEM solver

    eval_start_time = tic;
    angles = update_angles_from_test_point(test_point);

    % Call the solver 
    [nat_freq, weight] = calculate_wns_multi(angles.Angle_plate, ...
        angles.Angle_stiffener1, angles.Angle_stiffener2, ...
        angles.Angle_stiffener3, angles.Angle_stiffener4, ...
        angles.Angle_stiffener5, angles.Angle_stiffener6);

    % Invert weight to match maximization convention
    inv_weight = -1 * weight;
    objectives = [nat_freq, inv_weight];

    % Log the evaluation time
    eval_elapsed_time = toc(eval_start_time);
    fprintf('%s - Test point evaluation took %.2f seconds.\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), eval_elapsed_time);
end

function angles = update_angles_from_test_point(test_point)

    angles.Angle_plate      = test_point(121:160);   
    angles.Angle_stiffener1 = test_point(1:20);      
    angles.Angle_stiffener2 = test_point(21:40);
    angles.Angle_stiffener3 = test_point(41:60);
    angles.Angle_stiffener4 = test_point(61:80);
    angles.Angle_stiffener5 = test_point(81:100);
    angles.Angle_stiffener6 = test_point(101:120);


    angles.Angle_plate(angles.Angle_plate == -1) = [];
    angles.Angle_stiffener1(angles.Angle_stiffener1 == -1) = [];
    angles.Angle_stiffener2(angles.Angle_stiffener2 == -1) = [];
    angles.Angle_stiffener3(angles.Angle_stiffener3 == -1) = [];
    angles.Angle_stiffener4(angles.Angle_stiffener4 == -1) = [];
    angles.Angle_stiffener5(angles.Angle_stiffener5 == -1) = [];
    angles.Angle_stiffener6(angles.Angle_stiffener6 == -1) = [];
end
