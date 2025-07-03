% Author: Hossein Yousefimiab | Date: 2024
% =======================================================================
num_cores = 20;  
parpool('local', num_cores);  

% Logging
log_file = 'optimization_parallel.log';
diary(log_file);
diary on;

for generation = 1:51  
    gen_tic = tic;
    fprintf('%s - Generation %d: here we go again\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation);

    % Run CEGA.exe
    %___________________________________________________
    % IMPORTANT!
    % if the code is going to be run on a windows machine , using
    % system('CEGA.exe') is fine, BUT if using a LINUX machine use
    % system(python 'CEGA.py') and use the raw python code, LINUX cannot
    % run .exe files. dont forget to call python in the slurm
    % such is the art of shell scripting
    %___________________________________________________

    fprintf('%s - About to run CEGA.exe\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    [status, cmdout] = system('CEGA.exe');
    if status ~= 0
        error('%s - CEGA.exe died: %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'), cmdout);
    end
    fprintf('%s - CEGA.exe finished in %.2fs\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), toc(gen_tic));

    % Debugging CEGA
    if ~isfile('TestPoints.txt')
        error('%s - Where is TestPoints.txt? CEGA.exe is acting up.', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    end

    % Patching testpoints.txt
    % this is to avoid the condition when the algorithm tries removing all layers from the base plate

    fprintf('%s - Patching test points (NoLayerPatchGA.exe)\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    [status, cmdout] = system('NoLayerPatchGA.exe');
    if status ~= 0
        error('%s - NoLayerPatchGA.exe failed: %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'), cmdout);
    end

    % Pull in the latest test points
    fprintf('%s - Reading TestPoints.txt\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    test_points = load('TestPoints.txt');
    fprintf('%s - Loaded %d test points\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), size(test_points,1));

    num_test_points = size(test_points, 1);
    results = zeros(num_test_points, 1);

    % parallel loop
    eval_tic = tic;
    parfor i = 1:num_test_points
        test_point = test_points(i, :);
        results(i) = evaluate_test_point(test_point);

        % Log after every single test point for debugging
        fprintf('%s - Line %d: result = %f\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), i, results(i));
    end
    fprintf('%s - All test points done in %.2fs\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), toc(eval_tic));

   
    fid = fopen('Results.txt', 'w');
    for i = 1:num_test_points
        fprintf(fid, '%f\n', results(i));
    end
    fclose(fid);

    fprintf('%s - Generation %d complete! That took %.2fs\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), generation, toc(gen_tic));
end

diary off;
delete(gcp('nocreate'));

%% ---------- Helper functions ----------

function angles = update_angles_from_test_point(test_point)
    % Split a row of TestPoints.txt into named angle arrays.
    % Note: -1 means "no layer here".

    angles.Angle_plate      = test_point(61:80);  
    angles.Angle_stiffener1 = test_point(1:10);   
    angles.Angle_stiffener2 = test_point(11:20);
    angles.Angle_stiffener3 = test_point(21:30);
    angles.Angle_stiffener4 = test_point(31:40);
    angles.Angle_stiffener5 = test_point(41:50);
    angles.Angle_stiffener6 = test_point(51:60);

    % remove -1 values
    fields = fieldnames(angles);
    for f = 1:numel(fields)
        vals = angles.(fields{f});
        angles.(fields{f}) = vals(vals ~= -1);
    end
end

function nat_freq = evaluate_test_point(test_point)
 
    % this part is to be used in conjunction with SEM solver

    t0 = tic;
    angles = update_angles_from_test_point(test_point);

    nat_freq = calculate_wns(angles.Angle_plate, angles.Angle_stiffener1, ...
                             angles.Angle_stiffener2, angles.Angle_stiffener3, ...
                             angles.Angle_stiffener4, angles.Angle_stiffener5, ...
                             angles.Angle_stiffener6);

    fprintf('%s - Test point evaluated in %.2fs\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), toc(t0));
end
