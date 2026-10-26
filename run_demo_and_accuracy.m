%% RUN_DEMO_AND_ACCURACY Root Entry Point
% Executes the demo and accuracy evaluation suite

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'matlab'));
if exist('setup', 'file')
    setup();
end

run(fullfile(projectRoot, 'matlab', 'run_demo_and_accuracy.m'));
