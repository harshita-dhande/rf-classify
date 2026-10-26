function setup()
% SETUP Root project setup entry point for RF Signal Classification
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    projectRoot = fileparts(mfilename('fullpath'));
    matlabSetupDir = fullfile(projectRoot, 'matlab', 'setup');
    if exist(matlabSetupDir, 'dir')
        addpath(matlabSetupDir);
        addpath(fullfile(projectRoot, 'matlab'));
    end
    
    % Execute canonical setup script
    run(fullfile(matlabSetupDir, 'setup.m'));
end
