%% RUN_DEMO_AND_ACCURACY
% Standalone script to:
% 1. Setup project paths and environment
% 2. Run forward inference on an example input and show input stats & predictions
% 3. Run comprehensive model evaluation and display accuracy metrics

fprintf('\n=================================================================\n');
fprintf('   RF CLASSIFIER: RUN INFERENCE & ACCURACY EVALUATION           \n');
fprintf('=================================================================\n\n');

% 1. Setup paths
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..');
run(fullfile(scriptDir, 'setup', 'setup.m'));

% 2. Run Inference on a Sample Input
fprintf('\n---------------- STEP 1: SINGLE INPUT INFERENCE -----------------\n');
sampleFile = fullfile(projectRoot, 'data', 'examples', 'sample_wifi_bt_overlap.mat');
fprintf('Loading input sample: %s\n', sampleFile);

sampleData = load(sampleFile);
iq = sampleData.iqSignal;

fprintf('Input Signal Dimensions: %d x %d complex samples\n', size(iq, 1), size(iq, 2));
fprintf('Sampling Rate:           %.1f MSPS\n', sampleData.sampleRate / 1e6);
fprintf('Ground Truth Scenario:   %s\n', char(sampleData.scenario));
fprintf('Active True Protocols:   %s\n', strjoin(sampleData.classNames, ', '));
fprintf('First 3 Raw I/Q Samples: (I=%+.4f, Q=%+.4f), (I=%+.4f, Q=%+.4f), (I=%+.4f, Q=%+.4f)\n', ...
    real(iq(1)), imag(iq(1)), real(iq(2)), imag(iq(2)), real(iq(3)), imag(iq(3)));

% Execute forward inference
[results, activeProtocols, probs] = runInference(sampleFile);

% 3. Evaluate Dataset Accuracy
fprintf('\n---------------- STEP 2: MODEL ACCURACY EVALUATION --------------\n');
datasetFile = fullfile(projectRoot, 'data', 'rf_dataset_smoke.mat');
if ~exist(datasetFile, 'file')
    fprintf('[INFO] Generating validation dataset (smoke mode: 120 samples)...\n');
    generateDataset('smoke', fullfile(projectRoot, 'data'), 42);
end

modelFile = fullfile(projectRoot, 'models', 'trainedNetwork.mat');
evalSummary = evaluateRFClassifier(modelFile, datasetFile, fullfile(projectRoot, 'results'));

fprintf('\n[SUCCESS] Inference and Accuracy Evaluation Complete!\n');
