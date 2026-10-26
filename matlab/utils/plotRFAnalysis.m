function fig = plotRFAnalysis(inputData, modelPath)
% PLOTRFANALYSIS Comprehensive visual RF spectrum telemetry & AI classifier dashboard
%
%   fig = plotRFAnalysis(inputData, modelPath)
%
%   Displays a 4-quadrant publication-quality interactive visualization:
%       Quadrant 1: Time-Domain In-Phase (I) & Quadrature (Q) Baseband Waveform
%       Quadrant 2: Kaiser-Windowed 2D Log-PSD STFT Spectrogram [32 x 33]
%       Quadrant 3: Baseband Power Spectral Density (FFT) & Bandwidth
%       Quadrant 4: I/Q Constellation Phase Trajectory & AI Prediction Probabilities
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 2 || isempty(modelPath)
        modelPath = which('trainedNetwork.mat');
        if isempty(modelPath)
            modelPath = fullfile('models', 'trainedNetwork.mat');
        end
    end

    % 1. Ingest Input Signal
    fs = 20e6; % 20 MSPS
    scenarioName = 'Custom RF Signal';
    groundTruth = {};
    
    if nargin < 1 || isempty(inputData)
        sampleFile = which('sample_wifi_bt_overlap.mat');
        if ~isempty(sampleFile)
            inputData = sampleFile;
        else
            [inputData, meta] = createOverlappingRFExample('S4_WiFi_BT_PartialOverlap', 0);
            scenarioName = meta.scenario;
            groundTruth = meta.activeClassNames;
        end
    end

    if ischar(inputData) || isstring(inputData)
        foundFile = which(inputData);
        if ~isempty(foundFile)
            inputData = foundFile;
        end
        d = load(inputData);
        if isfield(d, 'iqSignal')
            iq = d.iqSignal(:);
        elseif isfield(d, 'X_iq')
            iq = d.X_iq(1, :).';
        end
        if isfield(d, 'scenario'), scenarioName = char(d.scenario); end
        if isfield(d, 'classNames'), groundTruth = d.classNames; end
        if isfield(d, 'sampleRate'), fs = double(d.sampleRate); end
    else
        iq = inputData(:);
    end

    % Ensure 1056 samples
    if length(iq) < 1056
        padded = zeros(1056, 1);
        padded(1:length(iq)) = iq;
        iq = padded;
    elseif length(iq) > 1056
        iq = iq(1:1056);
    end

    % 2. Compute Physical Parameters
    N = length(iq);
    t_us = (0:N-1) / (fs / 1e6); % time in microseconds
    pwr = abs(iq).^2;
    meanPwr = mean(pwr);
    peakPwr = max(pwr);
    papr_dB = 10 * log10(peakPwr / max(meanPwr, 1e-12));
    rmsV = sqrt(meanPwr);

    % 3. Preprocess to Kaiser STFT Spectrogram
    stftTensor = preprocessRFExample(iq, fs); % [32 x 33 x 1]
    spec2D = squeeze(stftTensor);             % [32 x 33]

    % 4. Run Model Inference
    [results, activeProtocols, probs] = runInference(iq, modelPath, 0.5);
    classNames = results.classNames;

    % 5. Compute FFT Power Spectrum
    nFft = 1024;
    fftVals = fftshift(fft(iq, nFft));
    psd_dB = 10 * log10(abs(fftVals).^2 / nFft + 1e-12);
    psd_dB = psd_dB - max(psd_dB); % Normalize peak to 0 dB
    freqMHz = linspace(-fs/2, fs/2, nFft) / 1e6;

    % -------------------------------------------------------------
    % RENDER FIGURE
    % -------------------------------------------------------------
    fig = figure('Name', 'RF Spectrum Telemetry & AI Classification', ...
        'Color', [0.98 0.98 0.98], ...
        'Position', [100, 100, 1150, 750]);

    % Global Title
    sgtitle(sprintf('\\bf 2.4 GHz RF Spectrum Monitoring & AI Classifier \\rm | Scenario: %s', ...
        strrep(scenarioName, '_', '\_')), 'FontSize', 14, 'Color', [0.1 0.1 0.1]);

    % --- SUBPLOT 1: Time-Domain Waveform ---
    subplot(2, 2, 1);
    hold on;
    plot(t_us, real(iq), 'Color', [0.1 0.1 0.15], 'LineWidth', 1.1, 'DisplayName', 'I Channel (Real)');
    plot(t_us, imag(iq), 'Color', [0.55 0.75 0.1], 'LineWidth', 1.0, 'DisplayName', 'Q Channel (Imag)');
    plot(t_us, sqrt(pwr), 'Color', [0.9 0.3 0.1], 'LineStyle', ':', 'LineWidth', 1.2, 'DisplayName', 'Envelope |z|');
    hold off;
    grid on;
    xlabel('Time (\mus)', 'FontSize', 9);
    ylabel('Amplitude (V)', 'FontSize', 9);
    title(sprintf('1D Complex Baseband Waveform (PAPR: %.1f dB | RMS: %.2f V)', papr_dB, rmsV), ...
        'FontSize', 10, 'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 8);
    xlim([0, max(t_us)]);

    % --- SUBPLOT 2: Kaiser 2D STFT Spectrogram ---
    subplot(2, 2, 2);
    timeFrames = 0:31;
    freqBins = linspace(-fs/2, fs/2, 33) / 1e6;
    imagesc(timeFrames, freqBins, spec2D.');
    axis xy;
    colormap(gca, turbo);
    cb = colorbar;
    cb.Label.String = 'Normalized Log-PSD (dB)';
    cb.FontSize = 8;
    xlabel('STFT Time Frames (32 hops)', 'FontSize', 9);
    ylabel('Frequency (MHz)', 'FontSize', 9);
    title('Kaiser-Windowed 2D STFT Spectrogram [32 \times 33] (\beta = 14.0)', ...
        'FontSize', 10, 'FontWeight', 'bold');

    % --- SUBPLOT 3: FFT Power Spectral Density ---
    subplot(2, 2, 3);
    plot(freqMHz, psd_dB, 'Color', [0.15 0.25 0.6], 'LineWidth', 1.2);
    grid on;
    hold on;
    yline(-3, '--r', '-3 dB Cutoff', 'LineWidth', 0.9, 'FontSize', 8);
    yline(-20, ':k', 'Noise Floor Margin', 'LineWidth', 0.8, 'FontSize', 8);
    hold off;
    xlabel('Baseband Frequency (MHz)', 'FontSize', 9);
    ylabel('Relative Power (dB)', 'FontSize', 9);
    title(sprintf('Power Spectral Density (FFT) | Sampling Rate: %.1f MSPS', fs/1e6), ...
        'FontSize', 10, 'FontWeight', 'bold');
    xlim([-10, 10]);
    ylim([-50, 5]);

    % --- SUBPLOT 4: AI Classifier Confidence Bars & Detection ---
    subplot(2, 2, 4);
    barH = barh(probs * 100, 0.55);
    barH.FaceColor = 'flat';
    
    % Color code detected vs below threshold
    for k = 1:length(probs)
        if probs(k) >= 0.5
            barH.CData(k, :) = [0.2 0.75 0.2]; % Green for active
        else
            barH.CData(k, :) = [0.7 0.7 0.75]; % Gray for below threshold
        end
    end
    
    hold on;
    xline(50, '--r', 'Threshold (\tau = 50%)', 'LineWidth', 1.2, 'FontSize', 8, 'LabelVerticalAlignment', 'bottom');
    hold off;
    
    set(gca, 'YTick', 1:length(classNames), 'YTickLabel', classNames, 'FontSize', 9);
    xlabel('Detection Confidence (%)', 'FontSize', 9);
    xlim([0, 105]);
    grid on;
    
    if isempty(activeProtocols)
        detStr = 'None';
    else
        detStr = strjoin(activeProtocols, ' + ');
    end
    title(sprintf('AI Classification (Predicted: %s | Latency: %.2f ms)', detStr, results.latencyMs), ...
        'FontSize', 10, 'FontWeight', 'bold');
end
