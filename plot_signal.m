%% PLOT_SIGNAL
% Opens the 4-quadrant RF Spectrum Telemetry & AI Classifier Dashboard
% Shows: Time-domain Waveform, Kaiser STFT Spectrogram, FFT Spectrum, and AI Predictions

if exist('setup', 'file')
    setup();
end

plotRFAnalysis('data/examples/sample_wifi_bt_overlap.mat');
