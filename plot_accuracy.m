%% PLOT_ACCURACY
% Plots multi-label performance, SNR noise resilience curve, and overlap regime benchmarks

if exist('setup', 'file')
    setup();
end

plotModelPerformance();
