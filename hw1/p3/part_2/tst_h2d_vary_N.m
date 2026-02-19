% Test ADI solver varying N (grid size) and measuring runtime
% Plots runtime with O(N^2) reference line

warning('off', 'all');  % Suppress Octave internal warnings

e2 = 1;
dt = 0.001;  % Fixed time step for all runs

% Array of N values to test
N_vals = [20 40 60 80 100 120 160 200];
n_tests = length(N_vals);

% Storage for results
times = zeros(n_tests, 1);

fprintf('N\t\tdt\t\t\truntime\n');
fprintf('----------------------------------------\n');

for k = 1:n_tests
    N = N_vals(k);
    t0 = tic;
    heat2d;
    times(k) = toc(t0);
    fprintf('%d\t\t%.6f\t\t%.4f s\n', N, dt, times(k));
end

% Create plot: Runtime vs N with O(N^2) reference line
figure(1); clf;
loglog(N_vals, times, 'rs-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
% Add O(N^2) reference line
c = times(end) / N_vals(end)^2;
loglog(N_vals, c * N_vals.^2, 'k--', 'LineWidth', 1);
xlabel('N (grid points)', 'FontSize', 12);
ylabel('Runtime (s)', 'FontSize', 12);
title('Runtime Scaling (dashed = O(N^2))', 'FontSize', 14);
legend('ADI', 'O(N^2)', 'Location', 'NorthWest');
grid on;

drawnow;  % Force figure to render

% Display summary
fprintf('\n--- Summary ---\n');
fprintf('Total runtime: %.2f s\n', sum(times));

% Keep plot open
input('Press Enter to close...', 's');
