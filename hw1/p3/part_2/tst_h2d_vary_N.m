% Test ADI solver varying N (grid size) and measuring runtime
% Plots errors and computation times

warning('off', 'all');  % Suppress Octave internal warnings

e2 = 1;
% Scale dt with h^2 = (1/N)^2 so temporal error matches spatial error
% This lets us observe O(h^2) spatial convergence
dt_base = 0.1;  % dt when N=20

% Array of N values to test
N_vals = [20 40 60 80 100 120 160 200];
n_tests = length(N_vals);

% Storage for results
errors = zeros(n_tests, 1);
times = zeros(n_tests, 1);

fprintf('N\t\tdt\t\t\ttime\t\terror\t\truntime\n');
fprintf('------------------------------------------------------------\n');

for k = 1:n_tests
    N = N_vals(k);
    dt = dt_base * (1/N)^2;  % Scale dt with h^2
    t0 = tic;
    heat2d;
    times(k) = toc(t0);
    errors(k) = e2;
    fprintf('%d\t\t%.6f\t\t%.2f\t\t%.2e\t\t%.4f s\n', N, dt, time, e2, times(k));
end

% Create plots
figure(1); clf;

% Plot 1: Error vs N
subplot(2,2,1);
loglog(N_vals, errors, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('N (grid points)', 'FontSize', 12);
ylabel('Relative Error', 'FontSize', 12);
title('Error vs Grid Size', 'FontSize', 14);
grid on;

% Plot 2: Runtime vs N
subplot(2,2,2);
loglog(N_vals, times, 'rs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('N (grid points)', 'FontSize', 12);
ylabel('Runtime (s)', 'FontSize', 12);
title('Runtime vs Grid Size', 'FontSize', 14);
grid on;

% Plot 3: Runtime vs N with reference lines
subplot(2,2,3);
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

% Plot 4: Error vs N with reference line
subplot(2,2,4);
loglog(N_vals, errors, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
% Add O(h^2) reference line (h ~ 1/N, so error ~ 1/N^2)
c = errors(1) * N_vals(1)^2;
loglog(N_vals, c ./ N_vals.^2, 'k--', 'LineWidth', 1);
xlabel('N (grid points)', 'FontSize', 12);
ylabel('Relative Error', 'FontSize', 12);
title('Error Convergence (dashed = O(h^2))', 'FontSize', 14);
legend('ADI', 'O(h^2)', 'Location', 'NorthEast');
grid on;

drawnow;  % Force figure to render

% Display summary
fprintf('\n--- Summary ---\n');
fprintf('Total runtime: %.2f s\n', sum(times));

% Keep plot open
input('Press Enter to close...', 's');
