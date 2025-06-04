%% Full GPB1 Particle Filter Implementation with Regime Pairs (R_t, R_{t-1})
clc; clear; close all;

%% Load Data
load('financial_data_old.mat', 'log_prices');
log_prices_old = log_prices;
load('financial_data.mat', 'log_prices');


%% Estimate Empirical Parameters
log_returns_old = diff(log_prices_old);
mu_empirical = mean(log_returns_old);
sigma2_empirical = var(log_returns_old);
mu = mu_empirical + 0.5 * sigma2_empirical;

model.mu = mu;
model.sigma = [0.01, 0.04];
model.P = [0.80, 0.20; 0.20, 0.80];
model.sigma_n = median(abs(diff(log_prices) - median(diff(log_prices)))) / 0.6745;

%% Run GPB1 Particle Filter (Full Equations)
[mean_est, P_vol, res_pf, true_res_pf, joint_particles, joint_weights] = run_gpb1_pf_full(log_prices, model);

%% Plot Results
figure;
drawnow
pause(0.1)
plot(exp(log_prices), 'k'); hold on;
plot(exp(mean_est), 'r');
legend('Observed', 'PF (Full GPB1)');
title('Price Estimation');
xlabel('Time'); ylabel('Price ($)'); grid on;

figure;
drawnow
pause(0.1)
plot(P_vol, 'b');
title('P(Volatile)'); xlabel('Time'); ylabel('Probability');
ylim([0 1]); grid on;

figure;
drawnow
pause(0.1)
plot(res_pf, 'k');
xlabel('Time'); ylabel('Residual (log-price)');
title('Estimation Error (Log Domain)');
grid on;

figure;
drawnow
pause(0.1)
plot(true_res_pf, 'k');
xlabel('Time'); ylabel('Residual (price)');
title('Estimation Error (Price Domain)');
grid on;


%% Compute MAE and RMSE
error_abs = abs(true_res_pf);
MAE = mean(error_abs);
RMSE = sqrt(mean(true_res_pf.^2));

fprintf('Mean Absolute Error (MAE): %.4f\n', MAE);
fprintf('Root Mean Square Error (RMSE): %.4f\n', RMSE);

%% Plot Joint Particle Cloud at Last Time Step
figure
kplot = length(log_prices);
s = joint_particles{kplot}(:,1);
m = joint_particles{kplot}(:,2);
m_jittered = m + 0.05 * randn(size(m));
gscatter(s, m_jittered, m, ['r', 'b'], 'xo', 5);
xlabel('log-price particles');
ylabel('Regime');
title(['Joint Posterior at Time Step k = ', num2str(kplot)]);
legend('Calm Regime (0)', 'Volatile Regime (1)', 'Location', 'best');
grid on;

%% === Funzione principale ===
function [mean_est, P_vol, residuals, true_residuals, joint_particles, joint_weights] = run_gpb1_pf_full(log_prices, model)
    N = 500; M = 2; Tmax = length(log_prices);
    s0 = log_prices(1);

    particles_prev = zeros(N, M);
    particles_prev(:,1) = s0 + model.sigma(1) * randn(N,1);
    particles_prev(:,2) = s0 + model.sigma(2) * randn(N,1);

    weights_prev = ones(N, M) / N;
    P_mode_prev = [0.5, 0.5];

    mean_est = zeros(1, Tmax);
    P_vol = zeros(1, Tmax);
    residuals = zeros(1, Tmax);
    true_residuals = zeros(1, Tmax);

    joint_particles = cell(1, Tmax);
    joint_weights = cell(1, Tmax);

    mean_est(1) = P_mode_prev(1)*mean(particles_prev(:,1)) + P_mode_prev(2)*mean(particles_prev(:,2));
    P_vol(1) = P_mode_prev(2);

    for t = 2:Tmax
        z = log_prices(t);

        particles_pair = zeros(N, M, M);
        weights_pair = zeros(N, M, M);
        for Rt = 1:M
            for Rprev = 1:M
                drift = model.mu - 0.5 * model.sigma(Rt)^2;
                sigma = model.sigma(Rt);
                s_prev = particles_prev(:, Rprev);
                s_pred = s_prev + drift + sigma * randn(N,1);
                particles_pair(:,Rt,Rprev) = s_pred;

                logL = -((z - s_pred).^2) / (2 * model.sigma_n^2);
                logL = logL - max(logL);
                L = exp(logL);

                weights_pair(:,Rt,Rprev) = weights_prev(:,Rprev) .* L * model.P(Rprev, Rt);
            end
        end

        alpha = zeros(1,M);
        for Rt = 1:M
            total = 0;
            for Rprev = 1:M
                total = total + sum(weights_pair(:,Rt,Rprev));
            end
            alpha(Rt) = total;
        end
        alpha = alpha / sum(alpha);

        new_particles = zeros(N, M);
        for Rt = 1:M
            merged_particles = [];
            merged_weights = [];
            for Rprev = 1:M
                merged_particles = [merged_particles; particles_pair(:,Rt,Rprev)];
                merged_weights = [merged_weights; weights_pair(:,Rt,Rprev)];
            end
            merged_weights = merged_weights / sum(merged_weights + eps);
            idx = randsample(size(merged_particles,1), N, true, merged_weights);
            new_particles(:,Rt) = merged_particles(idx);
        end

        particles_prev = new_particles;
        weights_prev = ones(N, M) / N;
        P_mode_prev = alpha;

        mean_est(t) = 0;
        for Rt = 1:M
            mean_est(t) = mean_est(t) + alpha(Rt) * mean(new_particles(:,Rt));
        end
        P_vol(t) = alpha(2);

        s_joint = [new_particles(:,1); new_particles(:,2)];
        m_joint = [zeros(N,1); ones(N,1)];
        w_joint = [ones(N,1)/N * alpha(1); ones(N,1)/N * alpha(2)];
        joint_particles{t} = [s_joint, m_joint];
        joint_weights{t} = w_joint;
    end

    for t = 1:Tmax
        residuals(t) = log_prices(t) - mean_est(t);
        true_residuals(t) = exp(log_prices(t)) - exp(mean_est(t));
    end
end