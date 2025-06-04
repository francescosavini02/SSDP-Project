%% GPB1 Kalman vs Particle Filter - Full Comparison Script
clc; clear; close all;

%% Load Data
load('financial_data_old.mat', 'log_prices'); % for mu estimation
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

%% Run Kalman Filter
[x_hist, P_vol_kf, res_kf, true_res_kf] = run_gpb1_kalman(log_prices, model);

%% Run Full Particle Filter GPB1
[mean_est, P_vol_pf, res_pf, true_res_pf, joint_particles, joint_weights] = run_gpb1_pf_full(log_prices, model);

%% Error Metrics
mae_kf = mean(abs(true_res_kf));
mae_pf = mean(abs(true_res_pf));
rmse_kf = sqrt(mean(true_res_kf.^2));
rmse_pf = sqrt(mean(true_res_pf.^2));

fprintf('\n--- Kalman GPB1 ---\nMAE = %.4f | RMSE = %.4f\n', mae_kf, rmse_kf);
fprintf('--- Particle GPB1 ---\nMAE = %.4f | RMSE = %.4f\n', mae_pf, rmse_pf);

%% Plot Results
figure; plot(exp(log_prices), 'k'); hold on;
plot(exp(x_hist), 'r'); plot(exp(mean_est), 'b');
legend('Observed', 'Kalman', 'PF'); title('Price Estimation');
xlabel('Time'); ylabel('Price ($)'); grid on;

figure; plot(P_vol_kf, 'r'); hold on;
plot(P_vol_pf, 'b'); title('P(Volatile)');
xlabel('Time'); ylabel('Probability'); legend('Kalman', 'PF');
ylim([0 1]); grid on;

figure; plot(true_res_kf, 'r'); hold on;
plot(true_res_pf, 'b'); title('True Residuals');
xlabel('Time'); ylabel('Residual'); legend('Kalman', 'PF'); grid on;

figure; plot(res_kf, 'r'); hold on;
plot(res_pf, 'b'); title('Log Residuals');
xlabel('Time'); ylabel('Log Residual'); legend('Kalman', 'PF'); grid on;

%% Entropy
entropy_kf = - P_vol_kf .* log2(P_vol_kf + eps) - (1 - P_vol_kf) .* log2(1 - P_vol_kf + eps);
entropy_pf = - P_vol_pf .* log2(P_vol_pf + eps) - (1 - P_vol_pf) .* log2(1 - P_vol_pf + eps);

figure; plot(entropy_kf, 'r'); hold on; plot(entropy_pf, 'b');
title('Entropy of P(Volatile)'); xlabel('Time'); ylabel('Entropy (bits)');
legend('Kalman', 'PF'); grid on;

%% KL Divergence
KL_kf = P_vol_kf .* log2((P_vol_kf + eps) ./ 0.5) + (1 - P_vol_kf) .* log2((1 - P_vol_kf + eps) ./ 0.5);
KL_pf = P_vol_pf .* log2((P_vol_pf + eps) ./ 0.5) + (1 - P_vol_pf) .* log2((1 - P_vol_pf + eps) ./ 0.5);

figure; plot(KL_kf, 'r'); hold on; plot(KL_pf, 'b');
title('KL Divergence from Uniform Prior'); xlabel('Time'); ylabel('KL Divergence (bits)');
legend('Kalman', 'PF'); grid on;

%% Rolling RMSE
rolling_rmse_kf = sqrt(movmean((log_prices(:) - x_hist(:)).^2, 20));
rolling_rmse_pf = sqrt(movmean((log_prices(:) - mean_est(:)).^2, 20));



figure;
plot(rolling_rmse_kf, 'r', 'LineWidth', 1.5); hold on;
plot(rolling_rmse_pf, 'b', 'LineWidth', 1.5);
title('Rolling RMSE (Window = 20)');
xlabel('Time'); ylabel('RMSE');
legend('Kalman', 'PF'); grid on;


%% Final Particle Cloud
kplot = length(log_prices);
s = joint_particles{kplot}(:,1);
m = joint_particles{kplot}(:,2);
m_jittered = m + 0.15 * randn(size(m));  % Jitter più ampio
[~, sort_idx] = sort(m);  % Ordina per regime

figure;
gscatter(s(sort_idx), m_jittered(sort_idx), m(sort_idx), ['r', 'b'], 'xo', 6, 'on');
xlabel('log-price particles');
ylabel('Regime (jittered)');
title(sprintf('Joint Posterior at Time Step k = %d', kplot));
legend('Calm Regime (0)', 'Volatile Regime (1)', 'Location', 'best');
ylim([-0.2 1.5]); grid on;

%% === Full GPB1 PF Function ===
function [mean_est, P_vol, residuals, true_residuals, joint_particles, joint_weights] = run_gpb1_pf_full(log_prices, model)
    N = 500; M = 2; Tmax = length(log_prices); s0 = log_prices(1);
    particles_prev = zeros(N, M);
    particles_prev(:,1) = s0 + model.sigma(1) * randn(N,1);
    particles_prev(:,2) = s0 + model.sigma(2) * randn(N,1);
    weights_prev = ones(N, M) / N; P_mode_prev = [0.5, 0.5];
    mean_est = zeros(1, Tmax); P_vol = zeros(1, Tmax);
    residuals = zeros(1, Tmax); true_residuals = zeros(1, Tmax);
    joint_particles = cell(1, Tmax); joint_weights = cell(1, Tmax);

    mean_est(1) = sum(P_mode_prev .* [mean(particles_prev(:,1)), mean(particles_prev(:,2))]);
    P_vol(1) = P_mode_prev(2);

    for t = 2:Tmax
        z = log_prices(t);
        particles_pair = zeros(N, M, M); weights_pair = zeros(N, M, M);

        for Rt = 1:M
            for Rprev = 1:M
                drift = model.mu - 0.5 * model.sigma(Rt)^2;
                s_pred = particles_prev(:, Rprev) + drift + model.sigma(Rt) * randn(N,1);
                particles_pair(:,Rt,Rprev) = s_pred;

                logL = -((z - s_pred).^2) / (2 * model.sigma_n^2);
                logL = logL - max(logL); L = exp(logL);
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
            all_particles = [particles_pair(:,Rt,1); particles_pair(:,Rt,2)];
            all_weights = [weights_pair(:,Rt,1); weights_pair(:,Rt,2)];
            all_weights = all_weights / max(sum(all_weights), eps);
            idx = randsample(length(all_particles), N, true, all_weights);
            new_particles(:,Rt) = all_particles(idx);
        end

        particles_prev = new_particles;
        weights_prev = ones(N, M) / N;
        P_mode_prev = alpha;

        mean_est(t) = alpha(1)*mean(new_particles(:,1)) + alpha(2)*mean(new_particles(:,2));
        P_vol(t) = alpha(2);

        s_joint = [new_particles(:,1); new_particles(:,2)];
        m_joint = [zeros(N,1); ones(N,1)];
        w_joint = [ones(N,1)/N * alpha(1); ones(N,1)/N * alpha(2)];
        joint_particles{t} = [s_joint, m_joint]; joint_weights{t} = w_joint;
    end

    for t = 1:Tmax
        residuals(t) = log_prices(t) - mean_est(t);
        true_residuals(t) = exp(log_prices(t)) - exp(mean_est(t));
    end
end

%% === GPB1 Kalman Filter Function ===
function [x_hist, P_vol, residuals, true_residuals] = run_gpb1_kalman(log_prices, model)
    T = length(log_prices); M = 2;
    F = 1; H = 1; Q = model.sigma.^2; R = model.sigma_n^2;

    x = repmat(log_prices(1), 1, M);     % Mean estimates per regime
    P = repmat(0.001, 1, M);             % Covariance estimates per regime
    P_mode = [0.5, 0.5];                 % Initial regime probabilities

    x_hist = zeros(T, 1);                % Final estimated means
    P_vol = zeros(T, 1);                 % Probability of volatile regime
    residuals = zeros(T, 1);             % log-price residuals
    true_residuals = zeros(T, 1);        % true (price domain) residuals

    x_hist(1) = sum(P_mode .* x);
    P_vol(1) = P_mode(2);

    for k = 2:T
        z = log_prices(k);                      % Observation at time k
        x_pred = zeros(M, M);                   % Predicted means
        P_pred = zeros(M, M);                   % Predicted covariances
        p_joint = zeros(M, M);                  % Joint regime probabilities
        ll = zeros(M, M);                       % Measurement likelihoods

        for m = 1:M                              % Loop over R_t
            for j = 1:M                          % Loop over R_{t-1}
                trans_prob = model.P(j, m);     % Transition P(R_t|R_{t-1})
                drift = model.mu - 0.5 * model.sigma(m)^2;

                x_prior = F * x(j) + drift;
                P_prior = F * P(j) * F' + Q(m);

                innovation_var = H * P_prior * H' + R;
                K = P_prior * H' / innovation_var;
                x_post = x_prior + K * (z - H * x_prior);
                P_post = (1 - K * H) * P_prior;

                ll(m, j) = (1 / sqrt(2 * pi * innovation_var)) * ...
                            exp(-0.5 * ((z - H * x_prior)^2) / innovation_var);

                x_pred(m, j) = x_post;
                P_pred(m, j) = P_post;
                p_joint(m, j) = ll(m, j) * trans_prob * P_mode(j);
            end
        end

        % Merging (moment matching)
        P_mode_new = sum(p_joint, 2)';
        x_new = zeros(1, M); P_new = zeros(1, M);
        for m = 1:M
            weights = p_joint(m, :) / max(sum(p_joint(m, :)), eps);
            x_new(m) = sum(weights .* x_pred(m, :));
            P_new(m) = sum(weights .* (P_pred(m, :) + (x_pred(m, :) - x_new(m)).^2));
        end

        % Normalize and update
        P_mode = P_mode_new / sum(P_mode_new);
        x = x_new;
        P = P_new;

        % Save results
        x_hist(k) = sum(P_mode .* x);
        residuals(k) = x_hist(k) - log_prices(k);
        true_residuals(k) = exp(x_hist(k)) - exp(log_prices(k));
        P_vol(k) = P_mode(2);
    end
end

