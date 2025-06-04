function gpb1_kalman_filter()
% GPB1 with Kalman filters for a two-regime log-price model using real observed data
% Structure: Prediction → Update → Statistical Merging (SSDP11 Sec. 2.3.2 and 2.3.4)
% Approximates mixture of Gaussians using moment-matching for each mode (per 2.3.4)

clc; clear;


%% Load data
dataFile = 'financial_data.mat';


%% Estimate expected return mu from external historical dataset

dataFileOld = 'financial_data_old.mat';
if exist(dataFileOld, 'file')
    load(dataFileOld, 'log_prices'); % this loads the OLD data
    log_prices_old = log_prices;
    % Compute empirical mu from old log-prices
    log_returns_old = diff(log_prices_old);
    mu_empirical = mean(log_returns_old);
    sigma2_empirical = var(log_returns_old);
    mu = mu_empirical + 0.5 * sigma2_empirical;
    
     
else
    error('Historical data file "%s" not found.', dataFileOld);
end

if exist(dataFile, 'file')
    load(dataFile, 'log_prices');
else
    error('Data file "%s" not found.', dataFile);
end

% %% Load data
% load('financial_data.mat', 'log_prices');
% load('financial_data_old.mat', 'log_prices');
% log_prices_old = log_prices;
% 
% %% Estimate mu from old data
% log_returns_old = diff(log_prices_old);
% mu_empirical = mean(log_returns_old);
% sigma2_empirical = var(log_returns_old);
% mu = mu_empirical + 0.5 * sigma2_empirical;

%% Model parameters
model.mu = mu;
model.sigma = [0.01, 0.04];
model.P = [0.80, 0.20; 0.20, 0.80];

% Robust estimation of observation noise std-dev from differenced series
residuals = diff(log_prices);
model.sigma_n = median(abs(residuals - median(residuals))) / 0.6745;  % MAD estimate ~ std for Gaussian

%% Initialization
T = length(log_prices);
M = 2;
F = 1; H = 1; Q = model.sigma.^2; R = model.sigma_n^2;

x = repmat(log_prices(1), 1, M);     % Mean estimates per regime
P = repmat(0.001, 1, M);              % Covariance estimates per regime
P_mode = [0.5, 0.5];                 % Initial mode probabilities

x_hist = zeros(T, 1);                % Final estimated means
P_vol = zeros(T, 1);                 % Probability of volatile mode
x_hist(1) = sum(P_mode .* x);
P_vol(1) = P_mode(2);
residuals = zeros(T, 1);  % Errori tra osservazione e stima finale
true_residuals = zeros(T,1);


%% GPB1 Main Loop
for k = 2:T
    z = log_prices(k);                      % Observation at time k
    x_pred = zeros(M, M);                   % Predicted means for each (m_k, m_k-1)
    P_pred = zeros(M, M);                   % Predicted covariances
    p_joint = zeros(M, M);                  % Joint mode probabilities P(m_k, m_k-1 | Z_k)
    ll = zeros(M, M);                       % Measurement likelihoods

    % --- Prediction & Update step ---R_
    for m = 1:M                              % Loop over current mode m_k
        for j = 1:M                          % Loop over previous mode m_k-1
            trans_prob = model.P(j, m);     % Transition probability P(m_k | m_k-1)
            drift = model.mu - 0.5 * model.sigma(m)^2;

            % Kalman prediction
            x_prior = F * x(j) + drift;
            P_prior = F * P(j) * F' + Q(m);

            % Kalman update
            innovation_var = H * P_prior * H' + R;
            K = P_prior * H' / innovation_var;
            x_post = x_prior + K * (z - H * x_prior);
            P_post = (1 - K * H) * P_prior;

            % Likelihood under Gaussian assumption
            ll(m, j) = (1 / sqrt(2 * pi * innovation_var)) * exp(-0.5 * ((z - H * x_prior)^2) / innovation_var);

            % Store posteriors
            x_pred(m, j) = x_post;
            P_pred(m, j) = P_post;

            % Compute unnormalized joint mode probability
            p_joint(m, j) = ll(m, j) * trans_prob * P_mode(j);
        end
    end

    % --- Merging step (Sec. 2.3.4: moment-matching for Gaussian mixtures) ---
    P_mode_new = sum(p_joint, 2)';  % Marginal over m_k-1 → P(m_k | Z_k)
    x_new = zeros(1, M);
    P_new = zeros(1, M);

    for m = 1:M
        weights = p_joint(m, :) / max(sum(p_joint(m, :)), eps);  % Normalize over m_k-1
        x_new(m) = sum(weights .* x_pred(m, :));
        P_new(m) = sum(weights .* (P_pred(m, :) + (x_pred(m, :) - x_new(m)).^2));  % Moment-matching
    end

    % Normalize mode probabilities
    P_mode = P_mode_new / sum(P_mode_new);
    x = x_new;
    P = P_new;

    % Save output
    x_hist(k) = sum(P_mode .* x);
    residuals(k) = x_hist(k) - log_prices(k);
    true_residuals(k) = exp(x_hist(k)) - exp(log_prices(k));
    P_vol(k) = P_mode(2);
end


% Compute global error metrics
error_abs = abs(true_residuals);
MAE = mean(error_abs);
RMSE = sqrt(mean(true_residuals.^2));

fprintf('Mean Absolute Error (MAE): %.4f\n', MAE);
fprintf('Root Mean Square Error (RMSE): %.4f\n', RMSE);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% Log-likelihood cumulata
% loglik_cumulative = 0;
% for k = 2:T
%     mu_k = x_hist(k);
%     z_k = log_prices(k);
%     loglik_k = -0.5 * log(2*pi*model.sigma_n) - 0.5 * ((z_k - mu_k)^2) / model.sigma_n;
%     loglik_cumulative = loglik_cumulative + loglik_k;
% end
% fprintf('Log-likelihood cumulativa: %.2f\n', loglik_cumulative);

% %% Rolling RMSE
% window = 20;
% rolling_rmse = sqrt(movmean((log_prices - x_hist).^2, window));
% figure;
% plot(rolling_rmse, 'LineWidth', 1.2);
% xlabel('Time'); ylabel('Rolling RMSE');
% title(['Rolling RMSE (window = ', num2str(window), ')']);
% grid on;
% 
% %% Entropia P_vol
% entropy_P = - P_vol .* log2(P_vol + eps) - (1 - P_vol) .* log2(1 - P_vol + eps);
% mean_entropy = mean(entropy_P);
% fprintf('Media entropia P_vol: %.3f bits\n', mean_entropy);
% figure;
% plot(entropy_P, 'b');
% xlabel('Time'); ylabel('Entropy (bits)');
% title('Entropy of Regime Probability');
% grid on;
% 
% %% KL-divergence da prior uniforme
% KL_Pvol = P_vol .* log2((P_vol + eps) ./ 0.5) + (1 - P_vol) .* log2(((1 - P_vol) + eps) ./ 0.5);
% mean_KL = mean(KL_Pvol);
% fprintf('KL-divergence media da prior uniforme: %.3f bits\n', mean_KL);
% figure;
% plot(KL_Pvol, 'r');
% xlabel('Time'); ylabel('KL Divergence');
% title('KL divergence da prior 0.5');
% grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Results
figure;
% --- Plot in scala dei prezzi (non log) ---
plot(exp(log_prices), 'k', 'DisplayName', 'Observed Price (AAPL)'); hold on;
plot(exp(x_hist), 'r', 'DisplayName', 'Estimated Price');
xlabel('Time (days)');
ylabel('Close Price ($)');
title('GPB1 + KF: Observed vs Estimated Price');
legend;
grid on;

% 
% subplot(3,1,1);
% plot(log_prices, 'k'); hold on;
% plot(x_hist, 'r');
% legend('Observed', 'Estimated');
% title('GPB1 with Kalman Filter (Moment-Matching Merge)');
% xlabel('Time');
% ylabel('log-price');

figure
plot(P_vol, 'b');
title('Probability of Volatile Regime');
ylabel('P(Volatile)');
xlabel('Time');
ylim([0 1]); grid on;

% --- Plot Errori ---
figure
plot(true_residuals, 'k');
xlabel('Time (days)');
ylabel('Residual');
title('Estimation Error');
grid on;

figure
plot(residuals, 'k');
ylabel('Residuals (log)');
xlabel('Time (days)');
title('Estimation error');
grid on

end
