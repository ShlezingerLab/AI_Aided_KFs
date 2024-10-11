clc
clear
close all

% ╭───────────────────────────────────────────────────────────╮
% │               load data                                   │
% ╰───────────────────────────────────────────────────────────╯
% Preallocate a 3D array to store the data
x_data = zeros(10, 3, 3000);
y_data = zeros(10, 3, 3000);
% Loop over the 10 files
n_data = 10;
for i_data = 1:10
    % Read the CSV file into a matrix
    file_name = ['data/test_x_', num2str(i_data), '.csv'];
    tempData = readmatrix(file_name);
    x_data(i_data, :, :) = tempData;

    file_name = ['data/test_y_', num2str(i_data), '.csv'];
    tempData = readmatrix(file_name);
    y_data(i_data, :, :) = tempData;
end

% ╭───────────────────────────────────────────────────────────╮
% │               common system settings                      │
% ╰───────────────────────────────────────────────────────────╯
x_dim = 3;
y_dim = 3;
n_track = 3000;
delta_t = 0.02; % sampling rate
J = 2; % order of Taylor Expansion

q = 0.3873;
Q = q^2 * eye(x_dim);
r = 1;
R = r^2 * eye(y_dim);

% x0 = [1; 1; 1];
x0 = x_data(1, :, 1)';
P0 = zeros(3,3);

error_type = "first-element"; %"norm"; "element-wise"

% ╭───────────────────────────────────────────────────────────╮
% │                    filter settings                        │
% ╰───────────────────────────────────────────────────────────╯
filters.model = {};
filters.filter = {};
filters.step = {};
filters.name = {};
% =====================================
% ============ True Model =============
% =====================================
% m.x_dim = x_dim; % state dimension
% m.y_dim = y_dim; % measurement dimension
% m.x0 = x0;
% m.P0 = P0;
% m.ffun = @f_true;
% m.Q = Q;
% m.hfun = @hfun;
% m.R = R;
% 
% % comment the following line to disable the filter
% filters = add_filter(filters,m,@true_initialize,@true_step,'TRUE');
% clear m % to ensure nothing is preserved from previous model

% =====================================
% ============== CKF ATRUE ==============
% =====================================
m.x_dim = x_dim; % state dimension
m.y_dim = y_dim; % measurement dimension
% m.NN = tmlp(m.x_dim, m.y_dim, 5);
m.NN = tmlp(m.x_dim, 9, [5 5]);
theta = m.NN.get_params(); % getting NN parameters
m.nn_dim = length(theta); % number of NN parameter
m.ax_dim = m.x_dim + m.nn_dim; % augmented state dimension

m.x0 = [x0; theta];
m.P0 = blkdiag(P0, 1e-2*eye(m.nn_dim));
m.ffun = @f_atrue;
m.Q = blkdiag(Q, 1e-6*eye(m.nn_dim));
m.y_pseudo = zeros(m.nn_dim, 1);
m.y_pseudo_dim = m.nn_dim; 
m.ay_dim = m.y_dim + m.y_pseudo_dim;
m.hfun = @hfun_apbm;
lambda_apbm = 0.05;
m.R = blkdiag(R,(1/lambda_apbm)*eye(m.nn_dim));

% comment the following line to disable the filter
filters = add_filter(filters,m,@atrue_initialize,@atrue_step,'ATRUE');
clear m % to ensure nothing is preserved from previous model

% =====================================
% ============== CKF PBM ==============
% =====================================
% m.x_dim = x_dim; % state dimension
% m.y_dim = y_dim; % measurement dimension
% m.x0 = x0;
% m.P0 = P0;
% m.ffun = @f_pbm;
% m.Q = Q;
% m.hfun = @hfun;
% m.R = R;
% 
% % comment the following line to disable the filter
% filters = add_filter(filters,m,@pbm_initialize,@pbm_step,'PBM');
% clear m % to ensure nothing is preserved from previous model

% =====================================
% ============== CKF APBM ==============
% =====================================
% m.x_dim = x_dim; % state dimension
% m.y_dim = y_dim; % measurement dimension
% m.NN = tmlp(m.x_dim, m.y_dim, 5);
% theta = m.NN.get_params(); % getting NN parameters
% m.nn_dim = length(theta); % number of NN parameter
% m.ax_dim = m.x_dim + m.nn_dim; % augmented state dimension
% 
% m.x0 = [x0; theta];
% m.P0 = blkdiag(P0, 1e-2*eye(m.nn_dim));
% m.ffun = @f_apbm;
% m.Q = blkdiag(Q, 1e-6*eye(m.nn_dim));
% m.y_pseudo = zeros(m.nn_dim, 1);
% m.y_pseudo_dim = m.nn_dim; 
% m.ay_dim = m.y_dim + m.y_pseudo_dim;
% m.hfun = @hfun_apbm;
% lambda_apbm = 0.05;
% m.R = blkdiag(R,(1/lambda_apbm)*eye(m.nn_dim));
% 
% % comment the following line to disable the filter
% filters = add_filter(filters,m,@apbm_initialize,@apbm_step,'APBM');
% clear m % to ensure nothing is preserved from previous model


% ╭───────────────────────────────────────────────────────────╮
% │                      RUN SIMULATIONS                      │
% ╰───────────────────────────────────────────────────────────╯
n_filter = length(filters.filter);
% memory allocation
save_x_error = cell(1, n_filter);
for i_filter = 1:n_filter
    save_x_error{i_filter} = zeros(n_data, n_track);
end

for i_data = 1:n_data % loop for datasets
    % memory allocation
    save_xhat = cell(1,n_filter);
    for i_filter = 1:n_filter
        save_xhat{i_filter} = zeros(x_dim,n_track);
    end
    % extract true state
    x = x_data(i_data,:,:);
    x = reshape(x, x_dim, n_track);
    % filtering 
    for i_track = 1:n_track % loop for time steps
        for i_filter = 1:n_filter % loop for filters
            % extract measurement
            y = y_data(i_data, :, i_track)';
            [xCorr,pCorr,xPred,pPred] = filters.step{i_filter}(filters.filter{i_filter},...
                delta_t, J, y, filters.model{i_filter});
            save_xhat{i_filter}(:,i_track) = xCorr(1:3);
        end
    end
    % compute error
    for i_filter = 1:n_filter
        error = save_xhat{i_filter} - x;
        if error_type == "norm"
            save_x_error{i_filter}(i_data,:) = vecnorm(error, 2, 1);
        end
        if error_type == "element-wise"
            save_x_error{i_filter}(i_data,:) = vecnorm(error, 2, 1).^2 / x_dim ;
        end
        if error_type == "first-element"
            save_x_error{i_filter}(i_data,:) = error(1,:) .^ 2;
        end
    end

    % reset state and NN parameter
    for i_filter = 1:n_filter
        m.x_dim = x_dim; % state dimension
        m.y_dim = y_dim; % measurement dimension
        % m.NN = tmlp(m.x_dim, m.y_dim, 5);
        m.NN = tmlp(m.x_dim, 9, [5]);
        theta = m.NN.get_params(); % getting NN parameters
        m.nn_dim = length(theta); % number of NN parameter
        m.ax_dim = m.x_dim + m.nn_dim; % augmented state dimension
        
        m.x0 = [x0; theta];
        m.P0 = blkdiag(P0, 1e-2*eye(m.nn_dim));
        m.ffun = @f_atrue;
        m.Q = blkdiag(Q, 1e-6*eye(m.nn_dim));
        m.y_pseudo = zeros(m.nn_dim, 1);
        m.y_pseudo_dim = m.nn_dim; 
        m.ay_dim = m.y_dim + m.y_pseudo_dim;
        m.hfun = @hfun_apbm;
        lambda_apbm = 0.05;
        m.R = blkdiag(R,(1/lambda_apbm)*eye(m.nn_dim));
        
        filters.model{i_filter} = m;
        filters.filter{i_filter} = atrue_initialize(m);
        clear m % to ensure nothing is preserved from previous model    
    end

end

% ╭───────────────────────────────────────────────────────────╮
% │                      Data Analysis                        │
% ╰───────────────────────────────────────────────────────────╯
% =====================================
% ============== MSE (dB) =============
% =====================================
save_x_mse = zeros(n_data, n_filter);
for i_filter = 1:n_filter
    for i_data = 1:n_data
        if error_type == "norm"
            save_x_mse(i_data, i_filter) = sum(save_x_error{i_filter}(i_data, :).^2)...
                / n_track;
        end
        if error_type == "element-wise"
            save_x_mse(i_data, i_filter) = sum(save_x_error{i_filter}(i_data, :))...
                / n_track;
        end
        if error_type == "first-element"
            save_x_mse(i_data, i_filter) = sum(save_x_error{i_filter}(i_data, :))...
                / n_track;
        end
    end
end

mse_dB = 10*log10(save_x_mse);

mse_mean = mean(save_x_mse, 1);
mse_std = std(save_x_mse, 0, 1);

mse_mean_dB = 10*log10(mse_mean);
mse_std_dB = 10*log10(mse_mean + mse_std) - mse_mean_dB;

mse_ratio = save_x_mse ./ repmat(save_x_mse(:,1), 1, n_filter);
mse_ratio_dB = 10*log10(mse_ratio);

fprintf("online unsupervised: MSE %.4f \n", mse_mean_dB(1))
fprintf("online unsupervised: STD %.4f \n", mse_std_dB(1))

%% Plot
fontsize=16;
set(groot,'DefaultLineLineWidth',2');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');
% =====================================
% ============== MSE (dB) =============
% =====================================
h_mse_dB = figure;
boxchart(mse_dB)
ax = gca; ax.FontSize = fontsize-2;
xticklabels(filters.name)
ylabel('MSE [dB]')
grid on

% =====================================
% =========== MSE (ratio dB) ==========
% =====================================
h_mse_ratio_dB = figure;
boxchart(mse_ratio_dB)
ax = gca; ax.FontSize = fontsize-2;
xticklabels(filters.name)
ylabel('MSE Ratio [dB]')
grid on

%% self-defined functions
% =====================================
% ========== True Model ==============
% =====================================
function x = f_true(x_k_1, delta_t, J)
    % Define B and C matrices
    B = [0 0 0; 0 0 -1; 0 1 0];
    C = [-10, 10, 0;
          28, -1, 0;
           0,  0, -8/3];

    Bx = [(B * x_k_1).';
        0, 0, 0;
        0, 0, 0]; 
    A = Bx.' + C;
    % Taylor Expansion for F
    F = eye(3);
    for j = 1:J
        F_add = (A * delta_t)^j / factorial(j);
        F = F + F_add;
    end
    
    % Compute the output
    x = F * x_k_1;
end

function [filter] = true_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = true_step(filter, delta_t, J, y, m)
  [xPred, pPred] = predict(filter, delta_t, J);
  [xCorr, pCorr] = correct(filter, y);
end

% =====================================
% ============== ATRUE =================
% =====================================
function x = f_atrue(x_k_1, delta_t, J, m)
    % extract x and NN parameter
    x_only = x_k_1(1:m.x_dim);
    theta = x_k_1(m.x_dim+1:end);
    m.NN.set_params(theta);

    % propagate x only
    B = [0 0 0; 0 0 -1; 0 1 0];
    C = [-10, 10, 0;
          28, -1, 0;
           0,  0, -8/3];

    Bx = [(B * x_only).';
        0, 0, 0;
        0, 0, 0]; 
    A = Bx.' + C;
    % Taylor Expansion for F
    F = eye(3);
    for j = 1:J
        F_add = (A * delta_t)^j / factorial(j);
        F = F + F_add;
    end
    % Compute the output with NN
    % x_only = F * x_only + m.NN.forward(x_only);

    % compute matrix G
    G = m.NN.forward(x_only);
    x_only = (F + reshape(G, [3, 3])) * x_only;

    % augment with NN parameter
    x = [x_only;theta];
end

function [filter] = atrue_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = atrue_step(filter, delta_t, J, y, m)
  [xPred, pPred] = predict(filter, delta_t, J, m);
  [xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m);
end

% =====================================
% ============== PBM ==================
% =====================================
function x = f_pbm(x_k_1, delta_t, J)
    % Define B and C matrices
    B = [0 0 0; 0 0 -1; 0 1 0];
    C = [-10, 10, 0;
          28, -1, 0;
           0,  0, -8/3];

    Bx = [(B * x_k_1).';
        0, 0, 0;
        0, 0, 0]; 
    A = Bx.' + C;
    % Taylor Expansion for F
    F = eye(3);
    for j = 1:J
        F_add = (A * delta_t)^j / factorial(j);
        F = F + F_add;
    end
    % set the first state as random walk
    F(1,:) = [1,0,0];
    % Compute the output
    x = F * x_k_1;
end

function [filter] = pbm_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = pbm_step(filter, delta_t, J, y, m)
  [xPred, pPred] = predict(filter, delta_t, J);
  [xCorr, pCorr] = correct(filter, y);
end

% =====================================
% ============== APBM =================
% =====================================
function x = f_apbm(x_k_1, delta_t, J, m)
    % extract x and NN parameter
    x_only = x_k_1(1:m.x_dim);
    theta = x_k_1(m.x_dim+1:end);
    m.NN.set_params(theta);

    % propagate x only
    B = [0 0 0; 0 0 -1; 0 1 0];
    C = [-10, 10, 0;
          28, -1, 0;
           0,  0, -8/3];

    Bx = [(B * x_only).';
        0, 0, 0;
        0, 0, 0]; 
    A = Bx.' + C;
    % Taylor Expansion for F
    F = eye(3);
    for j = 1:J
        F_add = (A * delta_t)^j / factorial(j);
        F = F + F_add;
    end
    % set the first state as random walk
    F(1,:) = [1,0,0];
    % Compute the output with NN
    x_only = F * x_only + m.NN.forward(x_only);

    % augment with NN parameter
    x = [x_only;theta];
end

function [filter] = apbm_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = apbm_step(filter, delta_t, J, y, m)
  [xPred, pPred] = predict(filter, delta_t, J, m);
  [xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m);
end

% =====================================
% ========= Measurement Model =========
% =====================================
function y = hfun(x)
    y = eye(3) * x;
end

function y = hfun_apbm(x, m)
    x_only = x(1:m.x_dim);
    theta = x(m.x_dim+1:end);
    y_only = eye(3) * x_only;
    y = [y_only; theta];
end

% =====================================
% ======== Filters Creation ===========
% =====================================
function [filters] = add_filter(filters,model,initialization,step,filterName)
  filters.model{end+1} = model;
  filters.filter{end+1} = initialization(model);
  filters.step{end+1} = step;
  filters.name{end+1} = filterName;
end