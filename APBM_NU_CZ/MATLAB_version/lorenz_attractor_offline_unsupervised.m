clc
clear
close all

% ╭───────────────────────────────────────────────────────────╮
% │               load data                                   │
% ╰───────────────────────────────────────────────────────────╯
% Preallocate a 3D array to store the train data
% Read the CSV file into a matrix
file_name = 'data/test_x_1.csv';
x_data_train = readmatrix(file_name);
y_data_train = zeros(10, 3, 3000);
% Loop over the 10 files
n_data_train = 100;
for i_data = 1:n_data_train
    file_name = ['data/train_y_', num2str(i_data), '.csv'];
    tempData = readmatrix(file_name);
    y_data_train(i_data, :, :) = tempData;
end

% Preallocate a 3D array to store the test data
x_data_test = zeros(10, 3, 3000);
y_data_test = zeros(10, 3, 3000);
% Loop over the 10 files
n_data_test = 10;
for i_data = 1:n_data_test
    % Read the CSV file into a matrix
    file_name = ['data/test_x_', num2str(i_data), '.csv'];
    tempData = readmatrix(file_name);
    x_data_test(i_data, :, :) = tempData;

    file_name = ['data/test_y_', num2str(i_data), '.csv'];
    tempData = readmatrix(file_name);
    y_data_test(i_data, :, :) = tempData;
end

% ╭───────────────────────────────────────────────────────────╮
% │               common system settings                      │
% ╰───────────────────────────────────────────────────────────╯
x_dim = 3;
y_dim = 3;
n_track = 3000;
delta_t = 0.02; % sampling rate
J = 2; % order of Taylor Expansion

q = 0.26;
Q = q^2 * eye(x_dim);
r = 1;
R = r^2 * eye(y_dim);

x0 = x_data_train(:,1);
% x0 = [1; 1; 1];
P0 = zeros(3,3);

error_type = "first-element"; %"norm"; "element-wise"

train_flag = 1;

% ╭───────────────────────────────────────────────────────────╮
% │                    filter settings                        │
% ╰───────────────────────────────────────────────────────────╯
filters.model = {};
filters.filter = {};
filters.step = {};
filters.name = {};

% =====================================
% ============== CKF ATRUE ==============
% =====================================
m.x_dim = x_dim; % state dimension
m.y_dim = y_dim; % measurement dimension
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


% ╭───────────────────────────────────────────────────────────╮
% │                      Training                             │
% ╰───────────────────────────────────────────────────────────╯
if train_flag == 1
n_filter = length(filters.filter);
% memory allocation
save_x_error = cell(1, n_filter);
for i_filter = 1:n_filter
    save_x_error{i_filter} = zeros(n_data_train, n_track);
end

for i_data = 1:n_data_train % loop for datasets
    i_data
    % memory allocation
    save_xhat = cell(1,n_filter);
    for i_filter = 1:n_filter
        save_xhat{i_filter} = zeros(x_dim,n_track);
    end
    % extract true state
    x = x_data_train;
    x = reshape(x, x_dim, n_track);
    % filtering 
    for i_track = 1:n_track % loop for time steps
        for i_filter = 1:n_filter % loop for filters
            % extract measurement
            y = y_data_train(i_data, :, i_track)';
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
    fprintf("Error during training (%d/%d): %.4f   \n",...
        i_data, n_data_train, sum(save_x_error{i_filter}(i_data, :))/n_track);

    % reset state
    theta = filters.filter{1}.State(x_dim + 1:end);
    for i_filter = 1:n_filter
        m.x_dim = x_dim; % state dimension
        m.y_dim = y_dim; % measurement dimension
        m.NN = tmlp(m.x_dim, 9, [5 5]);
        m.NN.set_params(theta); % getting NN parameters
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

% save training result

save('data/nn_parameter_offline_unsupervised.mat', "theta")
end

% ╭───────────────────────────────────────────────────────────╮
% │                           Test                            │
% ╰───────────────────────────────────────────────────────────╯
if train_flag == 0
    theta_trained = load('data/nn_parameter_offline_unsupervised.mat');
    theta_trained = theta_trained.theta;
else
    theta_trained = theta;
end
filters.model = {};
filters.filter = {};
filters.step = {};
filters.name = {};
% =====================================
% ===== APBM pretrained (fixed) =======
% =====================================
m.x_dim = x_dim; % state dimension
m.y_dim = y_dim; % measurement dimension
m.NN = tmlp(m.x_dim, 9, [5 5]);
m.NN.set_params(theta_trained);
m.x0 = x0;
m.P0 = P0;
m.ffun = @f_apbm_pretrained_fixed;
m.Q = Q;
m.hfun = @hfun;
m.R = R;

% comment the following line to disable the filter
filters = add_filter(filters,m,@apbm_pre_fixed_initialize,@apbm_pre_fixed_step,'APBM Pretrained Fixed');
clear m % to ensure nothing is preserved from previous model

% =====================================
% ===== APBM pretrained (evolving) ====
% =====================================
m.x_dim = x_dim; % state dimension
m.y_dim = y_dim; % measurement dimension
m.NN = tmlp(m.x_dim, 9, [5 5]);
m.NN.set_params(theta_trained); % getting NN parameters
m.nn_dim = length(theta_trained); % number of NN parameter
m.ax_dim = m.x_dim + m.nn_dim; % augmented state dimension

m.x0 = [x0; theta_trained];
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

% memory allocation
n_filter = length(filters.filter);
save_x_error = cell(1, n_filter);
for i_filter = 1:n_filter
    save_x_error{i_filter} = zeros(n_data_test, n_track);
end

for i_data = 1:n_data_test % loop for datasets
    % memory allocation
    save_xhat = cell(1,n_filter);
    for i_filter = 1:n_filter
        save_xhat{i_filter} = zeros(x_dim,n_track);
    end
    % extract true state
    x = x_data_train;
    x = reshape(x, x_dim, n_track);
    % filtering 
    for i_track = 1:n_track % loop for time steps
        for i_filter = 1:n_filter % loop for filters
            % extract measurement
            y = y_data_train(i_data, :, i_track)';
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
        if i_filter == 1
            m.x_dim = x_dim; % state dimension
            m.y_dim = y_dim; % measurement dimension
            m.NN = tmlp(m.x_dim, 9, [5 5]);
            m.NN.set_params(theta_trained);
            m.x0 = x0;
            m.P0 = P0;
            m.ffun = @f_apbm_pretrained_fixed;
            m.Q = Q;
            m.hfun = @hfun;
            m.R = R;

            % comment the following line to disable the filter
            filters = add_filter(filters,m,@apbm_pre_fixed_initialize,@apbm_pre_fixed_step,'APBM Pretrained Fixed');
            clear m % to ensure nothing is preserved from previous model
        end
        if i_filter == 2
            m.x_dim = x_dim; % state dimension
            m.y_dim = y_dim; % measurement dimension
            m.NN = tmlp(m.x_dim, 9, [5 5]);
            m.NN.set_params(theta_trained); % getting NN parameters
            m.nn_dim = length(theta_trained); % number of NN parameter
            m.ax_dim = m.x_dim + m.nn_dim; % augmented state dimension
            
            m.x0 = [x0; theta_trained];
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
end

% ╭───────────────────────────────────────────────────────────╮
% │                      Data Analysis                        │
% ╰───────────────────────────────────────────────────────────╯
% =====================================
% ============== MSE (dB) =============
% =====================================
save_x_mse = zeros(n_data_test, n_filter);
for i_filter = 1:n_filter
    for i_data = 1:n_data_test
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

fprintf("offline unsupervised (APBM fixed in test): MSE %.4f \n", mse_mean_dB(1))
fprintf("offline unsupervised (APBM fixed in test): STD %.4f \n", mse_std_dB(1))
fprintf("offline unsupervised (APBM evolving in test): MSE %.4f \n", mse_mean_dB(2))
fprintf("offline unsupervised (APBM evolving in test): STD %.4f \n", mse_std_dB(2))

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
% ===== APBM pretrained (fixed) =======
% =====================================
function x = f_apbm_pretrained_fixed(x_k_1, delta_t, J, m)
    x = x_k_1(1:m.x_dim);

    % compute matrix F
    B = [0 0 0; 0 0 -1; 0 1 0];
    C = [-10, 10, 0;
          28, -1, 0;
           0,  0, -8/3];

    Bx = [(B * x).';
        0, 0, 0;
        0, 0, 0]; 
    A = Bx.' + C;
    % Taylor Expansion for F
    F = eye(3);
    for j = 1:J
        F_add = (A * delta_t)^j / factorial(j);
        F = F + F_add;
    end

    % compute matrix G
    G = m.NN.forward(x);
    x = (F + reshape(G, [3, 3])) * x;

end

function [filter] = apbm_pre_fixed_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = apbm_pre_fixed_step(filter, delta_t, J, y, m)
  [xPred, pPred] = predict(filter, delta_t, J, m);
  [xCorr, pCorr] = correct(filter, y);
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