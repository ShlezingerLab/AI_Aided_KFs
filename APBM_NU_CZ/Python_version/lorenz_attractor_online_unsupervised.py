import numpy as np
import torch
import torch.nn as nn
import math
import cubatureKalmanFilter
from tmlp import MLP
from tqdm import tqdm
import pandas as pd
import time
import sys


# Define the model
# ======= True Model ======= #
def f_true(x, delta_t, J):
    x = torch.from_numpy(x).type(torch.float32)
    B = torch.tensor([[[0, 0, 0], [0, 0, -1], [0, 1, 0]], torch.zeros(3, 3), torch.zeros(3, 3)]).float()
    C = torch.tensor([[-10, 10, 0],
                      [28, -1, 0],
                      [0, 0, -8 / 3]]).float()
    BX = torch.reshape(torch.matmul(B, x), (3, 3))
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)), C))

    # Taylor Expansion for F
    F = torch.eye(3)
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j))
        F = torch.add(F, F_add)
    x_out = torch.matmul(F, x)
    x_out = x_out.numpy()
    return x_out


def h_true(x):
    H = np.eye(3)
    y = H @ x
    return y


# ======= APBM true ======= #
def f_APBM_true(x, delta_t, J, nn_mlp):
    # extract x from whole state
    n_theta = nn_mlp.get_num_parameters()
    x_only = x[:-n_theta]
    x_only = torch.from_numpy(x_only).type(torch.float32)
    theta = x[-n_theta:]
    theta = torch.from_numpy(theta).type(torch.float32)
    nn_mlp.set_parameters(theta)

    B = torch.tensor([[[0, 0, 0], [0, 0, -1], [0, 1, 0]], torch.zeros(3, 3), torch.zeros(3, 3)]).float()
    C = torch.tensor([[-10, 10, 0],
                      [28, -1, 0],
                      [0, 0, -8 / 3]]).float()
    BX = torch.reshape(torch.matmul(B, x_only), (3, 3))
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)), C))

    # Taylor Expansion for F
    F = torch.eye(3)
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j))
        F = torch.add(F, F_add)

    G = nn_mlp.forward(x_only).reshape(3, 3)
    x_out = torch.matmul(F + G, x_only)
    # x_out = torch.matmul(F, x_only) + nn_mlp.forward(x_only)
    x_out = x_out.detach().numpy()
    x_out = np.concatenate((x_out, theta))
    return x_out


def h_APBM_true(x, nn_mlp):
    n_theta = nn_mlp.get_num_parameters()
    x_only = x[:-n_theta]
    theta = x[-n_theta:]
    H = np.eye(3)
    y_only = H @ x_only
    y = np.concatenate((y_only, theta))
    return y


# ======= PBM ======= #
def f_PBM(x, delta_t, J):
    x = torch.from_numpy(x).type(torch.float32)
    B = torch.tensor([[[0, 0, 0], [0, 0, -1], [0, 1, 0]], torch.zeros(3, 3), torch.zeros(3, 3)]).float()
    C = torch.tensor([[-10, 10, 0],
                      [28, -1, 0],
                      [0, 0, -8 / 3]]).float()
    BX = torch.reshape(torch.matmul(B, x), (3, 3))
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)), C))

    # Taylor Expansion for F
    F = torch.eye(3)
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j))
        F = torch.add(F, F_add)
    F[0, :] = torch.tensor([1.0, 0.0, 0.0])
    x_out = torch.matmul(F, x)
    x_out = x_out.numpy()
    return x_out


def h_PBM(x):
    H = np.eye(3)
    y = H @ x
    return y


# ======= APBM ======= #
def f_APBM(x, delta_t, J, nn_mlp):
    # extract x from whole state
    n_theta = nn_mlp.get_num_parameters()
    x_only = x[:-n_theta]
    x_only = torch.from_numpy(x_only).type(torch.float32)
    theta = x[-n_theta:]
    theta = torch.from_numpy(theta).type(torch.float32)
    nn_mlp.set_parameters(theta)

    B = torch.tensor([[[0, 0, 0], [0, 0, -1], [0, 1, 0]], torch.zeros(3, 3), torch.zeros(3, 3)]).float()
    C = torch.tensor([[-10, 10, 0],
                      [28, -1, 0],
                      [0, 0, -8 / 3]]).float()
    BX = torch.reshape(torch.matmul(B, x_only), (3, 3))
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)), C))

    # Taylor Expansion for F
    F = torch.eye(3)
    for j in range(1, J + 1):
        F_add = (torch.matrix_power(A * delta_t, j) / math.factorial(j))
        F = torch.add(F, F_add)
    F[0, :] = torch.tensor([1.0, 0.0, 0.0])
    x_out = torch.matmul(F, x_only) + nn_mlp.forward(x_only)
    x_out = x_out.detach().numpy()
    x_out = np.concatenate((x_out, theta))
    return x_out


def h_APBM(x, nn_mlp):
    n_theta = nn_mlp.get_num_parameters()
    x_only = x[:-n_theta]
    theta = x[-n_theta:]
    H = np.eye(3)
    y_only = H @ x_only
    y = np.concatenate((y_only, theta))
    return y


if __name__ == '__main__':
    # Load data
    DatafileName = 'data/decimated_r0_Ttest3000.pt'
    [_, _, _, _, test_input, test_target] = torch.load(DatafileName)
    # common settings
    nDataset = 10  # test_input.shape[0]
    nEpoch = test_input.shape[2]
    x_dim = test_target.shape[1]
    y_dim = test_input.shape[1]
    delta_t = 0.02
    J = 2

    x_0 = np.ones(x_dim)
    P_0 = 1e-5 ** 2 * np.eye(x_dim)  # np.zeros((x_dim, x_dim))
    q = 0.3
    Q = q ** 2 * np.eye(x_dim)
    r = 1.0
    R = r ** 2 * np.eye(y_dim)

    # ===================================== #
    # ============ CKF Ture =============== #
    # ===================================== #
    # memory allocation
    mse_CKF_true_datasets = torch.empty(nDataset)
    # for iDataset in tqdm(range(nDataset), desc='CKF True Processing: '):
    #     x_true = test_target[iDataset, :, :]
    #     y_measurement = test_input[iDataset, :, :]
    #     # memory allocation
    #     x_est_CKF = np.zeros((x_dim, nEpoch))
    #     P_est_CKF = np.zeros((x_dim, x_dim, nEpoch))
    #     # initialize CKF
    #     ckf_true = cubatureKalmanFilter.CubatureKalmanFilter(x_dim, y_dim, delta_t, hx=h_true, fx=f_true)
    #     ckf_true.x = x_0
    #     ckf_true.P = P_0
    #     ckf_true.Q = Q
    #     ckf_true.R = R
    #     loss_fn = nn.MSELoss(reduction='mean')
    #     # filtering
    #     for iEpoch in range(nEpoch):
    #         y = y_measurement[:, iEpoch].numpy()
    #
    #         ckf_true.predict(delta_t, J)
    #         ckf_true.update(np.reshape(y, (-1, 1)))
    #         x_est_CKF[:, iEpoch] = np.squeeze(ckf_true.x)
    #         P_est_CKF[:, :, iEpoch] = ckf_true.P
    #
    #     # compute error
    #     mse_CKF_true_datasets[iDataset] = loss_fn(torch.from_numpy(x_est_CKF).type(torch.float32),
    #                                               test_target[iDataset, :, :])
    #
    # mse_CKF_true = torch.mean(mse_CKF_true_datasets)
    # mse_CKF_true_dB = 10 * torch.log10(mse_CKF_true)
    #
    # mse_CKF_true_std = torch.std(mse_CKF_true_datasets, unbiased=True)
    # mse_CKF_true_std_dB = 10 * torch.log10(mse_CKF_true_std + mse_CKF_true) - mse_CKF_true_dB
    #
    # print("CKF True - MSE:", mse_CKF_true_dB, "[dB]")
    # print("CKF True - SE STD:", mse_CKF_true_std_dB, "[dB]")
    # time.sleep(0.1)

    # ========================================== #
    # ============ CKF APBM True =============== #
    # ========================================== #
    # memory allocation
    mse_CKF_APBM_true_datasets = torch.empty(nDataset)
    for iDataset in tqdm(range(nDataset), desc='CKF APBM True Processing: '):
        x_true = test_target[iDataset, :, :]
        y_measurement = test_input[iDataset, :, :]
        # memory allocation
        x_est_CKF = np.zeros((x_dim, nEpoch))
        P_est_CKF = np.zeros((x_dim, x_dim, nEpoch))
        # initialize CKF
        mlp_input_size = 3
        mlp_output_size = 3 * 3
        mlp_hidden_sizes = [16, 16]
        mlp_APBM_true = MLP(mlp_input_size, mlp_hidden_sizes, mlp_output_size)
        n_params_mlp = mlp_APBM_true.get_num_parameters()
        theta = mlp_APBM_true.get_parameters()
        ax_dim = x_dim + n_params_mlp
        ay_dim = y_dim + n_params_mlp

        ckf_APBM_true = cubatureKalmanFilter.CubatureKalmanFilter(ax_dim, ay_dim, delta_t, hx=h_APBM_true, fx=f_APBM_true)
        ckf_APBM_true.x = np.concatenate((x_0, theta.detach().numpy()))
        P_0_nn = 1e-2 * np.eye(n_params_mlp)
        ckf_APBM_true.P = np.block([[P_0, np.zeros((x_dim, n_params_mlp))], [np.zeros((n_params_mlp, x_dim)), P_0_nn]])
        Q_nn = 1e-8 * np.eye(n_params_mlp)
        ckf_APBM_true.Q = np.block([[Q, np.zeros((x_dim, n_params_mlp))], [np.zeros((n_params_mlp, x_dim)), Q_nn]])
        lambda_APBM_true = 5
        pseudo_R = 1 / lambda_APBM_true * np.eye(n_params_mlp)
        ckf_APBM_true.R = np.block([[R, np.zeros((y_dim, n_params_mlp))], [np.zeros((n_params_mlp, y_dim)), pseudo_R]])
        loss_fn = nn.MSELoss(reduction='mean')
        pseudo_y = np.zeros(n_params_mlp)
        # filtering
        for iEpoch in range(nEpoch):
            y_only = y_measurement[:, iEpoch].numpy()
            y = np.concatenate((y_only, pseudo_y))

            ckf_APBM_true.predict(delta_t, fx_args=(J, mlp_APBM_true))
            ckf_APBM_true.update(np.reshape(y, (-1, 1)), hx_args=mlp_APBM_true)
            x_est_CKF[:, iEpoch] = np.squeeze(ckf_APBM_true.x[:3])
            P_est_CKF[:, :, iEpoch] = ckf_APBM_true.P[:3, :3]

        # compute error
        mse_CKF_APBM_true_datasets[iDataset] = loss_fn(torch.from_numpy(x_est_CKF).type(torch.float32),
                                                  test_target[iDataset, :, :])

    mse_CKF_APBM_true = torch.mean(mse_CKF_APBM_true_datasets)
    mse_CKF_APBM_true_dB = 10 * torch.log10(mse_CKF_APBM_true)

    mse_CKF_APBM_true_std = torch.std(mse_CKF_APBM_true_datasets, unbiased=True)
    mse_CKF_APBM_true_std_dB = 10 * torch.log10(mse_CKF_APBM_true_std + mse_CKF_APBM_true) - mse_CKF_APBM_true_dB

    print("CKF APBM_true - MSE:", mse_CKF_APBM_true_dB, "[dB]")
    print("CKF APBM_true - SE STD:", mse_CKF_APBM_true_std_dB, "[dB]")
    time.sleep(0.1)


    # ===================================== #
    # ============ CKF PBM =============== #
    # ===================================== #
    # memory allocation
    mse_CKF_PBM_datasets = torch.empty(nDataset)
    # for iDataset in tqdm(range(nDataset), desc='CKF PBM Processing: '):
    #     x_true = test_target[iDataset, :, :]
    #     y_measurement = test_input[iDataset, :, :]
    #     # memory allocation
    #     x_est_CKF = np.zeros((x_dim, nEpoch))
    #     P_est_CKF = np.zeros((x_dim, x_dim, nEpoch))
    #     # initialize CKF
    #     ckf_PBM = cubatureKalmanFilter.CubatureKalmanFilter(x_dim, y_dim, delta_t, hx=h_PBM, fx=f_PBM)
    #     ckf_PBM.x = x_0
    #     ckf_PBM.P = P_0
    #     ckf_PBM.Q = Q
    #     ckf_PBM.R = R
    #     loss_fn = nn.MSELoss(reduction='mean')
    #     # filtering
    #     for iEpoch in range(nEpoch):
    #         y = y_measurement[:, iEpoch].numpy()
    #
    #         ckf_PBM.predict(delta_t, J)
    #         ckf_PBM.update(np.reshape(y, (-1, 1)))
    #         x_est_CKF[:, iEpoch] = np.squeeze(ckf_PBM.x)
    #         P_est_CKF[:, :, iEpoch] = ckf_PBM.P
    #
    #     # compute error
    #     mse_CKF_PBM_datasets[iDataset] = loss_fn(torch.from_numpy(x_est_CKF).type(torch.float32),
    #                                              test_target[iDataset, :, :])
    #
    # mse_CKF_PBM = torch.mean(mse_CKF_PBM_datasets)
    # mse_CKF_PBM_dB = 10 * torch.log10(mse_CKF_PBM)
    #
    # mse_CKF_PBM_std = torch.std(mse_CKF_PBM_datasets, unbiased=True)
    # mse_CKF_PBM_std_dB = 10 * torch.log10(mse_CKF_PBM_std + mse_CKF_PBM) - mse_CKF_PBM_dB
    #
    # print("CKF PBM - MSE:", mse_CKF_PBM_dB, "[dB]")
    # print("CKF PBM - SE STD:", mse_CKF_PBM_std_dB, "[dB]")
    # time.sleep(0.1)

    # ===================================== #
    # ============ CKF APBM =============== #
    # ===================================== #
    # memory allocation
    mse_CKF_APBM_datasets = torch.empty(nDataset)
    # for iDataset in tqdm(range(nDataset), desc='CKF APBM Processing: '):
    #     x_true = test_target[iDataset, :, :]
    #     y_measurement = test_input[iDataset, :, :]
    #     # memory allocation
    #     x_est_CKF = np.zeros((x_dim, nEpoch))
    #     P_est_CKF = np.zeros((x_dim, x_dim, nEpoch))
    #     # initialize CKF
    #     mlp_input_size = 3
    #     mlp_output_size = 3
    #     mlp_hidden_sizes = [5]
    #     mlp_apbm = MLP(mlp_input_size, mlp_hidden_sizes, mlp_output_size)
    #     n_params_mlp = mlp_apbm.get_num_parameters()
    #     theta = mlp_apbm.get_parameters()
    #     ax_dim = x_dim + n_params_mlp
    #     ay_dim = y_dim + n_params_mlp
    #
    #     ckf_APBM = cubatureKalmanFilter.CubatureKalmanFilter(ax_dim, ay_dim, delta_t, hx=h_APBM, fx=f_APBM)
    #     ckf_APBM.x = np.concatenate((x_0, theta.detach().numpy()))
    #     P_0_nn = 1e-2 * np.eye(n_params_mlp)
    #     ckf_APBM.P = np.block([[P_0, np.zeros((x_dim, n_params_mlp))], [np.zeros((n_params_mlp, x_dim)), P_0_nn]])
    #     Q_nn = 1e-8 * np.eye(n_params_mlp)
    #     ckf_APBM.Q = np.block([[Q, np.zeros((x_dim, n_params_mlp))], [np.zeros((n_params_mlp, x_dim)), Q_nn]])
    #     lambda_apbm = 0.5
    #     pseudo_R = 1 / lambda_apbm * np.eye(n_params_mlp)
    #     ckf_APBM.R = np.block([[R, np.zeros((y_dim, n_params_mlp))], [np.zeros((n_params_mlp, y_dim)), pseudo_R]])
    #     loss_fn = nn.MSELoss(reduction='mean')
    #     pseudo_y = np.zeros(n_params_mlp)
    #     # filtering
    #     for iEpoch in range(nEpoch):
    #         y_only = y_measurement[:, iEpoch].numpy()
    #         y = np.concatenate((y_only, pseudo_y))
    #
    #         ckf_APBM.predict(delta_t, fx_args=(J, mlp_apbm))
    #         ckf_APBM.update(np.reshape(y, (-1, 1)), hx_args=mlp_apbm)
    #         x_est_CKF[:, iEpoch] = np.squeeze(ckf_APBM.x[:3])
    #         P_est_CKF[:, :, iEpoch] = ckf_APBM.P[:3, :3]
    #
    #     # compute error
    #     mse_CKF_APBM_datasets[iDataset] = loss_fn(torch.from_numpy(x_est_CKF).type(torch.float32),
    #                                               test_target[iDataset, :, :])
    #
    # mse_CKF_APBM = torch.mean(mse_CKF_APBM_datasets)
    # mse_CKF_APBM_dB = 10 * torch.log10(mse_CKF_APBM)
    #
    # mse_CKF_APBM_std = torch.std(mse_CKF_APBM_datasets, unbiased=True)
    # mse_CKF_APBM_std_dB = 10 * torch.log10(mse_CKF_APBM_std + mse_CKF_APBM) - mse_CKF_APBM_dB
    #
    # print("CKF APBM - MSE:", mse_CKF_APBM_dB, "[dB]")
    # print("CKF APBM - SE STD:", mse_CKF_APBM_std_dB, "[dB]")

    # ===================================== #
    # ========== Save Results ============= #
    # ===================================== #
    # save in a data frame
    data_mse = {
        'mse_CKF_true': mse_CKF_true_datasets.numpy(),
        'mse_CKF_APBM_true': mse_CKF_APBM_true_datasets.numpy(),
        'mse_CKF_PBM': mse_CKF_PBM_datasets.numpy(),
        'mse_CKF_APBM': mse_CKF_APBM_datasets.numpy()
    }
    df_mse = pd.DataFrame(data_mse)
    df_mse.to_csv('data/mse_results.csv', index=False)
