import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import cubatureKalmanFilter
from tmlp import MLP
from tqdm import tqdm
import time
import sys


# Define the model
# ======= True Model ======= #
def F_true(x, delta_t, J):
    # x = torch.from_numpy(x).type(torch.float32)
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
    # x_out = torch.matmul(F, x)
    # x_out = x_out.numpy()
    return F


def f_APBM_batch(x, delta_t, J, nn_mlp):
    batch_size = x.size(0)
    # Ensure x requires gradients
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    # compute PBM F matrix
    Fs = []
    for i in range(batch_size):
        # Compute f_true for each sample individually
        F = F_true(x[i], delta_t, J)  # (1, 3) input
        Fs.append(F)
    # Combine the results into a single tensor
    F_tensor = torch.cat(Fs, dim=0)  # Shape will be (batch_size, 3, 3)
    F_tensor = F_tensor.reshape(batch_size, 3, 3)
    # compute G matrix
    Gs = []
    for i in range(batch_size):
        # Compute output of MLP
        G = nn_mlp(x[i]).reshape(3, 3)
        Gs.append(G)
    # Combine the results into a single tensor
    G_tensor = torch.cat(Gs, dim=0)  # Shape will be (batch_size, 3, 3)
    G_tensor = G_tensor.reshape(batch_size, 3, 3)

    # combine F and G
    x_batch = x.unsqueeze(2)
    x_out = torch.bmm(F_tensor + G_tensor, x_batch).squeeze(2)

    return x_out


def f_APBM_pretrained(x, delta_t, J, nn_mlp):
    # compute PBM F matrix
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

    # compute G matrix
    G = nn_mlp(x).reshape(3, 3)

    # combine F and G
    # x_out = torch.matmul(F, x)
    x_out = torch.matmul(F + G, x)
    x_out = x_out.detach().numpy()
    return x_out


def h_true(x):
    H = np.eye(3)
    y = H @ x
    return y


if __name__ == '__main__':

    # Load data
    DatafileName = 'data/decimated_r0_Ttest3000.pt'
    [train_y, train_x, cv_input_long, cv_target_long, test_input, test_target] = torch.load(DatafileName)

    # Prepare the training data
    train_target = train_x[:, :, 1:]  # Target is x_k (k = 1 to 3000)
    train_input = train_x[:, :, :-1]  # Input is x_{k-1} (k = 0 to 2999)

    # parameter settings
    delta_t = 0.02
    J = 2
    train_flag = False

    if train_flag:
        # Initialize the MLP (G(x_k))
        input_size = 3  # x_k is a 3-element vector
        hidden_sizes = [16, 16]  # Example hidden layer sizes
        output_size = 9  # G(x_k) outputs 9 values to reshape to a 3x3 matrix
        mlp_model = MLP(input_size, hidden_sizes, output_size)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp_model.parameters(), lr=0.00001)
        # Learning rate scheduler: Decay by factor of 0.1 every 80 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        # Training process
        epochs = 360  # Number of epochs
        batch_size = 64  # Batch size

        for epoch in range(epochs):
            for i in range(0, train_input.size(2), batch_size):
                optimizer.zero_grad()  # Zero gradients
                # # Get batch data
                # input_batch = train_input[i:i+batch_size]
                # target_batch = train_target[i:i+batch_size]
                #
                # # Forward pass: compute predictions
                # outputs = []
                # for j in range(input_batch.size(2)):  # Iterate over time steps
                #     x_k_minus_1 = input_batch[:, :, j]
                #
                #     # Compute (F + G) * x_{k_-1}
                #     x_k = f_APBM_batch(x_k_minus_1, delta_t, J, mlp_model)
                #     outputs.append(x_k)
                # outputs = torch.stack(outputs, dim=2)  # Stack to form 3x(time_steps) output
                # # Compute loss
                # loss = criterion(outputs, target_batch)

                # test
                # x_true = train_x[0, :, :]
                # x_pred_test = f_APBM_batch(x_true[:, :-1].T, delta_t, J, mlp_model).T
                # loss_test = criterion(x_pred_test, x_true[:, 1:])
                # print(f'Loss Test: {loss_test.item():.4f}')

                # Get batch data
                x_k_minus_1 = train_input[0, :, i:i + batch_size].T
                target_batch = train_target[0, :, i:i + batch_size].T
                # x_k_minus_1 = x_k_minus_1.reshape(x_k_minus_1.size(1), 3)
                # target_batch = target_batch.reshape(target_batch.size(1), 3)

                # Compute (F + G) * x_{k_-1}
                x_k = f_APBM_batch(x_k_minus_1, delta_t, J, mlp_model)
                # print(x_k)

                # Compute loss
                loss = criterion(x_k, target_batch)

                # optimize
                loss.backward()
                optimizer.step()

            # Step the scheduler to decay the learning rate
            scheduler.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        nn_parameter = mlp_model.get_parameters()
        torch.save(nn_parameter, 'data/mlp_parameter.pt')

    # Load NN data
    if not train_flag:
        NNfileName = 'data/mlp_parameter.pt'
        nn_parameter = torch.load(NNfileName)
        # Initialize the MLP (G(x_k))
        input_size = 3  # x_k is a 3-element vector
        hidden_sizes = [32, 32, 32]  # Example hidden layer sizes
        output_size = 9  # G(x_k) outputs 9 values to reshape to a 3x3 matrix
        mlp_model = MLP(input_size, hidden_sizes, output_size)
        mlp_model.set_parameters(nn_parameter)
        # save it to .csv
        nn_parameter_array = nn_parameter.detach().numpy()
        pd.DataFrame(nn_parameter_array).to_csv('data/nn_parameter.csv', index=False, header=False)

    # ===================================== #
    # ======= CKF APBM (pre-trained) ====== #
    # ===================================== #
    # filering settings
    nDataset = 3  # test_input.shape[0]
    nEpoch = test_input.shape[2]
    x_dim = test_target.shape[1]
    y_dim = test_input.shape[1]

    # x_0 = np.ones(x_dim)
    x_0 = test_target[0, :, 0].numpy()
    P_0 = 1e-5 ** 2 * np.eye(x_dim)  # np.zeros((x_dim, x_dim))
    q = 0.26
    Q = q ** 2 * np.eye(x_dim)
    r = 1.0
    R = r ** 2 * np.eye(y_dim)
    # memory allocation
    mse_CKF_APBM_offline_datasets = torch.empty(nDataset)
    for iDataset in tqdm(range(nDataset), desc='CKF APBM offline Processing: '):
        x_true = train_x[iDataset, :, :]
        y_measurement = train_y[iDataset, :, :]
        x_test = test_target[iDataset, :, :]
        y_test = test_input[iDataset, :, :]
        # memory allocation
        x_est_CKF = np.zeros((x_dim, nEpoch))
        x_est_CKF[:, 0] = x_0
        P_est_CKF = np.zeros((x_dim, x_dim, nEpoch))
        # initialize CKF
        ckf_apbm_offline = cubatureKalmanFilter.CubatureKalmanFilter(x_dim, y_dim, delta_t, hx=h_true, fx=f_APBM_pretrained)
        ckf_apbm_offline.x = x_0
        ckf_apbm_offline.P = P_0
        ckf_apbm_offline.Q = Q
        ckf_apbm_offline.R = R
        loss_fn = nn.MSELoss(reduction='mean')
        # filtering
        for iEpoch in range(1, nEpoch):
            y = y_test[:, iEpoch].numpy()

            ckf_apbm_offline.predict(delta_t, fx_args=(J, mlp_model))
            ckf_apbm_offline.update(np.reshape(y, (-1, 1)))
            x_est_CKF[:, iEpoch] = np.squeeze(ckf_apbm_offline.x)
            P_est_CKF[:, :, iEpoch] = ckf_apbm_offline.P
        # compute error
        mse_CKF_APBM_offline_datasets[iDataset] = loss_fn(torch.from_numpy(x_est_CKF).type(torch.float32), x_test)

        # mapping
        # x_pred = torch.zeros((x_dim, nEpoch - 1))
        # for iEpoch in range(1, nEpoch):
        #     x_k_1 = x_true[:, iEpoch - 1]
        #     x_k_pred = f_APBM_pretrained(x_k_1.numpy(), delta_t, J, mlp_model)
        #     x_pred[:, iEpoch - 1] = torch.from_numpy(x_k_pred).type(torch.float32)
        # mse_CKF_APBM_offline_datasets[iDataset] = loss_fn(x_pred, x_true[:, 1:])

        # batch mapping
        # x_pred = f_APBM_batch(x_true[:, :-1].T, delta_t, J, mlp_model).T
        # mse_CKF_APBM_offline_datasets[iDataset] = loss_fn(x_pred, x_true[:, 1:])


    mse_CKF_APBM_offline = torch.mean(mse_CKF_APBM_offline_datasets)
    mse_CKF_APBM_offline_dB = 10 * torch.log10(mse_CKF_APBM_offline)

    mse_CKF_APBM_offline_std = torch.std(mse_CKF_APBM_offline_datasets, unbiased=True)
    mse_CKF_APBM_offline_std_dB = 10 * torch.log10(mse_CKF_APBM_offline_std + mse_CKF_APBM_offline) - mse_CKF_APBM_offline_dB

    print("CKF True - MSE:", mse_CKF_APBM_offline_dB, "[dB]")
    print("CKF True - SE STD:", mse_CKF_APBM_offline_std_dB, "[dB]")
    time.sleep(0.1)




    sys.exit()



