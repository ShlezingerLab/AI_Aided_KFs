import torch.nn as nn
import torch
import time
from Smoothers.EKF_NotBatched import ExtendedKalmanFilter


def EKFTest(args, SysModel, test_input, test_target, allStates=True):
    # Number of test samples
    N_T = test_target.size()[0]
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean') 
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.zeros(N_T)
    # Allocate empty tensor for output
    EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]]) # N_T x m x T
    KG_array = torch.zeros([N_T, test_input.size()[2], SysModel.m, SysModel.n]) # N_T x T x m x n 
    # mask on state
    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    start = time.time()

    EKF = ExtendedKalmanFilter(SysModel, args)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)  
    for j in range(0, N_T):
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)

        if(allStates):
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        else:
            MSE_EKF_linear_arr[j] = loss_fn(EKF.x[loc,:], test_target[j, :, :]).item()
        KG_array[j] = EKF.KG_array
        EKF_out[j,:,:] = EKF.x

    end = time.time()
    t = end - start

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_linear_std = torch.std(MSE_EKF_linear_arr, unbiased=True)

    # Confidence interval
    EKF_std_dB = 10 * torch.log10(MSE_EKF_linear_std + MSE_EKF_linear_avg) - MSE_EKF_dB_avg
    
    print("Extended Kalman Filter - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("Extended Kalman Filter - STD:", EKF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]



