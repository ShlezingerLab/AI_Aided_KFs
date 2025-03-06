import torch
import torch.nn as nn
import time
from Smoothers.Linear_KF import KalmanFilter

def KFTest(args, SysModel, test_input, test_target, allStates=True, randomInit = False, test_init=None):

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(args.N_T)
    start = time.time()
    KF = KalmanFilter(SysModel)
    j=0

    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    for sequence_target,sequence_input in zip(test_target,test_input):
        if(randomInit):
            KF.InitSequence(torch.unsqueeze(test_init[j,:],1), SysModel.m2x_0)        
        else:
            KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
            
        KF.GenerateSequence(sequence_input, sequence_input.size()[-1])

        
        if(allStates):
            MSE_KF_linear_arr[j] = loss_fn(KF.x, sequence_target).item()
        else:
            MSE_KF_linear_arr[j] = loss_fn(KF.x[loc,:], sequence_target[loc,:]).item()
        #MSE_KF_linear_arr[j] = loss_fn(test_input[j, :, :], test_target[j, :, :]).item()
        j=j+1
    end = time.time()
    t = end - start
    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]



