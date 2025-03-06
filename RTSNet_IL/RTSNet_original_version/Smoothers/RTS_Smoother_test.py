import torch
import torch.nn as nn
import time
from Smoothers.Linear_KF import KalmanFilter
from Smoothers.RTS_Smoother import rts_smoother

def S_Test(SysModel, test_input, test_target, allStates=True, randomInit = False,test_init=None):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    N_T = len(test_input)
    MSE_RTS_linear_arr = torch.empty(N_T)
    start = time.time()
    KF = KalmanFilter(SysModel)
    RTS = rts_smoother(SysModel)
    RTS_out = [] # allocate for saving output

    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    j=0
    # mask = torch.tensor([True,True,True,False,False,False])# for kitti

    for sequence_target,sequence_input in zip(test_target,test_input):
        if(randomInit):
            KF.InitSequence(torch.unsqueeze(test_init[j,:],1), SysModel.m2x_0)  
        else:
            KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)   
            
        KF.GenerateSequence(sequence_input, sequence_input.size()[-1])
        RTS.GenerateSequence(KF.x, KF.sigma, sequence_input.size()[-1])
        
        if(allStates):
            MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x, sequence_target).item()
        else:           
            MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x[loc,:], sequence_target[loc,:]).item()
        RTS_out.append(RTS.s_x)      
        j=j+1
    end = time.time()
    t = end - start

    # Average
    MSE_RTS_linear_avg = torch.mean(MSE_RTS_linear_arr)
    MSE_RTS_dB_avg = 10 * torch.log10(MSE_RTS_linear_avg)

    # Standard deviation
    MSE_RTS_linear_std = torch.std(MSE_RTS_linear_arr, unbiased=True)

    # Confidence interval
    RTS_std_dB = 10 * torch.log10(MSE_RTS_linear_std + MSE_RTS_linear_avg) - MSE_RTS_dB_avg


    print("RTS Smoother - MSE LOSS:", MSE_RTS_dB_avg, "[dB]")
    print("RTS Smoother - STD:", RTS_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg ,RTS_out]



