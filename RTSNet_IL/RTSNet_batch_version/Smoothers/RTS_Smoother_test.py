import torch
import torch.nn as nn
import time
from Smoothers.Linear_KF import KalmanFilter
from Smoothers.RTS_Smoother import rts_smoother

def S_Test(args, SysModel, test_input, test_target, allStates=True,\
     randomInit = False,test_init=None, test_lengthMask=None):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_RTS_linear_arr = torch.zeros(args.N_T)
    # Allocate empty tensor for output
    RTS_out = torch.zeros([args.N_T, SysModel.m, test_input.size()[2]]) # N_T x m x T

    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only
        # loc = torch.tensor([True,True,True,False,False,False])# for kitti
    
    start = time.time()

    KF = KalmanFilter(SysModel)
    RTS = rts_smoother(SysModel)
  
    # Init and Forward&Backward Computation 
    if(randomInit):
        KF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))        
    else:
        KF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(args.N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))           
    KF.GenerateBatch(test_input)
    RTS.GenerateBatch(KF.x, KF.sigma)

    end = time.time()
    t = end - start

    RTS_out = RTS.s_x  
    
    # MSE loss
    for j in range(args.N_T):# cannot use batch due to different length and std computation   
        if(allStates):
            if args.randomLength:
                MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x[j,:,:], test_target[j,:,:]).item()
        else: # mask on state
            if args.randomLength:
                MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_RTS_linear_arr[j] = loss_rts(RTS.s_x[j,loc,:], test_target[j,loc,:]).item()

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



