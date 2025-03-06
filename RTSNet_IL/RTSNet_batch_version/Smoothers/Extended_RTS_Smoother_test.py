import torch
import torch.nn as nn
import time
from Smoothers.EKF import ExtendedKalmanFilter
from Smoothers.Extended_RTS_Smoother import Extended_rts_smoother

def S_Test(args, SysModel, test_input, test_target, allStates=True,\
     randomInit = False,test_init=None, test_lengthMask=None):

    # LOSS
    loss_rts = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_ERTS_linear_arr = torch.zeros(args.N_T)
    # Allocate empty tensor for output
    ERTS_out = torch.zeros([args.N_T, SysModel.m, test_input.size()[2]]) # N_T x m x T
    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only
        # loc = torch.tensor([True,True,True,False,False,False])# for kitti
    
    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel)  
    ERTS = Extended_rts_smoother(SysModel)
    
   # Init and Forward&Backward Computation 
    if(randomInit):
        EKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))        
    else:
        EKF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(args.N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(args.N_T,-1,-1))           
    EKF.GenerateBatch(test_input)
    ERTS.GenerateBatch(EKF.x, EKF.sigma)

    end = time.time()
    t = end - start

    ERTS_out = ERTS.s_x 
    # MSE loss
    for j in range(args.N_T):# cannot use batch due to different length and std computation   
        if(allStates):
            if args.randomLength:
                MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
            else:      
                MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x[j,:,:], test_target[j,:,:]).item()
        else: # mask on state
            if args.randomLength:
                MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
            else:           
                MSE_ERTS_linear_arr[j] = loss_rts(ERTS.s_x[j,loc,:], test_target[j,loc,:]).item()

    # Average
    MSE_ERTS_linear_avg = torch.mean(MSE_ERTS_linear_arr)
    MSE_ERTS_dB_avg = 10 * torch.log10(MSE_ERTS_linear_avg)

    # Standard deviation
    MSE_ERTS_linear_std = torch.std(MSE_ERTS_linear_arr, unbiased=True)

    # Confidence interval
    ERTS_std_dB = 10 * torch.log10(MSE_ERTS_linear_std + MSE_ERTS_linear_avg) - MSE_ERTS_dB_avg

    print("Extended RTS Smoother - MSE LOSS:", MSE_ERTS_dB_avg, "[dB]")
    print("Extended RTS Smoother - STD:", ERTS_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out]



