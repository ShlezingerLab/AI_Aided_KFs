"""
The file contains utility functions for the simulations.
"""

import torch

def DataGen(args, SysModel_data, fileName, randomInit_train=False,randomInit_cv=False,randomInit_test=False,randomLength=False):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(args.N_E, args.T, randomInit=randomInit_train,randomLength=randomLength)
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target
    if(randomInit_train):
        training_init = SysModel_data.m1x_0_rand
    else:
        x0 = torch.squeeze(SysModel_data.m1x_0)
        training_init = x0.repeat(args.N_E,1) #size: N_E x m

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(args.N_CV, args.T, randomInit=randomInit_cv,randomLength=randomLength)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    if(randomInit_cv):
        cv_init = SysModel_data.m1x_0_rand
    else:
        x0 = torch.squeeze(SysModel_data.m1x_0)
        cv_init = x0.repeat(args.N_CV,1) #size: N_CV x m

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(args.N_T, args.T_test, randomInit=randomInit_test,randomLength=randomLength)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    if(randomInit_test):
        test_init = SysModel_data.m1x_0_rand
    else:
        x0 = torch.squeeze(SysModel_data.m1x_0)
        test_init = x0.repeat(args.N_T,1) #size: N_T x m

    #################
    ### Save Data ###
    #################
    if(randomInit_train or randomInit_cv or randomInit_test):
        torch.save([training_input, training_target, training_init, cv_input, cv_target, cv_init, test_input, test_target, test_init], fileName)
    else:
        torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName),pin_memory = False)
    training_input = training_input.squeeze()
    training_target = training_target.squeeze()
    cv_input = cv_input.squeeze()
    cv_target =cv_target.squeeze()
    test_input = test_input.squeeze()
    test_target = test_target.squeeze()
    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod/t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:,(0+offset)::ratio]
        if(i==0):
            all_tensors_out = torch.cat([tensor], dim=0).view(1,all_tensors.size()[1],-1)
        else:
            all_tensors_out = torch.cat([all_tensors_out,tensor.view(1,all_tensors.size()[1],-1)], dim=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    decimated_process = torch.cat(int(N_examples)*[decimated_process])
    noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])


    # Observations; additive Gaussian Noise
    observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r

    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    sequences_out = torch.zeros_like(sequences)
    # sequences_out = torch.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i,:,t] = h(sequence[:,t])
    i = i+1

    return sequences_out

def Short_Traj_Split(data_target, data_input, T):### Random Init is automatically incorporated
    data_target = list(torch.split(data_target,T+1,2)) # +1 to reserve for init
    data_input = list(torch.split(data_input,T+1,2)) # +1 to reserve for init

    data_target.pop()# Remove the last one which may not fullfill length T
    data_input.pop()# Remove the last one which may not fullfill length T

    data_target = torch.squeeze(torch.cat(list(data_target), dim=0))#Back to tensor and concat together
    data_input = torch.squeeze(torch.cat(list(data_input), dim=0))#Back to tensor and concat together
    # Split out init
    target = data_target[:,:,1:]
    input = data_input[:,:,1:]
    init = data_target[:,:,0]
    return [target, input, init]
