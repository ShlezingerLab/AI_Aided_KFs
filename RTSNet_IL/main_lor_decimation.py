import numpy as np
import torch
import pickle
import torch.nn as nn
from datetime import datetime

import Smoothers.EKF_test as EKF_test
from Smoothers.Extended_RTS_Smoother_test import S_Test

from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config
from Simulations.utils import Decimate_and_perturbate_Data,Short_Traj_Split
# batched model
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,delta_t_gen,delta_t,\
f, h, fInacc, Q_structure, R_structure
# not batched model
from Simulations.Lorenz_Atractor.parameters import Origin_f, Origin_fInacc, Origin_h

from Pipelines.Pipeline_EKF import Pipeline_EKF
from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_concat_models import Pipeline_twoRTSNets

from RTSNet.KalmanNet_nn import KalmanNetNN
from RTSNet.RTSNet_nn import RTSNetNN

from Plot import Plot_extended as Plot

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

##########################
### Parameter settings ###
##########################
args = config.general_settings()
args.use_cuda = False # use GPU or not
if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

### dataset parameters ###################################################
offset = 0 # offset for the data
chop = False # whether to chop the dataset sequences into smaller ones

# Noise q and r
r = torch.tensor([1])
lambda_q = torch.tensor([0.3873]) # Tuned to compensate for sampling mismatch
print("1/r2 [dB]: ", 10 * torch.log10(1/r[0]**2))
print("Tuned 1/q2 [dB]: ", 10 * torch.log10(1/lambda_q[0]**2))
Q = (lambda_q[0]**2) * Q_structure
R = (r[0]**2) * R_structure 

# 'data size' or 'orgin' 
sim_case = 'data size'
if sim_case == 'data size':
   args.N_E = 2
   args.N_CV = 5
   args.N_T = 10
   args.T = 3000
   args.T_test = 3000
   DatafolderName = 'Simulations/Lorenz_Atractor/data/data_size/'
   dataFileName = ['size2.pt','size5.pt','size10.pt','size50.pt','size100.pt','size1000.pt']
   DatafileName = dataFileName[0]

elif sim_case == 'origin':
   args.N_E = 100
   args.N_CV = 5
   args.N_T = 10
   args.T = 3000
   args.T_test = 3000
   DatafolderName = 'Simulations/Lorenz_Atractor/data/'
   DatafileName = 'decimated_r0_Ttest3000.pt'
else:
   raise Exception("Invalid sim_case")

# data_gen = 'data_gen.pt'
# data_gen_file = torch.load('Simulations/Lorenz_Atractor/data/'+data_gen)
# [true_sequence] = data_gen_file['All Data']
### training parameters ##################################################
args.n_steps = 2000
args.n_batch = min(10, args.N_E) # number of trajectories in a mini-batch
args.lr = 1e-3
args.wd = 1e-4

switch = 'full' # 'full' or 'partial'

# 1pass or 2pass
two_pass = True # if true: use two pass method, else: use one pass method

load_trained_pass1 = True # if True: load trained RTSNet pass1, else train pass1
# specify the path to save trained pass1 model
RTSNetPass1_path = "Simulations/Lorenz_Atractor/results/decimation/data_size/best-model-weights_size2.pt"
# Save the dataset generated from testing RTSNet1 on train and CV data
load_dataset_for_pass2 = False # if True: load dataset generated from testing RTSNet1 on train and CV data
# Specify the path to save the dataset
DatasetPass1_path = "Simulations/Lorenz_Atractor/data/data_size/2ndPass/size2.pt" 

path_results = 'RTSNet/'


#######################
###  System model   ###
#######################
# True Model
sys_model_true = SystemModel(f, Q, h, R, args.T, args.T_test,m,n, Origin_f, Origin_h)
sys_model_true.InitSequence(m1x_0, m2x_0)

# Model with partial Info
sys_model = SystemModel(fInacc, Q, h, R, args.T, args.T_test,m,n, Origin_fInacc, Origin_h)
sys_model.InitSequence(m1x_0, m2x_0)

# Model to feed to the KNet and RTSNet
if switch == 'full':
   sys_model_feed = sys_model_true
elif switch == 'partial':
   sys_model_feed = sys_model
else:
   raise Exception("Invalid switch")

##############################################
### Generate and load data Decimation case ###
##############################################
########################
# print("Data Gen")
# ########################
# [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, args.N_T, Origin_h, r[0], offset) 
# [train_target_long, train_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, args.N_E, Origin_h, r[0], offset)
# [cv_target_long, cv_input_long] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t, args.N_CV, Origin_h, r[0], offset)
# if chop:
#    print("chop training data")  
#    [train_target, train_input, train_init] = Short_Traj_Split(train_target_long, train_input_long, args.T)
#    args.N_E = train_target.size()[0]
# else:
#    print("no chopping") 
#    train_target = train_target_long
#    train_input = train_input_long

# if sim_case == 'data size':
#    print("Use the same test set from" + dataFileName[0])
#    [_,_, _,_, test_input, test_target] =  torch.load(DatafolderName + dataFileName[0]) 

# # Save dataset
# if(chop):
#    torch.save([train_input, train_target, train_init, cv_input_long, cv_target_long, test_input, test_target], DatafolderName+DatafileName)
# else:
#    torch.save([train_input, train_target, cv_input_long, cv_target_long, test_input, test_target], DatafolderName+DatafileName)

#########################
print("Data Load")
#########################
[train_input, train_target, cv_input_long, cv_target_long, test_input, test_target] = torch.load(DatafolderName+DatafileName)  

if(chop):
   print("chop training data")  
   [train_target, train_input, train_init] = Short_Traj_Split(train_target, train_input, args.T)
   args.N_E = train_target.size()[0]

print("testset size:",test_target.size())
print("trainset size:",train_target.size())
print("cvset size:",cv_target_long.size())

###############################
### Load data from GNN-BP's ###
###############################
# compact_path = "Simulations/Lorenz_Atractor/data/lorenz_trainset300k.pickle"
# with open(compact_path, 'rb') as f:
#    data = pickle.load(f)
# testdata = [data[0][0:T_test], data[1][0:T_test]]
# states, meas = testdata
# test_target =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
# test_input = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
# print("testset size:",test_target.size())
# traindata = [data[0][T_test:(T_test+T*N_E)], data[1][T_test:(T_test+T*N_E)]]
# states, meas = traindata
# train_target =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
# train_input = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
# [train_target, train_input, train_init] = Short_Traj_Split(train_target, train_input, T)
# cvdata = [data[0][(T_test+T*N_E):], data[1][(T_test+T*N_E):]]
# states, meas = cvdata
# cv_target_long =  torch.from_numpy(np.asarray(states, dtype=np.float32).transpose(1,0)).unsqueeze(0)
# cv_input_long = torch.from_numpy(np.asarray(meas, dtype=np.float32).transpose(1,0)).unsqueeze(0)
# [cv_target_long, cv_input_long, cv_init] = Short_Traj_Split(cv_target_long, cv_input_long, T)
# print("trainset size:",train_target.size())
# print("cvset size:",cv_target_long.size())

########################################
### Evaluate Observation Noise Floor ###
########################################
args.N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T)# MSE [Linear]
for j in range(0, args.N_T):        
   MSE_obs_linear_arr[j] = loss_obs(test_input[j], test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")
###################################################
args.N_E = len(train_input)
MSE_obs_linear_arr = torch.empty(args.N_E)# MSE [Linear]
for j in range(0, args.N_E):        
   MSE_obs_linear_arr[j] = loss_obs(train_input[j], train_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(train dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(train dataset) - STD:", obs_std_dB, "[dB]")

######################################
### Evaluate Filters and Smoothers ###
######################################
# ### EKF
# print("Start EKF test J=5")
# [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKF_test.EKFTest(args, sys_model_true, test_input, test_target)
# print("Start EKF test J=2")
# [MSE_EKF_linear_arr_partial, MSE_EKF_linear_avg_partial, MSE_EKF_dB_avg_partial, EKF_KG_array_partial, EKF_out_partial] = EKF_test.EKFTest(args, sys_model, test_input, test_target)

# ### MB Extended RTS
# print("Start RTS test J=5")
# [MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(args, sys_model_true, test_input, test_target)
# print("Start RTS test J=2")
# [MSE_ERTS_linear_arr_partial, MSE_ERTS_linear_avg_partial, MSE_ERTS_dB_avg_partial, ERTS_out_partial] = S_Test(args, sys_model, test_input, test_target)

####################
### KalmanNet ######
####################
## Build Neural Network
# KNet_model = KalmanNetNN()
# KNet_model.NNBuild(sys_model_feed, args)
# KNet_Pipeline = Pipeline_EKF(strTime, "RTSNet", "KalmanNet")
# KNet_Pipeline.setModel(KNet_model)
# KNet_Pipeline.setssModel(sys_model_feed)
# print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
# # Train Neural Network
# KNet_Pipeline.setTrainingParams(args)
# if(chop):
#    KNet_Pipeline.NNTrain(args.N_E, train_input, train_target, args.N_CV, cv_input_long, cv_target_long,randomInit=True,train_init=train_init)
# else:
#    KNet_Pipeline.NNTrain(args.N_E, train_input, train_target, args.N_CV, cv_input_long, cv_target_long)
# # Test Neural Network
# [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out] = KNet_Pipeline.NNTest(args.N_T, test_input, test_target)

###############
### RTSNet  ###
###############
# ## Build Neural Network
if load_trained_pass1:
   print("Load RTSNet pass 1")
else:
   if switch == 'full':
      print("RTSNet with full info")
   elif switch == 'partial':
      print("RTSNet with model mismatch")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_feed, args)
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(args)
   NumofParameter = RTSNet_Pipeline.count_parameters()
   print("Number of parameters for RTSNet: ",NumofParameter)
   if(chop):
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_feed, cv_input_long, cv_target_long, train_input, train_target, path_results,randomInit=True,train_init=train_init)
   else:
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_feed, cv_input_long, cv_target_long, train_input, train_target, path_results)
   ## Test Neural Network
   # RTSNet_Pipeline.model = torch.load('RTSNet/checkpoints/LorenzAttracotor/decimation/model/best-model_r0_J2_NE1000_MSE-15.5.pt')
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_feed, test_input, test_target, path_results)

if two_pass:
   ################################
   ## RTSNet - 2 with full info ###
   ################################
      if load_dataset_for_pass2:
         print("Load dataset for pass 2")
         [train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target] = torch.load(DatasetPass1_path)
         
         print("Data Shape for RTSNet pass 2:")
         print("testset state x size:",test_target.size())
         print("testset observation y size:",test_input.size())
         print("trainset state x size:",train_target_pass2.size())
         print("trainset observation y size:",len(train_input_pass2),train_input_pass2[0].size())
         print("cvset state x size:",cv_target_pass2.size())
         print("cvset observation y size:",len(cv_input_pass2),cv_input_pass2[0].size())  
      else:
         ### save result of RTSNet1 as dataset for RTSNet2 
         RTSNet_model_pass1 = RTSNetNN()
         RTSNet_model_pass1.NNBuild(sys_model_feed, args)
         RTSNet_Pipeline_pass1 = Pipeline(strTime, "RTSNet", "RTSNet")
         RTSNet_Pipeline_pass1.setssModel(sys_model_feed)
         RTSNet_Pipeline_pass1.setModel(RTSNet_model_pass1)
         RTSNet_Pipeline_pass1.setTrainingParams(args)
         ### Optional to test it on test-set, just for checking
         print("Test RTSNet pass 1 on test set")
         [_, _, _,rtsnet_out_test,_] = RTSNet_Pipeline_pass1.NNTest(sys_model_feed, test_input, test_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)

         print("Test RTSNet pass 1 on training set")
         [_, _, _,rtsnet_out_train,_] = RTSNet_Pipeline_pass1.NNTest(sys_model_feed, train_input, train_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)
         print("Test RTSNet pass 1 on cv set")
         [_, _, _,rtsnet_out_cv,_] = RTSNet_Pipeline_pass1.NNTest(sys_model_feed, cv_input_long, cv_target_long, path_results,load_model=True,load_model_path=RTSNetPass1_path)
         
         train_input_pass2 = rtsnet_out_train
         train_target_pass2 = train_target
         cv_input_pass2 = rtsnet_out_cv
         cv_target_pass2 = cv_target_long

         torch.save([train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target], DatasetPass1_path)
      #######################################
      ## RTSNet_2passes with full info   
      # Build Neural Network
      print("RTSNet(with full model info) pass 2 pipeline start!")
      RTSNet_model2 = RTSNetNN()
      RTSNet_model2.NNBuild(sys_model_feed, args)
      print("Number of trainable parameters for RTSNet pass 2:",sum(p.numel() for p in RTSNet_model2.parameters() if p.requires_grad))
      ## Train Neural Network
      RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet_pass2")
      RTSNet_Pipeline.setssModel(sys_model_feed)
      RTSNet_Pipeline.setModel(RTSNet_model2)
      RTSNet_Pipeline.setTrainingParams(args)
      #######################################
      [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_feed, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results)
      RTSNet_Pipeline.save()
      print("RTSNet pass 2 pipeline end!")
      
      ###########################
      ### Concat two RTSNets  ###
      ###########################
      ## load trained Neural Network
      print("Concatenated RTSNet-2")
      RTSNet_model1_weights = torch.load(RTSNetPass1_path)
      RTSNet_model2_weights = torch.load('RTSNet/best-model-weights.pt')
      RTSNet_model1 = RTSNetNN()
      RTSNet_model1.NNBuild(sys_model_feed, args)
      RTSNet_model2 = RTSNetNN()
      RTSNet_model2.NNBuild(sys_model_feed, args)
      RTSNet_model1.load_state_dict(RTSNet_model1_weights)
      RTSNet_model2.load_state_dict(RTSNet_model2_weights)
      ## Train Neural Network
      RTSNet_Pipeline = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
      RTSNet_Pipeline.setModel(RTSNet_model1, RTSNet_model2)
      RTSNet_Pipeline.setssModel(sys_model_feed)
      RTSNet_Pipeline.setParams(args)
      NumofParameter = RTSNet_Pipeline.count_parameters()
      print("Number of parameters for RTSNet: ",NumofParameter)
      ## Test Neural Network
      [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out_2pass,RunTime] = RTSNet_Pipeline.NNTest(sys_model_feed, test_input, test_target, path_results)

# Save trajectories
# trajfolderName = 'RTSNet/checkpoints/LorenzAttracotor/decimation/traj' + '/'
# DataResultName = 'traj_lor_dec.pt'
# target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
# input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
# torch.save({# 'PF J=5':PF_out,
#             # 'PF J=2':PF_out_partial,
#             # 'True':target_sample,
#             # 'Observation':input_sample,
#             # 'EKF J=5':EKF_out,
#             # 'EKF J=2':EKF_out_partial,
#             # 'RTS J=5':ERTS_out,
#             # 'RTS J=2':ERTS_out_partial,
#             # 'PS J=5':PS_out,
#             # 'PS J=2':PS_out_partial,
#             'RTSNet': rtsnet_out,
#             'RTSNet_2pass': rtsnet_out_2pass,
#             'KNet': knet_out,
#             }, trajfolderName+DataResultName)

#############
### Plot  ###
#############
# titles = ["True Trajectory","Observation","RTSNet","RTSNet_2pass"]
# input = [target_sample,input_sample,rtsnet_out,rtsnet_out_2pass]
# Net_Plot = Plot(trajfolderName,DataResultName)
# Net_Plot.plotTrajectories(input,3, titles,trajfolderName+"lor_dec_trajs.png")





