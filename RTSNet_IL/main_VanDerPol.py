import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np

from Smoothers.EKF_test import EKFTest
from Smoothers.Extended_RTS_Smoother_test import S_Test

from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen
import Simulations.config as config

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_EKF import Pipeline_EKF
from Pipelines.Pipeline_concat_models import Pipeline_twoRTSNets

from datetime import datetime

from RTSNet.RTSNet_nn import RTSNetNN

from Plot import Plot_extended as Plot
# batched model
from Simulations.VanDerPol.parameters import m1x_0_true, m2x_0_true, m1x_0_design, m2x_0_design,\
m, n, f, h, h_id, Q_structure, R_structure
# not batched model (for Jacobian calculation use)
from Simulations.VanDerPol.parameters import Origin_f, Origin_h, Origin_h_id

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
# noise q and r
r2 = torch.tensor([1]).float()
q2 = torch.tensor([0.01]).float()

Q = q2[0] * Q_structure
R = r2[0] * R_structure

print("r^2: ", r2[0])
print("q^2: ", q2[0])

traj_resultName = ['traj_VanDerPol.pt']

args.N_E = 10
args.N_CV = 5
args.N_T = 10
args.T = 40 # number of time-steps in each trajectory of training set
args.T_test = 40 # number of time-steps in each trajectory of test set
DatafolderName = 'Simulations/VanDerPol/data' + '/'
dataFileName_gen = 'data_VanDerPol.pt'
# specify the path to save trained pass1 model
RTSNetPass1_path = "Simulations/VanDerPol/results/RTSNet1.pt"

### training parameters ##################################################
args.n_steps = 2000
args.n_batch = 5
args.lr = 1e-3
args.wd = 1e-3
path_results = 'RTSNet/'
# args.in_mult_KNet = 50
# args.in_mult_RTSNet = 50
# 1pass or 2pass
two_pass = False # if true: use two pass method, else: use one pass method

load_trained_pass1 = False # if True: load trained RTSNet pass1, else train pass1
# Save the dataset generated from testing RTSNet1 on train and CV data
load_dataset_for_pass2 = False # if True: load dataset generated from testing RTSNet1 on train and CV data
# Specify the path to save the dataset
DatasetPass1_path = "Simulations/VanDerPol/data/ResultofPass1.pt" 


#######################
###  System model   ###
#######################
# Ground Truth model, for data generation
sys_model_gen = SystemModel(f, Q, h, R, args.T, args.T_test, m, n, Origin_f, Origin_h)# parameters for GT
sys_model_gen.InitSequence(m1x_0_true, m2x_0_true)# x0 and P0

# Partial model, feed to smoothers
sys_model_feed = SystemModel(f, Q, h, R, args.T, args.T_test, m, n, Origin_f, Origin_h)
sys_model_feed.InitSequence(m1x_0_design, m2x_0_design)# x0 and P0

# Model for 2nd pass, H=I
sys_model_pass2 = SystemModel(f, Q, h_id, r2[0]*torch.eye(m), args.T, args.T_test, m, m, Origin_f, Origin_h_id)
sys_model_pass2.InitSequence(m1x_0_design, m2x_0_design)# x0 and P0

#################################
###  Generate and load data   ###
#################################
# print("Start Data Gen")
# DataGen(args, sys_model_gen, DatafolderName + dataFileName_gen)
print("Data Load")
### load train set
print(dataFileName_gen)
[train_input,train_target, cv_input, cv_target, test_input, test_target,_,_,_] =  torch.load(DatafolderName + dataFileName_gen)  
### load test set
# Load .mat file
# data_test_target = sio.loadmat(DatafolderName +'GT_traj.mat')
# data_test_input = sio.loadmat(DatafolderName +'observation_traj.mat')
# # Get the array
# test_target_np = data_test_target['x_true']
# test_input_np = data_test_input['z']
# # Convert it to PyTorch tensor
# test_target = torch.from_numpy(test_target_np).float()
# test_input = torch.from_numpy(test_input_np).float()
# # Remove the initial state
# test_target = test_target[:, 1:].unsqueeze(0)
# test_input = test_input[:, 1:].unsqueeze(0)

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())

### Save trajectories
# Convert PyTorch tensor to numpy array
xtrue_np = test_target.numpy()
z_np = test_input.numpy()
# Add init state to the beginning of the array
new_col = np.repeat(m1x_0_design.numpy().reshape(1,-1,1), repeats=args.N_T, axis=0)
xtrue_np = np.concatenate((new_col, xtrue_np), axis=2)
zeros = np.zeros((args.N_T,1,1))  # Create a new array of zeros with shape (N_T,1,1)
z_np = np.concatenate((zeros, z_np), axis=2)  # Concatenate 'zeros' and 'x' along the third dimension
# Save numpy array as a .mat file
sio.savemat(DatafolderName +'10GT_traj.mat', {'x_true_array': xtrue_np})
sio.savemat(DatafolderName +'10obs_traj.mat', {'z_array': z_np})

########################################
### Evaluate Observation Noise Floor ###
########################################
N_T = len(test_input)
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(N_T)# MSE [Linear]

for j in range(0, N_T):   
   MSE_obs_linear_arr[j] = loss_obs(test_input[j], test_target[j]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor(test dataset) - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor(test dataset) - STD:", obs_std_dB, "[dB]")


######################################
### Evaluate Filters and Smoothers ###
######################################
### Evaluate EKF 
print("Evaluate EKF true")
[MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(args, sys_model_feed, train_input, train_target)

# ## Evaluate RTS 
print("Evaluate RTS true")
[MSE_ERTS_linear_arr, MSE_ERTS_linear_avg, MSE_ERTS_dB_avg, ERTS_out] = S_Test(args, sys_model_feed, train_input, train_target)

### Save trajectories
EKF_sample = torch.reshape(EKF_out[0],[m,args.T_test])
ERTS_sample = torch.reshape(ERTS_out[0],[m,args.T_test])
# Convert PyTorch tensor to numpy array
EKF_sample_np = EKF_sample.numpy()
ERTS_sample_np = ERTS_sample.numpy()
# Add init state to the beginning of the array
new_col = m1x_0_design.numpy().reshape(-1,1)
EKF_sample_np = np.concatenate((new_col, EKF_sample_np), axis=1)
ERTS_sample_np = np.concatenate((new_col, ERTS_sample_np), axis=1)
# Save numpy array as a .mat file
sio.savemat(DatafolderName +'EKF_traj.mat', {'EKF_out': EKF_sample_np})
sio.savemat(DatafolderName +'ERTS_traj.mat', {'ERTS_out': ERTS_sample_np})

#######################
### Evaluate RTSNet ###
#######################

## RTSNet with full info ####################################################################################
######################
## RTSNet - 1 full ###
######################
if load_trained_pass1:
   print("Load RTSNet pass 1")
else:
   ## Build Neural Network
   print("RTSNet with full model info")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_feed, args)
   # ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setssModel(sys_model_feed)
   RTSNet_Pipeline.setModel(RTSNet_model)
   print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
   RTSNet_Pipeline.setTrainingParams(args)    
   # [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_feed, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_feed, test_input, test_target, path_results)
   ### Save trajectories  
   RTSNet_sample = torch.reshape(rtsnet_out[0],[m,args.T_test])
   # Convert PyTorch tensor to numpy array
   RTSNet_sample_np = RTSNet_sample.detach().numpy()
   # Add init state to the beginning of the array
   new_col = m1x_0_design.numpy().reshape(-1,1)
   RTSNet_sample_np = np.concatenate((new_col, RTSNet_sample_np), axis=1)
   # Save numpy array as a .mat file
   sio.savemat(DatafolderName +'RTSNet1_traj.mat', {'RTSNet1_out': RTSNet_sample_np})


####################################################################################

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
      [_, _, _,rtsnet_out_cv,_] = RTSNet_Pipeline_pass1.NNTest(sys_model_feed, cv_input, cv_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)
      

      train_input_pass2 = rtsnet_out_train
      train_target_pass2 = train_target
      cv_input_pass2 = rtsnet_out_cv
      cv_target_pass2 = cv_target

      torch.save([train_input_pass2, train_target_pass2, cv_input_pass2, cv_target_pass2, test_input, test_target], DatasetPass1_path)
   #######################################
   ## RTSNet_2passes with full info   
   # Build Neural Network
   print("RTSNet(with full model info) pass 2 pipeline start!")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_pass2, args)
   print("Number of trainable parameters for RTSNet pass 2:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet_pass2")
   RTSNet_Pipeline.setssModel(sys_model_pass2)
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(args)
   #######################################
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_pass2, cv_input_pass2, cv_target_pass2, train_input_pass2, train_target_pass2, path_results)
   RTSNet_Pipeline.save()
   print("RTSNet pass 2 pipeline end!")
   #######################################
   # load trained Neural Network
   print("Concat two RTSNets and test")
   RTSNet_model1_weights = torch.load(RTSNetPass1_path)
   RTSNet_model2_weights = torch.load('RTSNet/best-model-weights.pt')
   RTSNet_model1 = RTSNetNN()
   RTSNet_model1.NNBuild(sys_model_feed, args)
   RTSNet_model2 = RTSNetNN()
   RTSNet_model2.NNBuild(sys_model_pass2, args)
   RTSNet_model1.load_state_dict(RTSNet_model1_weights)
   RTSNet_model2.load_state_dict(RTSNet_model2_weights)
   ## Set up Neural Network
   RTSNet_Pipeline_2passes = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline_2passes.setModel(RTSNet_model1, RTSNet_model2)
   RTSNet_Pipeline_2passes.setssModel(sys_model_feed)
   RTSNet_Pipeline_2passes.setParams(args)
   NumofParameter = RTSNet_Pipeline_2passes.count_parameters()
   print("Number of parameters for RTSNet with 2 passes: ",NumofParameter)
   ## Test Neural Network   
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline_2passes.NNTest(sys_model_feed, test_input, test_target, path_results)

   





