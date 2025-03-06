import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
from Simulations.utils import DataGen, DataLoader
import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,\
   m, n, m1_0, m2_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN
from RNN.RNN_FWandBW import Vanilla_RNN

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline

from Plot import Plot_RTS as Plot

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
path_results = 'RTSNet/'

####################
### Design Model ###
####################
InitIsRandom_train = False
InitIsRandom_cv = False
InitIsRandom_test = False
LengthIsRandom = False

args = config.general_settings()
### dataset parameters
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 100
args.T_test = 100
### training parameters
args.n_steps = 4000
args.n_batch = 30
args.lr = 1e-4
args.wd = 1e-3

r2 = torch.tensor([1e-3])
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

# True model
Q = q2 * Q_structure
R = r2 * R_structure
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/data/v0dB' + '/'
dataFileName = '2x2_rq3030_T100.pt'
# print("Start Data Gen")
# DataGen(args, sys_model, dataFolderName + dataFileName, randomInit_train=InitIsRandom_train,randomInit_cv=InitIsRandom_cv,randomInit_test=InitIsRandom_test,randomLength=LengthIsRandom)
print("Data Load")
if(InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   [train_input, train_target, train_init, cv_input, cv_target, cv_init, test_input, test_target, test_init] = torch.load(dataFolderName + dataFileName)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())
elif(LengthIsRandom):
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = torch.load(dataFolderName + dataFileName)
   ### Check sequence lengths
   # for sequences in train_target:
   #    print("trainset size:",sequences.size())
   # for sequences in test_target:
   #    print("testset size:",sequences.size())
else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader(dataFolderName + dataFileName)
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())
  


########################################
### Evaluate Observation Noise Floor ###
########################################
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

print("Observation Noise Floor - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor - STD:", obs_std_dB, "[dB]")

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
if InitIsRandom_test:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target, randomInit = True, test_init=test_init)
else: 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target)

#############################
### Evaluate RTS Smoother ###
#############################
print("Evaluate RTS Smoother True")
if InitIsRandom_test:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target, randomInit = True,test_init=test_init)
else:
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(sys_model, test_input, test_target)

PlotfolderName = 'Smoothers' + '/'
ComparedmodelName = 'Dataset'  
Plot = Plot(PlotfolderName, ComparedmodelName)
print("Plot")
Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, MSE_obs_linear_arr)

#######################
### RTSNet Pipeline ###
#######################

### RTSNet with full info ##############################################################################################
# Build Neural Network
print("RTSNet with full model info")
RTSNet_model = RTSNetNN()
RTSNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for RTSNet:",sum(p.numel() for p in RTSNet_model.parameters() if p.requires_grad))
## Train Neural Network
RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
RTSNet_Pipeline.setssModel(sys_model)
RTSNet_Pipeline.setModel(RTSNet_model)
RTSNet_Pipeline.setTrainingParams(args)
if (InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init)
else:
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
RTSNet_Pipeline.save()

### Vanilla RNN ###################################################################################
## Build RNN
print("Vanilla RNN")
RNN_model = Vanilla_RNN()
RNN_model.Build(args, sys_model,fully_agnostic = False)
print("Number of trainable parameters for RNN:",sum(p.numel() for p in RNN_model.parameters() if p.requires_grad))
RNN_Pipeline = Pipeline(strTime, "RTSNet", "VanillaRNN")
RNN_Pipeline.setssModel(sys_model)
RNN_Pipeline.setModel(RNN_model)
RNN_Pipeline.setTrainingParams(args)
if (InitIsRandom_train or InitIsRandom_cv or InitIsRandom_test):
   RNN_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, rnn=True, randomInit = True, cv_init=cv_init,train_init=train_init)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RNN_Pipeline.NNTest(sys_model, test_input, test_target, path_results, rnn=True,randomInit=True,test_init=test_init)
else:
   RNN_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, rnn=True)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RNN_Pipeline.NNTest(sys_model, test_input, test_target, path_results, rnn=True)
RNN_Pipeline.save()


