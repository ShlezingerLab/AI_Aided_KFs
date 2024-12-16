import torch
import torch.nn as nn
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
from Simulations.utils import DataGen
import Simulations.config as config
from Simulations.Linear_canonical.parameters import F, H, Q_structure, R_structure,\
   m, m1_0

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

from RTSNet.RTSNet_nn import RTSNetNN
from RTSNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_EKF import Pipeline_EKF 
from Pipelines.Pipeline_concat_models import Pipeline_twoRTSNets

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
# train, cv, test set size
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
# init condition
args.randomInit_train = False
args.randomInit_cv = False
args.randomInit_test = False
if args.randomInit_train or args.randomInit_cv or args.randomInit_test:
   # you can modify initial variance
   args.variance = 1
   args.distribution = 'normal' # 'uniform' or 'normal'
   m2_0 = args.variance * torch.eye(m)
else: 
   # deterministic initial condition
   m2_0 = 0 * torch.eye(m) 
# sequence length
args.T = 100
args.T_test = 100
args.randomLength = False
if args.randomLength:# you can modify T_max and T_min 
   args.T_max = 1000
   args.T_min = 100
   # set T and T_test to T_max for convenience of batch calculation
   args.T = args.T_max 
   args.T_test = args.T_max
else:
   train_lengthMask = None
   cv_lengthMask = None
   test_lengthMask = None
# noise
# determine noise distribution normal/exp (DEFAULT: "normal")
args.proc_noise_distri = "exponential"
args.meas_noise_distri = "exponential"
dataFileName = '2x2_rq3050_T100.pt'
r2 = torch.tensor([1e-3])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)
print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))
# True model
Q = q2 * Q_structure
R = r2 * R_structure
sys_model = SystemModel(F, Q, H, R, args.T, args.T_test,q2,r2)
sys_model.InitSequence(m1_0, m2_0)
print("State Evolution Matrix:",F)
print("Observation Matrix:",H)
sys_model_pass2 = sys_model # Since H=I, it can be used for pass2

### training parameters ##################################################
args.n_steps = 4000
args.n_batch = 30
args.lr = 1e-4
args.wd = 1e-3

path_results = 'RTSNet/'
# 1pass or 2pass
two_pass = True # if true: use two pass method, else: use one pass method

load_trained_pass1 = True # if True: load trained RTSNet pass1, else train pass1
# specify the path to save trained pass1 model
RTSNetPass1_path = "Simulations/Linear_canonical/results/exp(weights)/rq3050.pt"
# Save the dataset generated from testing RTSNet1 on train and CV data
load_dataset_for_pass2 = False # if True: load dataset generated from testing RTSNet1 on train and CV data
# Specify the path to save the dataset
DatasetPass1_path = "Simulations/Linear_canonical/data/exp/2ndPass_rq3050.pt" 

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Linear_canonical/data/exp' + '/'
# print("Start Data Gen")
# DataGen(args, sys_model, dataFolderName + dataFileName)
print("Data Load")
if args.randomLength:
   [train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask] = torch.load(dataFolderName + dataFileName)
else:
   [train_input, train_target, cv_input, cv_target, test_input, test_target,_,_,_] = torch.load(dataFolderName + dataFileName)

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())

########################################
### Evaluate Observation Noise Floor ###
########################################
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(args.N_T)# MSE [Linear]  
for i in range(args.N_T):
   MSE_obs_linear_arr[i] = loss_obs(test_input[i], test_target[i]).item()   
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
# print("Evaluate Kalman Filter True")
# if args.randomInit_test:
#    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target, randomInit = True, test_init=test_init, test_lengthMask=test_lengthMask)
# else: 
#    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target, test_lengthMask=test_lengthMask)

# #############################
# ### Evaluate RTS Smoother ###
# #############################
# print("Evaluate RTS Smoother True")
# if args.randomInit_test:
#    [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(args, sys_model, test_input, test_target, randomInit = True,test_init=test_init, test_lengthMask=test_lengthMask)
# else:
#    [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(args, sys_model, test_input, test_target, test_lengthMask=test_lengthMask)

# PlotfolderName = 'Smoothers' + '/'
# ComparedmodelName = 'Dataset'  
# Plot = Plot(PlotfolderName, ComparedmodelName)
# print("Plot")
# Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, MSE_obs_linear_arr)

#######################
### RTSNet Pipeline ###
#######################

### RTSNet with full info ##############################################################################################
if load_trained_pass1:
   print("Load RTSNet pass 1")
else:
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
   if (args.randomInit_train or args.randomInit_cv or args.randomInit_test):
      if args.randomLength:
         ## Train Neural Network
         [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init,train_lengthMask=train_lengthMask,cv_lengthMask=cv_lengthMask)
         ## Test Neural Network
         [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init,test_lengthMask=test_lengthMask)
      else:    
         [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init)
         [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init)
   else:
      if args.randomLength:
         [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,train_lengthMask=train_lengthMask,cv_lengthMask=cv_lengthMask)
         [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,test_lengthMask=test_lengthMask)
      else:
         [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
         [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   RTSNet_Pipeline.save()

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
      RTSNet_model_pass1.NNBuild(sys_model, args)
      RTSNet_Pipeline_pass1 = Pipeline(strTime, "RTSNet", "RTSNet")
      RTSNet_Pipeline_pass1.setssModel(sys_model)
      RTSNet_Pipeline_pass1.setModel(RTSNet_model_pass1)
      RTSNet_Pipeline_pass1.setTrainingParams(args)
      ### Optional to test it on test-set, just for checking
      print("Test RTSNet pass 1 on test set")
      [_, _, _,rtsnet_out_test,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, test_input, test_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)

      print("Test RTSNet pass 1 on training set")
      [_, _, _,rtsnet_out_train,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, train_input, train_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)
      print("Test RTSNet pass 1 on cv set")
      [_, _, _,rtsnet_out_cv,_] = RTSNet_Pipeline_pass1.NNTest(sys_model, cv_input, cv_target, path_results,load_model=True,load_model_path=RTSNetPass1_path)
      

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
   RTSNet_model1.NNBuild(sys_model, args)
   RTSNet_model2 = RTSNetNN()
   RTSNet_model2.NNBuild(sys_model_pass2, args)
   RTSNet_model1.load_state_dict(RTSNet_model1_weights)
   RTSNet_model2.load_state_dict(RTSNet_model2_weights)
   ## Set up Neural Network
   RTSNet_Pipeline_2passes = Pipeline_twoRTSNets(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline_2passes.setModel(RTSNet_model1, RTSNet_model2)
   RTSNet_Pipeline_2passes.setssModel(sys_model)
   RTSNet_Pipeline_2passes.setParams(args)
   NumofParameter = RTSNet_Pipeline_2passes.count_parameters()
   print("Number of parameters for RTSNet with 2 passes: ",NumofParameter)
   ## Test Neural Network   
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline_2passes.NNTest(sys_model, test_input, test_target, path_results)

##########################
### KalmanNet Pipeline ###
##########################

### KalmanNet with full info ##########################################################################################
# Build Neural Network
# print("KalmanNet with full model info")
# KalmanNet_model = KalmanNetNN()
# KalmanNet_model.NNBuild(sys_model, args)
# print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
# ## Train Neural Network
# KalmanNet_Pipeline = Pipeline_EKF(strTime, "RTSNet", "KalmanNet")
# KalmanNet_Pipeline.setssModel(sys_model)
# KalmanNet_Pipeline.setModel(KalmanNet_model)
# KalmanNet_Pipeline.setTrainingParams(args)
# if (args.randomInit_train or args.randomInit_cv or args.randomInit_test):
#    if args.randomLength:
#       ## Train Neural Network
#       [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init,train_lengthMask=train_lengthMask,cv_lengthMask=cv_lengthMask)
#       ## Test Neural Network
#       [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init,test_lengthMask=test_lengthMask)
#    else:    
#       [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, randomInit = True, cv_init=cv_init,train_init=train_init)
#       [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,randomInit=True,test_init=test_init)
# else:
#    if args.randomLength:
#       [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results,train_lengthMask=train_lengthMask,cv_lengthMask=cv_lengthMask)
#       [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results,test_lengthMask=test_lengthMask)
#    else:
#       [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
#       [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
# KalmanNet_Pipeline.save()