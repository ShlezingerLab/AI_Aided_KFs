import torch
from datetime import datetime

from Simulations.Linear_sysmdl import SystemModel
from Simulations.utils import DataGen
from Simulations.Linear_canonical.parameters import F, H, H_rotated, Q_structure, R_structure,\
   m, n, m1_0
import Simulations.config as config

from Pipelines.Pipeline_ERTS import Pipeline_ERTS as Pipeline
from Pipelines.Pipeline_EKF import Pipeline_EKF

from RTSNet.RTSNet_nn import RTSNetNN

from Smoothers.KalmanFilter_test import KFTest
from Smoothers.RTS_Smoother_test import S_Test

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
args.N_E = 1000
args.N_CV = 100
args.N_T = 200
args.T = 100
args.T_test = 100
args.variance = 0 # fixed initial state
m2_0 = args.variance * torch.eye(m) # 2nd moment of initial state
### training parameters ##################################################
args.n_steps = 2000
args.n_batch = 30
args.lr = 1e-3
args.wd = 1e-3

r2 = torch.tensor([10,1.,0.1,1e-2,1e-3])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)

dataFolderName = 'Simulations/Linear_canonical/H_rotated' + '/'
dataFileName = ['2x2_Hrot10_rq-1010_T100.pt','2x2_Hrot10_rq020_T100.pt','2x2_Hrot10_rq1030_T100.pt','2x2_Hrot10_rq2040_T100.pt','2x2_Hrot10_rq3050_T100.pt']

for index in range(0,len(r2)):

   print("1/r2 [dB]: ", 10 * torch.log10(1/r2[index]))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q2[index]))

   # True model
   Q = q2[index] * Q_structure
   R = r2[index] * R_structure
   sys_model = SystemModel(F, Q, H_rotated, R, args.T, args.T_test,q2[index],r2[index])
   sys_model.InitSequence(m1_0, m2_0)

   # Mismatched model
   sys_model_partialh = SystemModel(F, Q, H, R, args.T, args.T_test,q2[index],r2[index])
   sys_model_partialh.InitSequence(m1_0, m2_0)

   ###################################
   ### Data Loader (Generate Data) ###
   ###################################
   print("Start Data Gen")
   DataGen(args, sys_model, dataFolderName + dataFileName[index])
   print("Data Load")
   [train_input, train_target, cv_input, cv_target, test_input, test_target,_,_,_] = torch.load(dataFolderName + dataFileName[index])
   print("trainset size:",train_target.size())
   print("cvset size:",cv_target.size())
   print("testset size:",test_target.size())

   ##############################
   ### Evaluate Kalman Filter ###
   ##############################
   print("Evaluate Kalman Filter True")
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(args, sys_model, test_input, test_target)
   print("Evaluate Kalman Filter Partial")
   [MSE_KF_linear_arr_partialh, MSE_KF_linear_avg_partialh, MSE_KF_dB_avg_partialh] = KFTest(args, sys_model_partialh, test_input, test_target)


   #############################
   ### Evaluate RTS Smoother ###
   #############################
   print("Evaluate RTS Smoother True")
   [MSE_RTS_linear_arr, MSE_RTS_linear_avg, MSE_RTS_dB_avg, RTS_out] = S_Test(args, sys_model, test_input, test_target)
   print("Evaluate RTS Smoother Partial")
   [MSE_RTS_linear_arr_partialh, MSE_RTS_linear_avg_partialh, MSE_RTS_dB_avg_partialh, RTS_partialh_out] = S_Test(args, sys_model_partialh, test_input, test_target)


   #######################
   ### RTSNet Pipeline ###
   #######################

   # RTSNet with full info
   # Build Neural Network
   print("RTSNet with full model info")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model, args)
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setssModel(sys_model)
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(args)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
   RTSNet_Pipeline.save()
   ##########################################################################################################################################

   # RTSNet with mismatched model
   # Build Neural Network
   print("RTSNet with observation model mismatch")
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_partialh, args)
   ## Train Neural Network
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNet")
   RTSNet_Pipeline.setssModel(sys_model_partialh)
   RTSNet_Pipeline.setModel(RTSNet_model)
   RTSNet_Pipeline.setTrainingParams(args)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partialh, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_partialh, test_input, test_target, path_results)
   RTSNet_Pipeline.save()
   ##########################################################################################################################################

   print("RTSNet with estimated H")
   RTSNet_Pipeline = Pipeline(strTime, "RTSNet", "RTSNetEstH_"+ dataFileName[index])
   print("True Observation matrix H:", H_rotated)
   ### Least square estimation of H
   X = torch.squeeze(train_target[:,:,0])
   Y = torch.squeeze(train_input[:,:,0])
   for t in range(1,args.T):
      X_t = torch.squeeze(train_target[:,:,t])
      Y_t = torch.squeeze(train_input[:,:,t])
      X = torch.cat((X,X_t),0)
      Y = torch.cat((Y,Y_t),0)
   Y_1 = torch.unsqueeze(Y[:,0],1)
   Y_2 = torch.unsqueeze(Y[:,1],1)
   H_row1 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_1)
   H_row2 = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.T,X)),X.T),Y_2)
   H_hat = torch.cat((H_row1.T,H_row2.T),0)
   print("Estimated Observation matrix H:", H_hat)

   # Estimated model
   sys_model_esth = SystemModel(F, Q, H_hat, R, args.T, args.T_test,q2[index],r2[index])
   sys_model_esth.InitSequence(m1_0, m2_0)

   RTSNet_Pipeline.setssModel(sys_model_esth)
   RTSNet_model = RTSNetNN()
   RTSNet_model.NNBuild(sys_model_esth, args)
   RTSNet_Pipeline.setModel(RTSNet_model)
   
   RTSNet_Pipeline.setTrainingParams(args)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = RTSNet_Pipeline.NNTrain(sys_model_partialh, cv_input, cv_target, train_input, train_target, path_results)
   ## Test Neural Network
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,rtsnet_out,RunTime] = RTSNet_Pipeline.NNTest(sys_model_partialh, test_input, test_target, path_results)
   RTSNet_Pipeline.save()
