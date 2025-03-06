"""
This file contains the class Pipeline_concat_models. 
It concatenates trained RTSNet pass1 and pass2 models, and test the joint RTSNet - 2.
"""

import torch
import torch.nn as nn
import time
import random
from Plot import Plot_extended as Plot

class Pipeline_twoRTSNets:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.SysModel = ssModel

    def setModel(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    
    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, randomInit=False,test_init=None):

        self.N_T = test_input.size()[0]

        self.MSE_test_linear_arr = torch.empty([self.N_T])

        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        self.model1.eval()
        self.model2.eval()

        torch.no_grad()

        x_out_array = torch.empty(self.N_T,SysModel.m, SysModel.T_test)
        start = time.time()
        for j in range(0, self.N_T):
            
            if (randomInit):
                self.model1.InitSequence(test_init[j], SysModel.T_test)               
            else:
                self.model1.InitSequence(SysModel.m1x_0, SysModel.T_test) 

            y_mdl_tst = test_input[j, :, :]

            x_out_test_forward_1 = torch.empty(SysModel.m,SysModel.T_test)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test)
            for t in range(0, SysModel.T_test):
                x_out_test_forward_1[:, t] = self.model1(y_mdl_tst[:, t], None, None, None)
            x_out_test[:, SysModel.T_test-1] = x_out_test_forward_1[:, SysModel.T_test-1] # backward smoothing starts from x_T|T 
            self.model1.InitBackward(x_out_test[:, SysModel.T_test-1]) 
            x_out_test[:, SysModel.T_test-2] = self.model1(None, x_out_test_forward_1[:, SysModel.T_test-2], x_out_test_forward_1[:, SysModel.T_test-1],None)
            for t in range(SysModel.T_test-3, -1, -1):
                x_out_test[:, t] = self.model1(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t+1],x_out_test[:, t+2])
            
            ########################################################################
            # Second pass
            x_out_test_forward_2 = torch.empty(SysModel.m,SysModel.T_test)
            x_out_test_2 = torch.empty(SysModel.m, SysModel.T_test)
            # Init with results from pass1
            if (randomInit):
                self.model2.InitSequence(test_init[j], SysModel.T_test)               
            else:
                self.model2.InitSequence(SysModel.m1x_0, SysModel.T_test) 
            # Second filtering pass
            for t in range(0, SysModel.T_test):
                x_out_test_forward_2[:, t] = self.model2(x_out_test[:, t],None, None, None)
            x_out_test_2[:, SysModel.T_test-1] = x_out_test_forward_2[:, SysModel.T_test-1] # backward smoothing starts from x_T|T 
            self.model2.InitBackward(x_out_test_2[:, SysModel.T_test-1]) 
            x_out_test_2[:, SysModel.T_test-2] = self.model2(None, x_out_test_forward_2[:, SysModel.T_test-2], x_out_test_forward_2[:, SysModel.T_test-1],None)
            # Second smoothing pass
            for t in range(SysModel.T_test-3, -1, -1):
                x_out_test_2[:, t] = self.model2(None, x_out_test_forward_2[:, t], x_out_test_forward_2[:, t+1],x_out_test_2[:, t+2])          
            x_out_test = x_out_test_2

            if(MaskOnState):
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test[mask], test_target[j][mask]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()              
            x_out_array[j,:,:] = x_out_test
        
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE Cross Validation and STD
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_array, t]


    def count_parameters(self):
        return sum(p.numel() for p in self.model1.parameters() if p.requires_grad)+sum(p.numel() for p in self.model2.parameters() if p.requires_grad)