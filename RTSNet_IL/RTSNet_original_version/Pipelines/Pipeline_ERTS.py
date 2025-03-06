"""
This file contains the class Pipeline_ERTS, 
which is used to train and test RTSNet in both linear and non-linear cases.
"""

import torch
import torch.nn as nn
import time
import random
from Plot import Plot_extended as Plot

class Pipeline_ERTS:

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

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, args, alpha=0.5):
        self.N_steps = args.n_steps  # Number of Training Steps
        self.N_B = args.n_batch # Number of Samples in Batch
        self.learningRate = args.lr # Learning Rate
        self.weightDecay = args.wd # L2 Weight Regularization - Weight Decay
        self.alpha = alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=0.9, patience=20)


    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, CompositionLoss = False, MaskOnState=False, rnn=False, randomInit = False, cv_init=None,train_init=None):

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_steps])
        self.MSE_cv_dB_epoch = torch.empty([self.N_steps])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_steps])
        self.MSE_train_dB_epoch = torch.empty([self.N_steps])
        
        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_steps):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            
            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                
                n_e = random.randint(0, self.N_E - 1)
                y_training = train_input[n_e]
                SysModel.T = y_training.size()[-1]

                x_out_training_forward = torch.empty(SysModel.m, SysModel.T)
                x_out_training = torch.empty(SysModel.m, SysModel.T)
                
                if(randomInit):
                    self.model.InitSequence(train_init[n_e], SysModel.T)
                else:
                    self.model.InitSequence(SysModel.m1x_0, SysModel.T)
                
                for t in range(0, SysModel.T):
                    x_out_training_forward[:, t] = self.model(y_training[:, t], None, None, None)
                x_out_training[:, SysModel.T-1] = x_out_training_forward[:, SysModel.T-1] # backward smoothing starts from x_T|T 
                self.model.InitBackward(x_out_training[:, SysModel.T-1]) 
                x_out_training[:, SysModel.T-2] = self.model(None, x_out_training_forward[:, SysModel.T-2], x_out_training_forward[:, SysModel.T-1],None)
                for t in range(SysModel.T-3, -1, -1):
                    x_out_training[:, t] = self.model(None, x_out_training_forward[:, t], x_out_training_forward[:, t+1],x_out_training[:, t+2])
                    

                # Compute Training Loss
                LOSS = 0
                if (CompositionLoss):
                    if(MaskOnState):                      
                        y_hat = torch.empty([SysModel.n, SysModel.T])
                        for t in range(SysModel.T):
                            y_hat[:,t] = SysModel.h(x_out_training[:,t])
                        LOSS = self.alpha * self.loss_fn(x_out_training[mask], train_target[n_e][mask])+(1-self.alpha)*self.loss_fn(y_hat[mask], train_input[n_e][mask])
                    else:
                        y_hat = torch.empty([SysModel.n, SysModel.T])
                        for t in range(SysModel.T):
                            y_hat[:,t] = SysModel.h(x_out_training[:,t])
                        LOSS = self.alpha * self.loss_fn(x_out_training, train_target[n_e])+(1-self.alpha)*self.loss_fn(y_hat, train_input[n_e])
                
                else:
                    if(MaskOnState):
                        LOSS = self.loss_fn(x_out_training[mask], train_target[n_e][mask])
                    else:
                        LOSS = self.loss_fn(x_out_training, train_target[n_e])
                
                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward(retain_graph=True)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            with torch.no_grad():
                for j in range(0, self.N_CV):
                    y_cv = cv_input[j]
                    SysModel.T_test = y_cv.size()[-1]

                    x_out_cv_forward = torch.empty(SysModel.m, SysModel.T_test)
                    x_out_cv = torch.empty(SysModel.m, SysModel.T_test)
                    
                    if(randomInit):
                        if(cv_init==None):
                            self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
                        else:
                            self.model.InitSequence(cv_init[j], SysModel.T_test)                       
                    else:
                        self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)
 
                    for t in range(0, SysModel.T_test):
                        x_out_cv_forward[:, t] = self.model(y_cv[:, t], None, None, None)
                    x_out_cv[:, SysModel.T_test-1] = x_out_cv_forward[:, SysModel.T_test-1] # backward smoothing starts from x_T|T
                    self.model.InitBackward(x_out_cv[:, SysModel.T_test-1]) 
                    x_out_cv[:, SysModel.T_test-2] = self.model(None, x_out_cv_forward[:, SysModel.T_test-2], x_out_cv_forward[:, SysModel.T_test-1],None)
                    for t in range(SysModel.T_test-3, -1, -1):
                        x_out_cv[:, t] = self.model(None, x_out_cv_forward[:, t], x_out_cv_forward[:, t+1],x_out_cv[:, t+2])                       

                    # Compute CV Loss
                    if(MaskOnState):
                        MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv[mask], cv_target[j][mask]).item()
                    else:
                        MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j]).item()

                # Average
                self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    if(rnn):
                        torch.save(self.model, path_results + 'rnn_best-model.pt')
                    else:
                        torch.save(self.model, path_results + 'best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")
            
            
            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, MaskOnState=False, rnn=False,randomInit=False,test_init=None,load_model=False,load_model_path=None):

        self.N_T = len(test_input)

        self.MSE_test_linear_arr = torch.empty([self.N_T])

        if MaskOnState:
            mask = torch.tensor([True,False,False])
            if SysModel.m == 2: 
                mask = torch.tensor([True,False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Load model
        if load_model:
            self.model = torch.load(load_model_path) 
        else:
            if (rnn):
                self.model = torch.load(path_results+'rnn_best-model.pt')
            else:
                self.model = torch.load(path_results+'best-model.pt')

        self.model.eval()

        torch.no_grad()

        x_out_list = []
        start = time.time()
        for j in range(0, self.N_T):

            y_mdl_tst = test_input[j]
            SysModel.T_test = y_mdl_tst.size()[-1]

            x_out_test_forward_1 = torch.empty(SysModel.m,SysModel.T_test)
            x_out_test = torch.empty(SysModel.m, SysModel.T_test)

            if (randomInit):
                self.model.InitSequence(test_init[j], SysModel.T_test)               
            else:
                self.model.InitSequence(SysModel.m1x_0, SysModel.T_test)         
           
            for t in range(0, SysModel.T_test):
                x_out_test_forward_1[:, t] = self.model(y_mdl_tst[:, t], None, None, None)
            x_out_test[:, SysModel.T_test-1] = x_out_test_forward_1[:, SysModel.T_test-1] # backward smoothing starts from x_T|T 
            self.model.InitBackward(x_out_test[:, SysModel.T_test-1]) 
            x_out_test[:, SysModel.T_test-2] = self.model(None, x_out_test_forward_1[:, SysModel.T_test-2], x_out_test_forward_1[:, SysModel.T_test-1],None)
            for t in range(SysModel.T_test-3, -1, -1):
                x_out_test[:, t] = self.model(None, x_out_test_forward_1[:, t], x_out_test_forward_1[:, t+1],x_out_test[:, t+2])
            
            if(MaskOnState):
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test[mask], test_target[j][mask]).item()
            else:
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j]).item()
            x_out_list.append(x_out_test)
        
        end = time.time()
        t = end - start

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_list, t]

    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_steps, self.N_B, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

    def PlotTrain_RTS(self, MSE_KF_linear_arr, MSE_KF_dB_avg, MSE_RTS_linear_arr, MSE_RTS_dB_avg):
    
        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_E,self.N_steps, self.N_B, MSE_KF_dB_avg, MSE_RTS_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, MSE_RTS_linear_arr, self.MSE_test_linear_arr)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)