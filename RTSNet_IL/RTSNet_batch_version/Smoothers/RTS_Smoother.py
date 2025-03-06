"""# **Class: RTS Smoother**
Theoretical Linear RTS Smoother
"""
import torch

class rts_smoother:

    def __init__(self, SystemModel): 
        self.F = SystemModel.F
        self.m = SystemModel.m

        self.Q = SystemModel.Q

        self.H = SystemModel.H
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test


    # Compute the Smoother Gain
    def SGain(self, filter_sigma):
        self.SG = torch.bmm(filter_sigma, self.batched_F_T)
        filter_sigma_prior = torch.bmm(self.batched_F, filter_sigma)
        filter_sigma_prior = torch.bmm(filter_sigma_prior, self.batched_F_T) + self.Q
        self.SG = torch.bmm(self.SG, torch.inverse(filter_sigma_prior))

        # save smoothing gain
        self.SG_array[:,:,:,self.i] = self.SG
        self.i += 1
    
    # Innovation for Smoother
    def S_Innovation(self, filter_x, filter_sigma):
        filter_x_prior = torch.bmm(self.batched_F, filter_x)
        filter_sigma_prior = torch.bmm(self.batched_F, filter_sigma)
        filter_sigma_prior = torch.bmm(filter_sigma_prior, self.batched_F_T) + self.Q
        self.dx = self.s_m1x_nexttime - filter_x_prior
        self.dsigma = filter_sigma_prior - self.s_m2x_nexttime

    # Compute previous time step backwardly
    def S_Correct(self, filter_x, filter_sigma):
        # Compute the 1-st moment
        self.s_m1x_nexttime = filter_x + torch.bmm(self.SG, self.dx)

        # Compute the 2-nd moment
        self.s_m2x_nexttime = torch.bmm(self.dsigma, torch.transpose(self.SG, 1, 2))
        self.s_m2x_nexttime = filter_sigma - torch.bmm(self.SG, self.s_m2x_nexttime)

    def S_Update(self, filter_x, filter_sigma):
        self.SGain(filter_sigma)
        self.S_Innovation(filter_x, filter_sigma)
        self.S_Correct(filter_x, filter_sigma)

        return self.s_m1x_nexttime,self.s_m2x_nexttime

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, filter_x, filter_sigma):
        """
        input filter_x: batch of forward filtered x [batch_size, m, T]
        input filter_sigma: batch of forward filtered sigma [batch_size, m, m, T]
        """
        self.batch_size = filter_x.shape[0] # batch size
        T = filter_x.shape[2] # sequence length (maximum length if randomLength=True)

        # Pre allocate an array for predicted state and variance (use zero padding)
        self.s_x = torch.zeros([self.batch_size,self.m, T])
        self.s_sigma = torch.zeros([self.batch_size,self.m, self.m, T])
        # Pre allocate SG array
        self.SG_array = torch.zeros([self.batch_size,self.m,self.m,T]) # self.batch_size comes from the filter EKF
        self.i = 0 # Index for KG_array alocation
           
        # Set 1st and 2nd order moments for t=T
        self.s_m1x_nexttime = torch.unsqueeze(filter_x[:,:, T-1],2)
        self.s_m2x_nexttime = filter_sigma[:,:, :, T-1]
        self.s_x[:,:, T-1] = torch.squeeze(self.s_m1x_nexttime,2)
        self.s_sigma[:,:, :, T-1] = self.s_m2x_nexttime

        # Batched F and H
        self.batched_F = self.F.view(1,self.m,self.m).expand(self.batch_size,-1,-1)
        self.batched_F_T = torch.transpose(self.batched_F, 1, 2)
        self.batched_H = self.H.view(1,self.n,self.m).expand(self.batch_size,-1,-1)
        self.batched_H_T = torch.transpose(self.batched_H, 1, 2)

        # Generate in a batched manner
        for t in range(T-2,-1,-1):
            filter_xt = torch.unsqueeze(filter_x[:,:, t],2)
            filter_sigmat = filter_sigma[:,:, :, t]
            s_xt,s_sigmat = self.S_Update(filter_xt, filter_sigmat)
            self.s_x[:,:, t] = torch.squeeze(s_xt,2)
            self.s_sigma[:,:, :, t] = s_sigmat




   

   