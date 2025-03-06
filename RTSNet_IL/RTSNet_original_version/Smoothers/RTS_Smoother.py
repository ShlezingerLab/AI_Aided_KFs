"""# **Class: RTS Smoother**
Theoretical Linear RTS Smoother
"""
import torch

class rts_smoother:

    def __init__(self, SystemModel): 
        self.F = SystemModel.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = SystemModel.m

        self.Q = SystemModel.Q

        self.H = SystemModel.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test


    # Compute the Smoother Gain
    def SGain(self, filter_sigma):
        self.SG = torch.matmul(filter_sigma, self.F_T)
        filter_sigma_prior = torch.matmul(self.F, filter_sigma)
        filter_sigma_prior = torch.matmul(filter_sigma_prior, self.F_T) + self.Q
        self.SG = torch.matmul(self.SG, torch.inverse(filter_sigma_prior))
    
    # Innovation for Smoother
    def S_Innovation(self, filter_x, filter_sigma):
        filter_x_prior = torch.matmul(self.F, filter_x)
        filter_sigma_prior = torch.matmul(self.F, filter_sigma)
        filter_sigma_prior = torch.matmul(filter_sigma_prior, self.F_T) + self.Q
        self.dx = self.s_m1x_nexttime - filter_x_prior
        self.dsigma = filter_sigma_prior - self.s_m2x_nexttime

    # Compute previous time step backwardly
    def S_Correct(self, filter_x, filter_sigma):
        # Compute the 1-st moment
        self.s_m1x_nexttime = filter_x + torch.matmul(self.SG, self.dx)

        # Compute the 2-nd moment
        self.s_m2x_nexttime = torch.matmul(self.dsigma, torch.transpose(self.SG, 0, 1))
        self.s_m2x_nexttime = filter_sigma - torch.matmul(self.SG, self.s_m2x_nexttime)

    def S_Update(self, filter_x, filter_sigma):
        self.SGain(filter_sigma)
        self.S_Innovation(filter_x, filter_sigma)
        self.S_Correct(filter_x, filter_sigma)

        return self.s_m1x_nexttime,self.s_m2x_nexttime


    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, filter_x, filter_sigma, T):
        # Pre allocate an array for predicted state and variance
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])

        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)

        for t in range(T-2,-1,-1):
            filter_xt = filter_x[:, t]
            filter_sigmat = filter_sigma[:, :, t]
            s_xt,s_sigmat = self.S_Update(filter_xt, filter_sigmat)
            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)




   

   