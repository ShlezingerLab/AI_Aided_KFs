"""# **Class: Extended RTS Smoother**
Theoretical Non-linear Linear RTS Smoother
"""
import torch
from Simulations.Lorenz_Atractor.parameters import getJacobian

class Extended_rts_smoother:

    def __init__(self, SystemModel):
        self.f = SystemModel.f
        self.m = SystemModel.m

        self.Q = SystemModel.Q

        self.h = SystemModel.h
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test


    # Compute the Smoothing Gain
    def SGain(self, filter_x, filter_sigma):
        # Predict the 1-st moment of x
        self.filter_x_prior = self.f(filter_x)
        # Compute the Jacobians
        self.UpdateJacobians(getJacobian(filter_x,self.f), getJacobian(self.filter_x_prior, self.h))
        self.SG = torch.matmul(filter_sigma, self.F_T)
        self.filter_sigma_prior = torch.matmul(self.F, filter_sigma)
        self.filter_sigma_prior = torch.matmul(self.filter_sigma_prior, self.F_T) + self.Q
        self.SG = torch.matmul(self.SG, torch.inverse(self.filter_sigma_prior))

    # Innovation for Smoother
    def S_Innovation(self, filter_x, filter_sigma):
        self.dx = self.s_m1x_nexttime - self.filter_x_prior
        self.dsigma = self.filter_sigma_prior - self.s_m2x_nexttime

    # Compute previous time step backwardly
    def S_Correct(self, filter_x, filter_sigma):
        # Compute the 1-st moment
        self.s_m1x_nexttime = filter_x + torch.matmul(self.SG, self.dx)

        # Compute the 2-nd moment
        self.s_m2x_nexttime = torch.matmul(self.dsigma, torch.transpose(self.SG, 0, 1))
        self.s_m2x_nexttime = filter_sigma - torch.matmul(self.SG, self.s_m2x_nexttime)

    def S_Update(self, filter_x, filter_sigma):
        self.SGain(filter_x, filter_sigma)
        self.S_Innovation(filter_x, filter_sigma)
        self.S_Correct(filter_x, filter_sigma)

        return self.s_m1x_nexttime,self.s_m2x_nexttime

        #########################

    def UpdateJacobians(self, F, H):
        self.F = F
        self.F_T = torch.transpose(F,0,1)
        self.H = H
        self.H_T = torch.transpose(H,0,1)
        #print(self.H,self.F,'\n')
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, filter_x, filter_sigma, T):
        # Pre allocate an array for predicted state and variance
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])
        # Pre allocate SG array
        self.SG_array = torch.zeros((self.T,self.m,self.m))

        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)

        for t in range(T-2,-1,-1):
            filter_xt = torch.squeeze(filter_x[:, t])
            filter_sigmat = torch.squeeze(filter_sigma[:, :, t])
            s_xt,s_sigmat = self.S_Update(filter_xt, filter_sigmat)
            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)