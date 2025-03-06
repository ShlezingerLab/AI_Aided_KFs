"""# **Class: System Model for Non-linear Cases**

1 Store system model parameters: 
    state transition function f, 
    observation function h, 
    process noise Q, 
    observation noise R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test,
    state dimension m,
    observation dimension n, etc.

2 Generate datasets for non-linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:

    def __init__(self, f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.m = m
        self.Q = Q
        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n
        self.R = R
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = torch.squeeze(m1x_0)
        self.m2x_0 = torch.squeeze(m2x_0)


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################   
            if torch.equal(Q_gen,torch.zeros(self.m,self.m)):# No noise
                 xt = self.f(self.x_prev)   
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:            
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m])              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            yt = self.h(xt)
            # Observation Noise         
            if self.n == 1: # 1 dim noise
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:  
                mean = torch.zeros([self.n])            
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], yt.size())       
                # Additive Observation Noise
                yt = torch.add(yt,er)
            
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, randomInit=False, randomLength=False):
        if(randomLength):
            # Allocate Empty list for Input
            self.Input = []
            # Allocate Empty list for Target
            self.Target = []
            # Init Sequence Lengths
            T_tensor = torch.round(900*torch.rand(size)).int()+100 # Uniform distribution [100,1000]
        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T)

        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.empty(size, self.m)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):
            # Generate Sequence

            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
                self.m1x_0_rand[i,:] = torch.squeeze(initConditions)
            self.InitSequence(initConditions, self.m2x_0)
            
            if(randomLength):
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                # Training sequence input
                self.Input.append(self.y)
                # Training sequence output
                self.Target.append(self.x)           
            else:
                self.GenerateSequence(self.Q, self.R, T)
                # Training sequence input
                self.Input[i, :, :] = self.y
                # Training sequence output
                self.Target[i, :, :] = self.x