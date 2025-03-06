"""# **Class: System Model for Linear Cases**

1 Store system model parameters: 
    state transition matrix F, 
    observation matrix H, 
    process noise covariance matrix Q, 
    observation noise covariance matrix R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test, etc.

2 Generate dataset for linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:

    def __init__(self, F, Q, H, R, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]
        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]
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
        

    def f(self, x):
        return torch.matmul(self.F, x)
    
    def h(self, x):
        return torch.matmul(self.H, x)
        
    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0


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
                xt = self.F.matmul(self.x_prev)
            elif self.m == 1: # 1 dim noise
                xt = self.F.matmul(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:            
                xt = self.F.matmul(self.x_prev)
                mean = torch.zeros([self.m])              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                # eq = torch.normal(mean, self.q)
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            # Observation Noise
            if torch.equal(R_gen,torch.zeros(self.n,self.n)):# No noise
                yt = self.H.matmul(xt)
            elif self.n == 1: # 1 dim noise
                yt = self.H.matmul(xt)
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:  
                yt = self.H.matmul(xt)
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
                """ 
                ### Uncomment this if Uniform Distribution for random init 
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
                self.m1x_0_rand[i,:] = torch.squeeze(initConditions)
                """

                ### if Normal Distribution for random init
                distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                initConditions = distrib.rsample()
                self.m1x_0_rand[i,:] = torch.squeeze(initConditions)
                
            
            self.InitSequence(initConditions, self.m2x_0)### for sequence generation

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


    def sampling(self, q, r, gain):

        if (gain != 0):
            gain_q = 0.1
            #aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m)
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m) + aq
        Q_gen = torch.transpose(Aq, 0, 1) * Aq

        if (gain != 0):
            gain_r = 0.5
            #ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n)
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n) + ar
        R_gen = torch.transpose(Ar, 0, 1) * Ar

        return [Q_gen, R_gen]
