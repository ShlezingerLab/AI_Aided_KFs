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
from torch.distributions import Exponential

class SystemModel:

    def __init__(self, F, Q, H, R, T, T_test, q2, r2, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]
        self.Q = Q
        self.q2 = q2
        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]
        self.R = R
        self.r2 = r2
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
        return torch.bmm(self.F.view(1,self.F.shape[0],self.F.shape[1]).expand(x.shape[0],-1,-1), x)
    
    def h(self, x):
        return torch.bmm(self.H.view(1,self.H.shape[0],self.H.shape[1]).expand(x.shape[0],-1,-1), x)
        
    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, args, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################  
            # No noise 
            if torch.equal(Q_gen,torch.zeros(self.m,self.m)):
                xt = self.F.matmul(self.x_prev)
            # 1 dim noise
            elif self.m == 1: 
                xt = self.F.matmul(self.x_prev)
                if args.proc_noise_distri == 'normal':
                    eq = torch.normal(mean=0, std=Q_gen)
                elif args.proc_noise_distri == 'exponential':
                    lambda_exp = torch.sqrt(1/self.q2)
                    exponential_dist = Exponential(torch.tensor(lambda_exp))
                    # Sample from the Exponential distribution
                    eq = exponential_dist.sample()
                else:
                    raise ValueError('Unknown process noise distribution')
                # Additive Process Noise
                xt = torch.add(xt,eq)
            # Multi dim noise
            else:            
                xt = self.F.matmul(self.x_prev)
                # Sample from the Multivariate Normal distribution
                if args.proc_noise_distri == 'normal':
                    mean = torch.zeros([self.m])              
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                    eq = distrib.rsample()
                    # eq = torch.normal(mean, self.q)
                    eq = torch.reshape(eq[:], xt.size())
                # Sample from the Exponential distribution
                elif args.proc_noise_distri == 'exponential':
                    lambda_exp = torch.sqrt(1/self.q2)
                    exponential_dist = Exponential(lambda_exp * torch.ones([self.m])) # here we use same lambda for all dimensions of xt
                    eq = exponential_dist.sample((xt.size(),))
                    eq = torch.reshape(eq[:], xt.size())
                else:
                    raise ValueError('Unknown process noise distribution')
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            # Observation Noise
            # No noise
            if torch.equal(R_gen,torch.zeros(self.n,self.n)):
                yt = self.H.matmul(xt)
            # 1 dim noise
            elif self.n == 1: 
                yt = self.H.matmul(xt)
                if args.meas_noise_distri == 'normal':
                    er = torch.normal(mean=0, std=R_gen)
                elif args.meas_noise_distri == 'exponential':
                    lambda_exp = torch.sqrt(1/self.r2)
                    exponential_dist = Exponential(torch.tensor(lambda_exp))
                    # Sample from the Exponential distribution
                    er = exponential_dist.sample()
                else:
                    raise ValueError('Unknown measurement noise distribution')
                
                # Additive Observation Noise
                yt = torch.add(yt,er)
            # Multi dim noise
            else:  
                yt = self.H.matmul(xt)
                if args.meas_noise_distri == 'normal':
                    mean = torch.zeros([self.n])            
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                    er = distrib.rsample()
                    er = torch.reshape(er[:], yt.size())   
                elif args.meas_noise_distri == 'exponential':
                    lambda_exp = torch.sqrt(1/self.r2)
                    exponential_dist = Exponential(lambda_exp * torch.ones([self.n])) # here we use same λ for all dimensions of yt
                    er = exponential_dist.sample((yt.size(),))
                    er = torch.reshape(er[:], yt.size())
                else:
                    raise ValueError('Unknown measurement noise distribution')
                              
                # Additive Observation Noise
                yt = torch.add(yt,er)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt,1)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt,1)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, args, size, T, randomInit=False):
        ### init conditions ############################
        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            if args.distribution == 'uniform':
                ### if Uniform Distribution for random init
                for i in range(size):           
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)     
            
            elif args.distribution == 'normal':
                ### if Normal Distribution for random init
                for i in range(size):
                    distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m,1)
                    self.m1x_0_rand[i,:,0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')
            
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)### for sequence generation
        else: # fixed init
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)
            self.Init_batched_sequence(initConditions, self.m2x_0)### for sequence generation
        
        ### generate sequences ############################
        if(args.randomLength):
            # Allocate Array for Input and Target (use zero padding)
            self.Input = torch.zeros(size, self.n, args.T_max)
            self.Target = torch.zeros(size, self.m, args.T_max)
            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)# init with all false
            # Init Sequence Lengths
            T_tensor = torch.round((args.T_max-args.T_min)*torch.rand(size)).int()+args.T_min # Uniform distribution [100,1000]
            for i in range(0, size):
                # Generate Sequence
                self.GenerateSequence(args, self.Q, self.R, T_tensor[i].item())
                # Training sequence input
                self.Input[i, :, 0:T_tensor[i].item()] = self.y             
                # Training sequence output
                self.Target[i, :, 0:T_tensor[i].item()] = self.x
                # Mask for sequence length
                self.lengthMask[i, 0:T_tensor[i].item()] = True

        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T)

            # Set x0 to be x previous
            self.x_prev = self.m1x_0_batch
            xt = self.x_prev

            # Generate in a batched manner
            for t in range(0, T):
                ########################
                #### State Evolution ###
                ######################## 
                # No noise  
                if torch.equal(self.Q,torch.zeros(self.m,self.m)):
                    xt = self.f(self.x_prev)
                # 1 dim noise
                elif self.m == 1: 
                    xt = self.f(self.x_prev)
                    if args.proc_noise_distri == 'normal':
                        eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
                    elif args.proc_noise_distri == 'exponential':
                        lambda_exp = torch.sqrt(1/self.q2)
                        exponential_dist = Exponential(torch.tensor(lambda_exp))
                        # Sample from the Exponential distribution
                        eq = exponential_dist.sample((size,))
                        eq = torch.reshape(eq, (size,1,1))
                    else:
                        raise ValueError('args.proc_noise_distri not supported!')
                    # Additive Process Noise
                    xt = torch.add(xt,eq)
                # Multi dim noise
                else:             
                    xt = self.f(self.x_prev)
                    if args.proc_noise_distri == 'normal':
                        mean = torch.zeros([size, self.m])              
                        distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
                        eq = distrib.rsample().view(size,self.m,1)
                    elif args.proc_noise_distri == 'exponential':
                        lambda_exp = torch.sqrt(1/self.q2)
                        exponential_dist = Exponential(lambda_exp * torch.ones([self.m])) # here we use the same lambda for all dimensions
                        eq = exponential_dist.sample((size,))
                        eq = torch.reshape(eq, (size,self.m,1))
                    else:
                        raise ValueError('args.proc_noise_distri not supported!')
                    # Additive Process Noise
                    xt = torch.add(xt,eq)

                ################
                ### Emission ###
                ################
                # Observation Noise
                # No noise
                if torch.equal(self.R,torch.zeros(self.n,self.n)):
                    yt = self.h(xt)
                # 1 dim noise
                elif self.n == 1: 
                    yt = self.h(xt)
                    if args.meas_noise_distri == 'normal':
                        er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
                    elif args.meas_noise_distri == 'exponential':
                        lambda_exp = torch.sqrt(1/self.r2)
                        exponential_dist = Exponential(torch.tensor(lambda_exp))
                        er = exponential_dist.sample((size,))
                        er = torch.reshape(er, (size,1,1))
                    else:
                        raise ValueError('args.meas_noise_distri not supported!')
                    # Additive Observation Noise
                    yt = torch.add(yt,er)
                # Multi dim noise
                else:  
                    yt = self.H.matmul(xt)
                    if args.meas_noise_distri == 'normal':
                        mean = torch.zeros([size,self.n])            
                        distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
                        er = distrib.rsample().view(size,self.n,1)   
                    elif args.meas_noise_distri == 'exponential':
                        lambda_exp = torch.sqrt(1/self.r2)
                        exponential_dist = Exponential(lambda_exp * torch.ones([self.n])) # here we use the same lambda for all dimensions
                        er = exponential_dist.sample((size,))
                        er = torch.reshape(er, (size,self.n,1))
                    else:
                        raise ValueError('args.meas_noise_distri not supported!')       
                    # Additive Observation Noise
                    yt = torch.add(yt,er)

                ########################
                ### Squeeze to Array ###
                ########################

                # Save Current State to Trajectory Array
                self.Target[:, :, t] = torch.squeeze(xt,2)

                # Save Current Observation to Trajectory Array
                self.Input[:, :, t] = torch.squeeze(yt,2)

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt


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
