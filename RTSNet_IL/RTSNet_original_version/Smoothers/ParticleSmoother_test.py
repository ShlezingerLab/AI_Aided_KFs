import pyparticleest.models.nlg as nlg
import pyparticleest.simulator as simulator
import pyparticleest.utils.kalman as kalman
import time
import torch
import torch.nn as nn
import numpy as np

            
class Model(nlg.NonlinearGaussianInitialGaussian):
    def __init__(self, SystemModel, x_0=None):
        if x_0 == None:
            x0 = SystemModel.m1x_0
        else:
            x0 = x_0
        P0 = SystemModel.m2x_0
        Q = SystemModel.Q.numpy()
        R = SystemModel.R.numpy()
        super(Model, self).__init__(x0=x0, Px0=P0, Q=Q, R=R)
        self.f = SystemModel.f
        self.n = SystemModel.n
        self.g = lambda x: torch.squeeze(SystemModel.h(x))
        self.m = SystemModel.m


    def calc_f(self, particles, u, t):
        N_p = particles.shape[0]
        particles_f = np.empty((N_p, self.n))
        for k in range(N_p):
            particles_f[k,:] = self.f(torch.tensor(particles[k,:]))
        return particles_f

    def calc_g(self, particles, t):
        N_p = particles.shape[0]
        particles_g = np.empty((N_p, self.m))
        for k in range(N_p):
            particles_g[k,:] = self.g(torch.tensor(particles[k,:]))
        return particles_g

def PSTest(SysModel, test_input, test_target, N_FWParticles=100, M_BWTrajs=10, init_cond=None):
    
    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear] not [dB]
    MSE_PS_linear_arr = torch.empty(N_T)

    PS_out = torch.empty((N_T, test_target.size()[1], SysModel.T_test))

    start = time.time()

    for j in range(N_T):
        if init_cond is None:
            model = Model(SysModel, SysModel.m1x_0)
        else:
            model = Model(SysModel, x_0=init_cond[j, :])
        y_in = test_input[j, :, :].T.numpy().squeeze()
        sim = simulator.Simulator(model, u=None, y=y_in)
        sim.simulate(N_FWParticles, M_BWTrajs, filter='PF', smoother='full', meas_first=False)
        PS_out[j, :, :] = torch.from_numpy(sim.get_smoothed_mean()[1:,].T) # start from T=1 since 0 is for x0


    for j in range(N_T):
        MSE_PS_linear_arr[j] = loss_fn(torch.tensor(PS_out[j, :, :]), test_target[j, :, :])

    end = time.time()
    t = end - start

    MSE_PS_linear_avg = torch.mean(MSE_PS_linear_arr)
    MSE_PS_dB_avg = 10 * torch.log10(MSE_PS_linear_avg)

    # Standard deviation
    MSE_PS_linear_std = torch.std(MSE_PS_linear_arr, unbiased=True)

    # Confidence interval
    PS_std_dB = 10 * torch.log10(MSE_PS_linear_std + MSE_PS_linear_avg) - MSE_PS_dB_avg

    print("Particle Smoother - MSE LOSS:", MSE_PS_dB_avg, "[dB]")
    print("Particle Smoother - STD:", PS_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_PS_linear_arr, MSE_PS_linear_avg, MSE_PS_dB_avg, PS_out, t]


