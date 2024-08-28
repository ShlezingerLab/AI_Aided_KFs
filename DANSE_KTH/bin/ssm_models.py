#####################################################
# Creator: Anubhab Ghosh 
# Nov 2023
#####################################################
from os import sys, path
# __file__ should be defined in this case
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import numpy as np
import math
import torch
from utils.utils import dB_to_lin, normalize
from scipy.integrate import solve_ivp

class LinearSSM(object):
    
    def __init__(self, n_states, n_obs, mu_e=None, mu_w=None, gamma=0.8, beta=1.0, drive_noise=False, normalize=False):
        
        self.n_states = n_states
        self.n_obs = n_obs
        self.gamma = gamma
        self.beta = beta
        if not mu_e is None:
            self.mu_e = mu_e
        else:
            self.mu_e = np.zeros((self.n_states,))
        if not mu_w is None:
            self.mu_w = mu_w
        else:
            self.mu_w = np.zeros((self.n_obs,))
        self.mu_w = mu_w
        self.drive_noise = drive_noise
        self.construct_F()
        self.construct_H()
        self.normalize = normalize
        
    def setStateCov(self, sigma_e2):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2):
        self.Cw = sigma_w2 * np.eye(self.n_obs)
    
    def construct_F(self):
        self.F = np.eye(self.n_states) + np.concatenate((np.zeros((self.n_states,1)), 
                                    np.concatenate((np.ones((1,self.n_states-1)), 
                                                    np.zeros((self.n_states-1,self.n_states-1))), 
                                                   axis=0)), 
                                   axis=1)
        
        self.F = self.F * self.gamma
        assert (np.linalg.eig(self.F)[0] <= 1.0).all() == True, "System is not stable!"
        
    def construct_H(self):
        
        self.H = np.rot90(np.eye(self.n_states)) + np.concatenate((np.concatenate((np.ones((1, self.n_states-1)), 
                                                            np.zeros((self.n_states-1, self.n_states-1))), 
                                                            axis=0), 
                                                            np.zeros((self.n_states,1))), 
                                                            axis=1)

        self.H = self.H[:self.n_obs, :] * self.beta 
        
    def generate_driving_noise(self, k, a=1.0, add_noise=False):
    
        #u_k = np.cos(a*k) # Previous idea (considering start at k=0)
        if add_noise == False:
            u_k = np.cos(a*(k+1)) # Current modification (considering start at k=1)
        elif add_noise == True:
            u_k = np.cos(a*(k+1) + np.random.normal(loc=0, scale=np.pi, size=(1,1))) # Adding noise to the sample
        return u_k

    def generate_state_sequence(self, T, sigma_e2_dB):
        
        self.G = np.ones((self.n_states, 1))
        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        x_arr = np.zeros((T, self.n_states))
        e_k_arr = np.random.multivariate_normal(self.mu_e, self.Ce, size=(T,))
        
        # Generate the sequence iteratively
        for k in range(0,T-1):

            # Generate driving noise (which is time varying)
            # Driving noise should be carefully selected as per value of k (start from k=0 or =1)
            if self.drive_noise == True: 
                u_k = self.generate_driving_noise(k, a=1.2, add_noise=False)
            else:
                u_k = 0.0

            # For each instant k, sample e_k, w_k
            e_k = e_k_arr[k]

            # Equation for updating the hidden state
            x_arr[k+1] = (self.F @ x_arr[k].reshape((-1,1)) + self.G @ np.array([u_k]).reshape((-1,1)) + e_k.reshape((-1,1))).reshape((-1,))
        
        if self.normalize == True:
            x_arr = normalize(x_arr)

        return x_arr
    
    def generate_measurement_sequence(self, x_arr, T, smnr_dB=10.0):
        
        #signal_p = ((np.einsum('ij,nj->ni', self.H, x_arr) - np.zeros_like(x_arr))**2).mean(axis=None)
        signal_p = np.var(np.einsum('ij,nj->ni', self.H, x_arr)) 
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        y_arr = np.zeros((T, self.n_obs))
        
        #print("sigma_w2: {}".format(self.sigma_w2))
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))

        #print(self.H.shape, x_arr.shape, y_arr.shape)        
        # Generate the sequence iteratively
        for k in range(T):
            # For each instant k, sample e_k, w_k
            w_k = w_k_arr[k]
            # Equation for calculating the output state
            y_arr[k] = self.H @ (x_arr[k]) + w_k
        
        return y_arr
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):

        x_arr = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_arr = self.generate_measurement_sequence(x_arr=x_arr, T=T, smnr_dB=smnr_dB)

        return x_arr, y_arr

class LorenzSSM(object):

    def __init__(self, n_states, n_obs, J, delta, delta_d, alpha=0.0, decimate=False, mu_e=None, mu_w=None, H=None, use_Taylor=True, normalize=False) -> None:
        
        self.n_states = n_states
        self.J = J
        self.delta = delta
        self.alpha = alpha # alpha = 0 -- Lorenz attractor, alpha = 1 -- Chen attractor
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.mu_e = mu_e
        if H is None:
            self.H = np.eye(self.n_obs)
        else:
            self.H = H
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor
        self.normalize = normalize
    
    def A_fn(self, z): 
        return np.array([
        [-(10 + 25*self.alpha), (10 + 25*self.alpha), 0],
        [(28 -  35*self.alpha), (29*self.alpha - 1), -z],
        [0, z, -(8.0 + self.alpha)/3]
    ])
    
    #def A_fn(self, z):
    #    return np.array([
    #                [-10, 10, 0],
    #                [28, -1, -z],
    #                [0, z, -8.0/3]
    #            ])
    
    def h_fn(self, x):
        """ 
        Linear measurement setup y = x + w
        """
        if type(x).__module__ == np.__name__:
            H_ = np.copy(self.H)
        elif type(x).__module__ == torch.__name__:
            H_ = torch.from_numpy(self.H).type(torch.FloatTensor)

        if len(x.shape) == 1:   
            y_ = H_ @ x
        elif len(x.shape) > 1:
            if type(x).__module__ == np.__name__:
                y_ = np.einsum('ij,nj->ni', H_, x)
            elif type(x).__module__ == torch.__name__:
                y_ = H_ @ x
        return y_
            
    def f_linearize(self, x):

        self.F = np.eye(self.n_states)
        for j in range(1, self.J+1):
            #self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)
            self.F += np.linalg.matrix_power(self.A_fn(x[0])*self.delta, j) / np.math.factorial(j)

        return self.F @ x
    
    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T, sigma_e2_dB):

        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        self.K = int(self.delta / self.delta_d)
        x_lorenz = np.zeros((T, self.n_states))
        e_k_arr = np.random.multivariate_normal(self.mu_e, self.Ce, size=(T,))
        #print(x_lorenz.shape)
        for t in range(0,T-1):
            x_lorenz[t+1] = self.f_linearize(x_lorenz[t]) + e_k_arr[t]
        
        if self.decimate == True:
            
            x_lorenz_d = x_lorenz[0:T:self.K,:]
        else:        
            x_lorenz_d = np.copy(x_lorenz)

        if self.normalize == True:
            x_lorenz_d = normalize(x_lorenz_d)
        return x_lorenz_d
    
    def generate_measurement_sequence(self, x_lorenz, T, smnr_dB=10.0):
        
        #signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        signal_p = np.var(self.h_fn(x_lorenz))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_lorenz = np.zeros((T, self.n_obs))
        
        #print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))
        
        #print(self.H.shape, x_lorenz.shape, y_lorenz.shape)
        for t in range(0,T):
            y_lorenz[t] = self.h_fn(x_lorenz[t]) + w_k_arr[t]

        return y_lorenz
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):

        #print(T)
        x_lorenz = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_lorenz = self.generate_measurement_sequence(x_lorenz=x_lorenz, T=T//self.K, smnr_dB=smnr_dB)

        #print(x_lorenz.shape, y_lorenz.shape)
        return x_lorenz, y_lorenz, self.Cw

class RosslerSSM(object):

    def __init__(self, n_states, n_obs, J, delta, delta_d, a=0.1, b=0.1, c=14.0, decimate=False, mu_e=None, mu_w=None, H=None, use_Taylor=True) -> None:
        
        self.n_states = n_states
        self.J = J
        self.delta = delta
        self.a = a
        self.b = b
        self.c = c
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.mu_e = mu_e
        if H is None:
            self.H = np.eye(self.n_obs)
        else:
            self.H = H
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor
    
    def A_fn(self, z): 
        return np.array([
        [0, -1, -1],
        [1, self.a, 0],
        [0, 0, (self.b / z[2]) + (z[0] - self.c)]
    ])
    
    #def A_fn(self, z):
    #    return np.array([
    #                [-10, 10, 0],
    #                [28, -1, -z],
    #                [0, z, -8.0/3]
    #            ])
    
    def h_fn(self, x):
        """ 
        Linear measurement setup y = x + w
        """
        if type(x).__module__ == np.__name__:
            H_ = np.copy(self.H)
        elif type(x).__module__ == torch.__name__:
            H_ = torch.from_numpy(self.H).type(torch.FloatTensor)

        if len(x.shape) == 1:   
            y_ = H_ @ x
        elif len(x.shape) > 1:
            if type(x).__module__ == np.__name__:
                y_ = np.einsum('ij,nj->ni', H_, x)
            elif type(x).__module__ == torch.__name__:
                y_ = H_ @ x
        return y_
            
    def f_linearize(self, x):

        self.F = np.eye(self.n_states)
        for j in range(1, self.J+1):
            #self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)
            #print(self.A_fn(x))
            self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)

        return self.F @ x
    
    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)
        self.Ce[self.n_states-1, self.n_states-1] = 1e-10 # Set the noise covariance in the z-dim to be zero, basically making it noise-free as it is zero mean!

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T, sigma_e2_dB):

        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        self.setStateCov(sigma_e2=self.sigma_e2)
        self.K = int(self.delta / self.delta_d)
        x_rossler = np.zeros((T, self.n_states))
        x_rossler[0, :] = np.ones(self.n_states) #* np.abs(np.random.normal(0, np.sqrt(self.sigma_e2), (1,)))
        e_k_arr = np.random.multivariate_normal(self.mu_e, self.Ce, size=(T,))

        #print(x_rossler[0, :])
        
        for t in range(0,T-1):
            x_rossler[t+1] = self.f_linearize(x_rossler[t]) + e_k_arr[t]
        
        if self.decimate == True:
            x_rossler_d = x_rossler[0:T:self.K,:]
        else:        
            x_rossler_d = np.copy(x_rossler)

        return x_rossler_d
    
    def generate_measurement_sequence(self, x_rossler, T, smnr_dB=10.0):
        
        signal_p = np.var(self.h_fn(x_rossler))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_rossler = np.zeros((T, self.n_obs))
        
        #print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))
        
        #print(self.H.shape, x_rossler.shape, y_rossler.shape)
        for t in range(0,T):
            y_rossler[t] = self.h_fn(x_rossler[t]) + w_k_arr[t]

        return y_rossler
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):

        x_rossler = self.generate_state_sequence(T=T, sigma_e2_dB=sigma_e2_dB)
        y_rossler = self.generate_measurement_sequence(x_rossler=x_rossler, T=T//self.K, smnr_dB=smnr_dB)
        return x_rossler, y_rossler, self.Cw

def L96(t, x, N=20, F_mu=8, sigma_e2=.1):
    """Lorenz 96 model with constant forcing
    Adapted from: https://www.wikiwand.com/en/Lorenz_96_model 
    """
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    F_N = np.random.normal(loc=F_mu, scale=np.sqrt(sigma_e2), size=(N,)) # Incorporating Process noise through the forcing constant
    for i in range(N):
        #print(F_N[i])
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F_N[i]
    return d

class Lorenz96SSM(object):

    def __init__(self, n_states, n_obs, delta, delta_d, F_mu=8, decimate=False, mu_w=None, H=None, method='RK45') -> None:
        
        self.n_states = n_states
        self.delta = delta
        self.delta_d = delta_d
        self.n_obs = n_obs
        self.decimate = decimate
        self.F_mu = F_mu
        if H is None:
            self.H = np.eye(self.n_obs)
        else:
            self.H = H
        self.mu_w = mu_w
        self.method = method
    
    def h_fn(self, x):
        """ 
        Linear measurement setup y = x + w
        """
        if len(x.shape) == 1:   
            y_ = self.H @ x
        elif len(x.shape) > 1:
            y_ = np.einsum('ij,nj->ni', self.H, x)
        return y_
    
    def setStateCov(self, sigma_e2=0.1):
        self.Ce = sigma_e2 * np.eye(self.n_states)

    def setMeasurementCov(self, sigma_w2=1.0):
        self.Cw = sigma_w2 * np.eye(self.n_obs)

    def generate_state_sequence(self, T_time, sigma_e2_dB):

        self.sigma_e2 = dB_to_lin(sigma_e2_dB)
        x0 = self.F_mu * np.ones(self.n_states)  # Initial state (equilibrium)
        x0[0] += self.delta  # Add small perturbation to the first variable
        sol = solve_ivp(L96, 
                        t_span=(0.0, T_time), 
                        y0=x0, 
                        args=(self.n_states, self.F_mu, self.sigma_e2,), 
                        method=self.method, 
                        t_eval=np.arange(0.0, T_time, self.delta), 
                        max_step=self.delta)
    
        x_lorenz = np.concatenate((sol.y.T, x0.reshape((1, -1))), axis=0)
        assert x_lorenz.shape[-1] == self.n_states, "Shape mismatch for generated state trajectory"
        
        T = x_lorenz.shape[0]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            x_lorenz_d = x_lorenz[0:T:K,:]
        else:        
            x_lorenz_d = np.copy(x_lorenz)

        return x_lorenz_d
    
    def generate_measurement_sequence(self, T, x_lorenz, smnr_dB=10.0):
        
        #signal_p = ((self.h_fn(x_lorenz) - np.zeros_like(x_lorenz))**2).mean()
        signal_p = np.var(self.h_fn(x_lorenz))
        #print("Signal power: {:.3f}".format(signal_p))
        self.sigma_w2 = signal_p / dB_to_lin(smnr_dB)
        self.setMeasurementCov(sigma_w2=self.sigma_w2)
        w_k_arr = np.random.multivariate_normal(self.mu_w, self.Cw, size=(T,))
        y_lorenz = np.zeros((T, self.n_obs))
        
        #print("smnr: {}, signal power: {}, sigma_w: {}".format(smnr_dB, signal_p, self.sigma_w2))
        
        #print(self.H.shape, x_lorenz.shape, y_lorenz.shape)
        for t in range(0,T):
            y_lorenz[t] = self.h_fn(x_lorenz[t]) + w_k_arr[t]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            y_lorenz_d = y_lorenz[0:T:K,:] 
        else:
            y_lorenz_d = np.copy(y_lorenz)

        return y_lorenz_d
    
    def generate_single_sequence(self, T, sigma_e2_dB, smnr_dB):
        
        T_time = T * self.delta
        x_lorenz = self.generate_state_sequence(T_time=T_time, sigma_e2_dB=sigma_e2_dB)
        y_lorenz = self.generate_measurement_sequence(T=T, x_lorenz=x_lorenz, smnr_dB=smnr_dB)

        return x_lorenz, y_lorenz, self.Cw
