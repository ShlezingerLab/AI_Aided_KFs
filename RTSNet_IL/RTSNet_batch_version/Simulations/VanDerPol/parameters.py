"""This file contains the parameters for the Van Der Pol Oscillator simulation.

Created: 2023-07-21
"""


import torch
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd

#########################
### Design Parameters ###
#########################
m = 2 # state dim
n = 1 # observation dim

m1x_0_true = torch.tensor([0, -5]).float()
m2x_0_true = 0 * 0 * torch.eye(m)

m1x_0_design = torch.tensor([0, -5]).float()
m2x_0_design = 10 * 10 * torch.eye(m)

### Decimation
delta_t_gen =  0.1
delta_t = 0.1
ratio = delta_t_gen/delta_t

### ODE parameter
mu = 2


##################################
### State evolution function f ###
##################################
# Original f (not batched)
def Origin_f(x):
    
    y = torch.zeros(2)

    y[0] = x[0] + x[1] * delta_t
    y[1] = x[1] + (mu * (1 - x[0]*x[0]) * x[1] - x[0]) * delta_t

    return y

#########################################################################################
# batched version of f
def f(x):
    # Here, x.shape should be [batch_size, m, 1]
    batch_size, m, _ = x.shape
    y = torch.zeros(batch_size, m, 1)

    y[:, 0, 0] = x[:, 0, 0] + x[:, 1, 0] * delta_t
    y[:, 1, 0] = x[:, 1, 0] + (mu * (1 - x[:, 0, 0]*x[:, 0, 0]) * x[:, 1, 0] - x[:, 0, 0]) * delta_t

    return y

##############################
### Observation function h ###
##############################
# Observe only the postion
H_onlyPos = torch.tensor([[1, 0]]).float()
H_id = torch.eye(m)

# Original h (not batched)
def Origin_h(x):
    H = H_onlyPos.to(x.device)
    y = torch.matmul(H,x)
    return y

# Original identity h (not batched)
def Origin_h_id(x):
    H = H_id.to(x.device)
    y = torch.matmul(H,x)
    return y


#########################################################################################
# batched version of h
def h(x):
    """
    input x : [batch_size, 2, 1]
    output y: [batch_size, 1, 1]
    """
    y = x[:,0,:].unsqueeze(1)

    return y

# batched version of identity h
def h_id(x):
    """
    input x : [batch_size, 2, 1]
    output y: [batch_size, 2, 1]
    """
    y = x

    return y


###############################################
### process noise Q and observation noise R ###
###############################################

Q_structure = torch.eye(m)
R_structure = 1



