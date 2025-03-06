"""This file contains the parameters for the Lorenz Atractor simulation.

Update 2023-02-06: f and h support batch size speed up

Update 2023-06-16: add original f and h (not batched) 
"""


import torch
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd

#########################
### Design Parameters ###
#########################
m = 3
n = 3
variance = 0
m1x_0 = torch.ones(m, 1) 
m2x_0 = 0 * 0 * torch.eye(m)

### Decimation
delta_t_gen =  1e-5
delta_t = 0.02
ratio = delta_t_gen/delta_t

### Taylor expansion order
J = 5 
J_mod = 2

### Angle of rotation in the 3 axes
roll_deg = yaw_deg = pitch_deg = 1

roll = roll_deg * (math.pi/180)
yaw = yaw_deg * (math.pi/180)
pitch = pitch_deg * (math.pi/180)

RX = torch.tensor([
                [1, 0, 0],
                [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)]])
RY = torch.tensor([
                [math.cos(pitch), 0, math.sin(pitch)],
                [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)]])
RZ = torch.tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]])

RotMatrix = torch.mm(torch.mm(RZ, RY), RX)

### Auxiliar MultiDimensional Tensor B and C (they make A --> Differential equation matrix)
B = torch.tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(m,m), torch.zeros(m,m)]).float()

C = torch.tensor([[-10, 10,    0],
                  [ 28, -1,    0],
                  [  0,  0, -8/3]]).float()

######################################################
### State evolution function f for Lorenz Atractor ###
######################################################
# Original f (not batched)
def Origin_f_gen(x):

    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.add(torch.reshape(torch.matmul(B, x),(m,m)).T,C)
    
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t_gen, j)/math.factorial(j)).to(x.device)
        F = torch.add(F, F_add).to(x.device)

    return torch.matmul(F, x)

def Origin_f(x):

    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = (torch.add(torch.reshape(torch.matmul(B, x),(m,m)).T,C)).to(x.device)
    
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j)).to(x.device)
        F = torch.add(F, F_add).to(x.device)

    return torch.matmul(F, x)

def Origin_fInacc(x):
    A = torch.add(torch.reshape(torch.matmul(B, x),(m,m)).T,C)   
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J_mod+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j)).to(x.device)
        F = torch.add(F, F_add).to(x.device)

    return torch.matmul(F, x)

def Origin_fRotate(x):
    A = (torch.add(torch.reshape(torch.matmul(B, x),(m,m)).T,C)).to(x.device)
    A_rot = torch.mm(RotMatrix,A)   
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A_rot*delta_t, j)/math.factorial(j)).to(x.device)
        F = torch.add(F, F_add).to(x.device)
    return torch.matmul(F, x)

#########################################################################################
# batched version of f

### f_gen is for dataset generation
def f_gen(x, jacobian=False):
    BX = torch.zeros([x.shape[0],m,m]).float().to(x.device) #[batch_size, m, m]
    BX[:,1,0] = torch.squeeze(-x[:,2,:]) 
    BX[:,2,0] = torch.squeeze(x[:,1,:])
    Const = C.to(x.device)
    A = torch.add(BX, Const)  
    # Taylor Expansion for F    
    F = torch.eye(m).to(x.device)
    F = F.reshape((1, m, m)).repeat(x.shape[0], 1, 1) # [batch_size, m, m] identity matrix
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t_gen, j)/math.factorial(j))
        F = torch.add(F, F_add)
    if jacobian:
        return torch.bmm(F, x), F
    else:
        return torch.bmm(F, x)

### f will be fed to filters and KNet, note that the mismatch comes from delta_t
def f(x, jacobian=False):
    BX = torch.zeros([x.shape[0],m,m]).float().to(x.device) #[batch_size, m, m]
    BX[:,1,0] = torch.squeeze(-x[:,2,:]) 
    BX[:,2,0] = torch.squeeze(x[:,1,:]) 
    Const = C.to(x.device)
    A = torch.add(BX, Const) 
    # Taylor Expansion for F    
    F = torch.eye(m).to(x.device)
    F = F.reshape((1, m, m)).repeat(x.shape[0], 1, 1) # [batch_size, m, m] identity matrix
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    if jacobian:
        return torch.bmm(F, x), F
    else:
        return torch.bmm(F, x)

### fInacc will be fed to filters and KNet, note that the mismatch comes from delta_t and J_mod
def fInacc(x, jacobian=False):
    BX = torch.zeros([x.shape[0],m,m]).float().to(x.device) #[batch_size, m, m]
    BX[:,1,0] = torch.squeeze(-x[:,2,:]) 
    BX[:,2,0] = torch.squeeze(x[:,1,:]) 
    Const = C.to(x.device)
    A = torch.add(BX, Const)     
    # Taylor Expansion for F    
    F = torch.eye(m).to(x.device)
    F = F.reshape((1, m, m))
    F = F.repeat(x.shape[0], 1, 1) # [batch_size, m, m] identity matrix
    for j in range(1,J_mod+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    if jacobian:
        return torch.bmm(F, x), F
    else:
        return torch.bmm(F, x)

### fInacc will be fed to filters and KNet, note that the mismatch comes from delta_t and rotation
def fRotate(x, jacobian=False):
    BX = torch.zeros([x.shape[0],m,m]).float().to(x.device) #[batch_size, m, m]
    BX[:,1,0] = torch.squeeze(-x[:,2,:]) 
    BX[:,2,0] = torch.squeeze(x[:,1,:])
    Const = C.to(x.device)
    A = torch.add(BX, Const)   
    # Taylor Expansion for F    
    F = torch.eye(m).to(x.device)
    F = F.reshape((1, m, m))
    F = F.repeat(x.shape[0], 1, 1) # [batch_size, m, m] identity matrix
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    F_rotated = torch.bmm(RotMatrix.reshape(1,m,m).repeat(x.shape[0],1,1),F)
    if jacobian:
        return torch.bmm(F_rotated, x), F_rotated
    else:
        return torch.bmm(F_rotated, x)

##################################################
### Observation function h for Lorenz Atractor ###
##################################################
H_design = torch.eye(n)
H_Rotate = torch.mm(RotMatrix,H_design)
H_Rotate_inv = torch.inverse(H_Rotate)

# Original h (not batched)
def Origin_h(x):
    H = H_design.to(x.device)
    y = torch.matmul(H,x)
    return y
    
def Origin_h_nonlinear(x):
    return Origin_toSpherical(x)

def Origin_hRotate(x):
    H = H_Rotate.to(x.device)
    return torch.matmul(H,x)

#########################################################################################
# batched version of h

def h(x, jacobian=False):
    H = H_design.to(x.device).reshape((1, n, n)).repeat(x.shape[0], 1, 1) # [batch_size, n, n] identity matrix   
    y = torch.bmm(H,x)
    if jacobian:
        return y, H
    else:
        return y

def h_nonlinear(x):
    return toSpherical(x)

def hRotate(x, jacobian=False):
    H = H_Rotate.to(x.device).reshape((1, n, n)).repeat(x.shape[0], 1, 1)# [batch_size, n, n] rotated matrix
    if jacobian:
        return torch.bmm(H,x), H
    else:
        return torch.bmm(H,x)


###############################################
### process noise Q and observation noise R ###
###############################################
Q_non_diag = False
R_non_diag = False

Q_structure = torch.eye(m)
R_structure = torch.eye(n)

if(Q_non_diag):
    q_d = 1
    q_nd = 1/2
    Q = torch.tensor([[q_d, q_nd, q_nd],[q_nd, q_d, q_nd],[q_nd, q_nd, q_d]])

if(R_non_diag):
    r_d = 1
    r_nd = 1/2
    R = torch.tensor([[r_d, r_nd, r_nd],[r_nd, r_d, r_nd],[r_nd, r_nd, r_d]])

##################################
### Utils for non-linear cases ###
##################################
# Original version of getJacobian
def Origin_getJacobian(x, g):   
    y = torch.reshape((x.T),[x.size()[0]])
    m = x.size()[0]
    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,m)
    
    return Jac

# FIXME: Currently autograd.functional.jacobian does not support batch version, so we have to use for loop
def getJacobian(x, Origin_g):
    """
    Currently, pytorch does not have a built-in function to compute Jacobian matrix
    in a batched manner, so we have to iterate over the batch dimension.
    
    input x (torch.tensor): [batch_size, m/n, 1]
    input g (function): function to be differentiated
    output Jac (torch.tensor): [batch_size, m, m] for f, [batch_size, n, m] for h
    """
    ### Method 1: using F, H directly (not working due to computation precision)
    # _,Jac = g(x, jacobian=True)

    ### Method 2: using autograd.functional.jacobian and for loop
    batch_size = x.shape[0]
    m = x.shape[1]
    y0 = torch.squeeze(x[0,:,:].T)
    Jac_x0 = torch.squeeze(autograd.functional.jacobian(Origin_g, y0)).view(-1,m)
    Jac = torch.zeros([batch_size, Jac_x0.shape[0], Jac_x0.shape[1]])
    Jac[0,:,:] = Jac_x0
    for i in range(1,batch_size):
        y = torch.squeeze(x[i,:,:].T)
        Jac[i,:,:] = torch.squeeze(autograd.functional.jacobian(Origin_g, y))

    return Jac

# Original version of toSpherical
def Origin_toSpherical(cart):

    rho = torch.norm(cart, p=2).view(1,1)
    phi = torch.atan2(cart[1, ...], cart[0, ...]).view(1, 1)
    phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)

    theta = torch.acos(cart[2, ...] / rho).view(1, 1)

    spher = torch.cat([rho, theta, phi], dim=0)

    return spher

# batched version of toSpherical
def toSpherical(cart):
    """
    input cart (torch.tensor): [batch_size, m, 1] or [batch_size, m]
    output spher (torch.tensor): [batch_size, n, 1]
    """
    rho = torch.linalg.norm(cart,dim=1).reshape(cart.shape[0], 1)# [batch_size, 1]
    phi = torch.atan2(cart[:, 1, ...], cart[:, 0, ...]).reshape(cart.shape[0], 1) # [batch_size, 1]
    phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)
    
    theta = torch.div(torch.squeeze(cart[:, 2, ...]), torch.squeeze(rho))
    theta = torch.acos(theta).reshape(cart.shape[0], 1) # [batch_size, 1]

    spher = torch.cat([rho, theta, phi], dim=1).reshape(cart.shape[0],3,1) # [batch_size, n, 1]

    return spher

def toCartesian(sphe):
    """
    input sphe (torch.tensor): [batch_size, n, 1] or [batch_size, n]
    output cart (torch.tensor): [batch_size, n]
    """
    rho = sphe[:, 0, ...]
    theta = sphe[:, 1, ...]
    phi = sphe[:, 2, ...]

    x = (rho * torch.sin(theta) * torch.cos(phi)).reshape(sphe.shape[0],1)
    y = (rho * torch.sin(theta) * torch.sin(phi)).reshape(sphe.shape[0],1)
    z = (rho * torch.cos(theta)).reshape(sphe.shape[0],1)

    cart = torch.cat([x,y,z],dim=1).reshape(cart.shape[0],3,1) # [batch_size, n, 1]

    return cart