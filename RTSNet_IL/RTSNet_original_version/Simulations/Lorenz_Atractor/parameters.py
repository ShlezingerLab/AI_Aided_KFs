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
### f_gen is for dataset generation
def f_gen(x):
    BX = torch.reshape(torch.matmul(B, x),(m,m))
    A = torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C)  
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t_gen, j)/math.factorial(j))
        F = torch.add(F, F_add)

    return torch.matmul(F, x)

### f will be fed to smoothers & RTSNet, note that the mismatch comes from delta_t
def f(x):
    BX = torch.reshape(torch.matmul(B, x),(m,m))
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C))
    
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    x_out = torch.matmul(F, x)
    return x_out

### fInacc will be fed to smoothers & RTSNet, note that the mismatch comes from delta_t and J_mod
def fInacc(x):
    BX = torch.reshape(torch.matmul(B, x),(m,m))
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C)
    
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J_mod+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)

    return torch.matmul(F, x)

### fInacc will be fed to smoothers & RTSNet, note that the mismatch comes from delta_t and rotation
def fRotate(x):
    BX = torch.reshape(torch.matmul(B, x),(m,m)) 
    A = (torch.add(BX.permute(*torch.arange(BX.ndim - 1, -1, -1)),C))  
    # Taylor Expansion for F    
    F = torch.eye(m)
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    F_rotated = torch.mm(RotMatrix,F)
    return torch.matmul(F_rotated, x)

##################################################
### Observation function h for Lorenz Atractor ###
##################################################
H_design = torch.eye(n)
H_Rotate = torch.mm(RotMatrix,H_design)
H_Rotate_inv = torch.inverse(H_Rotate)

def h(x):
    y = torch.matmul(H_design,x)
    return y

def h_nonlinear(x):
    return torch.squeeze(toSpherical(x))

def hRotate(x):
    return torch.matmul(H_Rotate,x)


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
def getJacobian(x, g):
    # if(x.size()[1] == 1):
    #     y = torch.reshape((x.T),[x.size()[0]])
    
    y = torch.reshape((x.permute(*torch.arange(x.ndim - 1, -1, -1))),[x.size()[0]])

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1,m)
    return Jac

def toSpherical(cart):

    rho = torch.norm(cart, p=2).view(1,1)
    phi = torch.atan2(cart[1, ...], cart[0, ...]).view(1, 1)
    phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)

    theta = torch.acos(cart[2, ...] / rho).view(1, 1)

    spher = torch.cat([rho, theta, phi], dim=0)

    return spher

def toCartesian(sphe):

    rho = sphe[0]
    theta = sphe[1]
    phi = sphe[2]

    x = (rho * torch.sin(theta) * torch.cos(phi)).view(1,-1)
    y = (rho * torch.sin(theta) * torch.sin(phi)).view(1,-1)
    z = (rho * torch.cos(theta)).view(1,-1)

    cart = torch.cat([x,y,z],dim=0)

    return torch.squeeze(cart)