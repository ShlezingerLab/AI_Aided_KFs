#####################################################
# Creator: Anubhab Ghosh 
# Nov 2023
#####################################################
# This function is used to define the parameters of the model
import numpy as np
import math
import torch
from utils.utils import dB_to_lin, partial_corrupt
from bin.ssm_models import LinearSSM
import torch
import scipy    
from scipy.linalg import block_diag
from torch.autograd.functional import jacobian

torch.manual_seed(10)
delta_t = 0.02 # Hardcoded for now
delta_t_chen = 0.002 # Hardcoded for now
decimation_factor_lorenz = 1
decimation_factor_chen = 5
decimation_factor_rossler = 10
delta_t_L96 = 0.01
delta_t_rossler = 0.008 # Hardcoded for now
J_gen = 5 
J_test = 5 # hardcoded for now

H_rn = np.array([[ 0.3799163 ,  0.34098765,  1.04316576],
                [ 0.98069622, -0.70476889,  2.17907879],
                [-1.03118098,  0.97651857, -0.59419465]])

H_id = np.eye(3)

H_rn_20_20 = np.array([[-1.80839597e+00,  9.54086162e-01,  1.15780153e-01,
        -1.98184986e+00,  1.24186787e+00, -5.79884026e-01,
         1.74444511e-01, -1.20828909e+00,  8.20536823e-01,
        -8.70868919e-01, -2.29270728e-02,  2.87945729e-01,
        -4.48162548e-01, -2.28350871e-01,  7.39143674e-01,
        -3.06205114e-01,  1.78613663e+00, -1.46524863e+00,
        -8.99077907e-01, -6.38235215e-01],
       [ 2.67924432e-01,  6.37061889e-01, -2.52784324e-01,
        -5.88124419e-01, -5.84615248e-01,  1.86183870e-01,
         3.70377571e-01,  1.11994730e-03,  9.63306229e-02,
         5.84316866e-01,  5.20251191e-01, -6.95070161e-02,
        -1.34669327e-01, -4.23942653e-01,  1.12535985e+00,
         1.09402977e-01, -2.96315561e-01, -5.51709729e-01,
         1.89593868e-01, -5.54478552e-01],
       [-7.07424955e-01,  1.18615514e+00, -4.70141213e-01,
        -9.01082487e-01, -6.51704053e-01,  3.17174293e+00,
         2.57374260e+00, -2.79420935e-01, -5.23524140e-01,
        -1.16714717e+00,  2.15736956e-01,  1.48290350e+00,
        -1.80585394e+00, -1.22170291e-01,  6.86643848e-01,
         1.13040013e+00, -1.48807855e-01,  4.53074901e-01,
        -1.12651611e+00, -4.12459131e-01],
       [ 9.66751078e-02, -7.64318627e-01,  5.19096926e-01,
         2.18917775e-01,  5.01018691e-02, -1.04655259e+00,
         1.23104470e+00,  1.64399263e-01, -3.05563988e-01,
        -1.76810130e+00,  6.58399958e-01, -1.62627194e+00,
         8.33849186e-01, -1.84821356e+00, -7.97169619e-01,
         1.34974496e-02,  1.98782886e+00,  1.59210765e+00,
         2.69776149e-01,  1.11732884e-01],
       [-1.04247687e+00,  1.38796130e-01, -1.17174650e-01,
         1.73415348e+00, -9.56872307e-01, -7.75715581e-02,
         6.77069831e-03,  5.02676542e-01,  1.13298782e+00,
        -2.80055274e-01,  5.86672706e-01, -7.00485655e-01,
        -1.06464846e+00,  1.50588385e+00, -3.84231661e-01,
         1.27733366e+00, -4.73367580e-01,  7.74302426e-01,
         2.35680762e-01, -8.46232762e-01],
       [-5.20026514e-01,  1.33933517e+00,  2.51942555e-01,
        -1.49456834e-02, -6.33885061e-01,  6.50875279e-01,
         9.54894354e-02,  1.42522319e+00, -3.21450852e-01,
        -2.55295799e+00, -6.54504332e-01,  2.80018463e-02,
        -1.32555623e+00,  1.22490797e+00,  3.74457387e-01,
         2.20985256e-01,  8.56890851e-02,  4.91187828e-01,
         2.73830852e-01, -1.35868857e+00],
       [ 2.64721787e-01, -1.90695035e-01, -9.64518487e-01,
        -6.56602744e-04,  2.17372981e-01, -1.22071750e+00,
         1.04723224e-01, -4.55953955e-01, -6.86607952e-01,
         9.46618911e-01, -6.99275355e-01,  2.31414481e-01,
        -1.86534237e+00, -1.90480891e+00, -1.11101444e+00,
         5.26739492e-01,  2.24471141e-01, -2.82052581e-01,
        -6.54325922e-02, -3.03825823e-01],
       [ 6.46446788e-01,  7.32055124e-02, -9.00325139e-01,
         1.31853639e+00,  1.35865710e+00,  3.55043608e-01,
         1.28046341e+00, -7.45339527e-02, -6.95821972e-01,
        -1.19538164e+00,  2.26481646e+00,  1.18685729e+00,
         6.58048690e-01, -1.20197272e+00, -8.68686862e-01,
        -8.97492589e-01,  2.33583241e-01, -2.31293440e+00,
         2.02791181e-01, -1.29353104e-01],
       [ 4.14166060e-02,  2.35318106e+00, -9.90300592e-01,
         2.01021987e-01, -7.28247668e-01, -7.66280959e-01,
         1.92010618e+00, -4.17112576e-01,  5.22990033e-01,
         6.93603206e-01, -9.19696732e-01,  9.36186819e-02,
        -2.67423389e-02,  9.75635033e-01,  2.08065377e+00,
        -2.79054859e+00,  1.88419120e+00, -1.24870074e+00,
         4.66497746e-01,  1.78678170e+00],
       [ 8.97291120e-01, -1.57885598e+00, -1.44696858e+00,
         8.02025226e-01, -3.80527478e-01, -8.72898618e-01,
        -2.69780417e-01, -5.98437250e-01,  1.75633895e-01,
         7.02508787e-01,  1.50033743e+00, -4.03702130e-01,
         1.38895927e+00,  2.48551661e-01, -9.00965575e-01,
        -5.86860308e-01, -1.50682544e+00, -1.70117873e+00,
        -1.22524131e+00,  5.51711287e-03],
       [ 1.59542166e+00, -4.59219873e-01, -5.08456982e-01,
         3.36715300e-01, -5.41868248e-01, -2.18566244e+00,
         8.87690059e-01, -2.38852932e+00,  1.03209471e-01,
         1.65846804e+00,  1.28782296e-01,  1.13595560e+00,
        -3.23978508e+00, -1.44801465e-01,  4.90656166e-01,
        -8.71787528e-01, -1.34387641e+00,  4.20014324e-01,
         1.88730139e+00,  1.27416225e-01],
       [ 5.49649886e-01, -2.10344540e-01,  2.14335263e-01,
         2.03291625e-02, -3.72713395e-01,  1.23964942e+00,
         1.79347764e+00, -7.56474566e-02, -1.47738439e+00,
         2.44404350e+00, -7.91038638e-01,  1.10008700e-01,
        -1.02076056e+00, -2.42978607e+00, -1.97862827e+00,
         4.67049646e-01,  3.15824202e-01, -3.91974257e-01,
        -1.26283585e+00,  1.07626513e+00],
       [-1.23537991e-01,  1.27630650e+00, -1.03877481e+00,
        -7.24583437e-01, -1.97048054e+00,  8.67305746e-01,
         1.69483812e-01,  2.55696360e-02,  1.58319002e+00,
        -5.92978668e-01, -1.93906968e-01, -3.20137785e-01,
        -7.63835259e-01,  4.42182131e-01, -2.83013025e-01,
        -4.58365883e-01,  1.34385075e+00,  6.19730917e-02,
         6.78106713e-01,  8.27517683e-01],
       [ 6.04382902e-01,  6.50595593e-03,  2.70894132e-01,
         1.66602273e-01,  3.73374557e-01,  5.89971292e-02,
        -2.02209902e+00,  3.10580164e-01,  5.77673014e-03,
        -9.46082851e-02,  1.55420511e+00, -1.50789914e+00,
        -9.53386299e-01,  9.45487264e-01,  3.45069656e-01,
        -7.00937371e-01,  1.47490799e-01, -4.76779668e-01,
        -6.07128319e-01,  9.18641103e-01],
       [ 8.54294257e-01,  3.48310364e-02,  1.35961719e+00,
         1.04670449e+00, -5.99020279e-01, -1.09291359e+00,
        -1.25890447e+00, -1.11832870e-01, -1.22254178e+00,
        -1.15023294e-01, -3.05130787e-01,  2.40274635e-01,
        -3.23658831e-01,  1.20932595e-01, -1.50599003e+00,
         2.02633880e+00, -6.93528715e-01, -2.78550663e-01,
         1.35173959e+00, -7.61770510e-01],
       [ 1.72460942e+00, -3.13955228e-01,  7.82154613e-01,
         7.37895703e-01, -9.95875129e-02,  1.27909214e+00,
         9.11679984e-01,  1.60492759e+00,  2.58914507e+00,
        -5.49363117e-01,  1.74442884e-01,  7.51757003e-01,
        -7.56630226e-01,  1.29481912e+00, -9.62243769e-02,
         9.52710543e-01, -5.04617744e-01, -5.13256063e-01,
        -3.44200126e-02,  2.25734855e-01],
       [-1.10671457e+00, -6.43546026e-01,  4.85990895e-01,
         9.24615264e-01, -1.04585044e+00,  9.07796350e-01,
        -1.31697589e+00, -7.01975666e-01, -9.56329141e-01,
         2.20677320e+00,  7.34845509e-01,  8.90693284e-01,
        -3.43787852e-01, -6.06951650e-02, -8.69466046e-01,
        -1.45503902e+00,  5.58977006e-01, -2.64683082e+00,
         1.46503352e+00,  1.66950958e-01],
       [ 4.15378049e-01,  1.34138744e+00,  4.19449830e-01,
        -1.24841660e-01, -6.75984021e-01,  3.31944632e-01,
        -1.79949987e-01, -3.58642470e-02,  1.59325101e+00,
        -1.20962282e+00, -1.12878072e-01, -4.38700438e-01,
        -4.65562710e-01, -4.05302578e-01, -8.40684340e-01,
        -2.14458470e-01, -4.69208500e-01, -1.70038999e+00,
        -1.50764061e-01, -8.03490063e-01],
       [-5.69713637e-01, -3.30474314e-01, -1.17934817e+00,
         8.55767909e-02,  1.01484648e+00,  8.06233508e-01,
         7.61618173e-01,  2.43196718e+00, -3.54369833e-01,
         4.74625130e-01, -5.49119689e-01,  4.51745401e-01,
        -1.44742855e+00,  3.60227727e-01,  1.59668436e+00,
        -6.40117118e-01, -1.21054958e+00, -2.00667312e-02,
         6.43599267e-01, -1.68416825e-01],
       [-7.97530330e-01, -4.16408081e-01, -1.57514536e+00,
        -6.23953175e-01, -3.41538206e-01,  3.68054286e-01,
        -2.20287143e+00,  1.83003136e+00,  3.11828928e-01,
        -9.31687547e-01, -6.56176021e-01,  2.96258208e-01,
         2.11172560e-01,  1.25437528e+00, -4.38257355e-01,
        -3.12681550e-02, -8.50991038e-01, -2.21364975e-01,
        -1.30495474e-01,  8.46948763e-01]])

def A_fn(z):
    return np.array([
                    [-10, 10, 0],
                    [28, -1, -z],
                    [0, z, -8.0/3]
                ])

def h_fn(z):
    return z

"""
# The KalmanNet implementation
def f_lorenz(x):

    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    #A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) #(torch.add(torch.reshape(torch.matmul(B, x),(3,3)).T,C))
    A = (torch.add(torch.reshape(torch.matmul(B, x),(3,3)).T,C))
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)
"""
def f_lorenz(x):

    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) + C
    #delta_t = 0.02 # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)

def f_lorenz_ukf(x, dt):

    x = torch.from_numpy(x).type(torch.FloatTensor)
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-10, 10,    0],
                    [ 28, -1,    0],
                    [  0,  0, -8/3]]).type(torch.FloatTensor)
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) 
    #A = torch.reshape(torch.matmul(B, x),(3,3)).T # For KalmanNet
    A += C
    #delta = delta_t # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x).numpy()

def f_chen(x):

    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0, 1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-35, 35,    0],
                    [ -7, 28,    0],
                    [  0,  0, -9/3]]).type(torch.FloatTensor)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) + C
    #delta_t = 0.02 # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x)

def f_chen_ukf(x, dt):

    x = torch.from_numpy(x).type(torch.FloatTensor)
    B = torch.Tensor([[[0,  0, 0],[0, 0, -1],[0,  1, 0]], torch.zeros(3,3), torch.zeros(3,3)]).type(torch.FloatTensor)
    C = torch.Tensor([[-35, 35,    0],
                    [ -7, 28,    0],
                    [  0,  0, -9/3]]).type(torch.FloatTensor)
    #A = torch.add(torch.einsum('nhw,wa->nh', B, x).T,C)
    A = torch.einsum('kn,nij->ij',x.reshape((1,-1)),B) 
    #A = torch.reshape(torch.matmul(B, x),(3,3)).T # For KalmanNet
    A += C
    #delta = delta_t # Hardcoded for now
    # Taylor Expansion for F    
    F = torch.eye(3)
    J = J_test # Hardcoded for now
    for j in range(1,J+1):
        F_add = (torch.matrix_power(A*delta_t, j)/math.factorial(j))
        F = torch.add(F, F_add)
    return torch.matmul(F, x).numpy()

def A_fn_rossler(z): 

    a = 0.2 
    b = 0.2 
    c = 5.7
    return torch.Tensor([
        [0.0, -1.0, -1.0],
        [1.0, a, 0.0],
        [0.0, 0.0, (b / z[2]) + (z[0] - c)]
    ]).type(torch.FloatTensor)

def f_rossler(x):

    F = torch.eye(3)
    for j in range(1, J_test+1):
        F_add = torch.matrix_power(A_fn_rossler(x)*delta_t_rossler, j) / math.factorial(j)
        F = torch.add(F, F_add)

    return torch.matmul(F, x)

def f_rossler_ukf(x, dt):

    F = torch.eye(3)
    for j in range(1, J_test+1):
        F_add = torch.matrix_power(A_fn_rossler(x)*delta_t_rossler, j) / math.factorial(j)
        F = torch.add(F, F_add)

    return torch.matmul(F, x).numpy()

def f_sinssm_fn(z, alpha=0.9, beta=1.1, phi=0.1*math.pi, delta=0.01):
    return alpha * torch.sin(beta * z + phi) + delta

def h_sinssm_fn(z, a=1, b=1, c=0):
    return a * (b * z + c)

def get_H_DANSE(type_, n_states, n_obs):
    if type_ == "LinearSSM":
        return LinearSSM(n_states=n_states, n_obs=n_obs).H
    elif type_ == "LorenzSSM" or type_ == "ChenSSM" or type_ == "RosslerSSM" or type_ == "Lorenz96SSM":
        return np.eye(n_obs, n_states)
    elif type_ == "LorenzSSMn2" or type_ == "ChenSSMn2" or type_ == "RosslerSSMn2":
        return H_id[1:, :]
    elif type_ == "LorenzSSMn1" or type_ == "ChenSSMn1" or type_ == "RosslerSSMn1":
        return H_id[0, :].reshape((1,-1))
    elif type_ == "LorenzSSMrn3" or type_ == "ChenSSMrn3" or type_ == "RosslerSSMrn3":
        return H_rn
    elif type_ == "LorenzSSMrn2" or type_ == "ChenSSMrn2" or type_ == "RosslerSSMrn2":
        return H_rn[0:2,:]
    elif type_ == "LorenzSSMrn1" or type_ == "ChenSSMrn1" or type_ == "RosslerSSMrn1":
        return H_rn[0,:]
    elif type_ == "Lorenz96SSMn{}".format(n_obs):
        return np.concatenate((np.eye(n_obs), np.zeros((n_obs,n_states-n_obs))), axis=1)
    elif type_ == "Lorenz96SSMrn{}".format(n_obs):
        return H_rn_20_20[:n_obs, :]
    elif type_ == "SinusoidalSSM":
        return jacobian(h_sinssm_fn, torch.randn(n_states,)).numpy()

def get_parameters(n_states=5, n_obs=5, device='cpu'):

    ssm_parameters_dict = {
        # Parameters of the linear model 
        "LinearSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "gamma":0.8,
            "beta":1.0,
            "normalize":False
        },
        # Parameters of the Lorenz Attractor model
        "LorenzSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t,
            "alpha":0.0, # alpha = 0.0, implies a Lorenz model
            "H":None, # By default, H is initialized to an identity matrix
            "delta_d":delta_t / decimation_factor_lorenz,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "ChenSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t_chen,
            "alpha":1.0, # alpha = 0.0, implies a Lorenz model
            "H":None, # By default, H is initialized to an identity matrix
            "delta_d":delta_t_chen / decimation_factor_chen,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "ChenSSMrn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t_chen,
            "alpha":1.0, # alpha = 0.0, implies a Lorenz model
            "H":get_H_DANSE(type_="ChenSSMrn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t_chen / decimation_factor_chen,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "ChenSSMn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t_chen,
            "alpha":1.0, # alpha = 0.0, implies a Lorenz model
            "H":get_H_DANSE(type_="ChenSSMn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t_chen / decimation_factor_chen,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "LorenzSSMrn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t,
            "alpha":0.0, # alpha = 0.0, implies a Lorenz model
            "H":get_H_DANSE(type_="LorenzSSMrn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "LorenzSSMn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t,
            "alpha":0.0, # alpha = 0.0, implies a Lorenz model
            "H":get_H_DANSE(type_="LorenzSSMn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "RosslerSSM".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t_rossler,
            "a":0.2, "b":0.2, "c":5.7,
            "H":None, # By default, H is initialized to an identity matrix
            "delta_d":delta_t_rossler / decimation_factor_rossler,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "RosslerSSMrn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t_rossler,
            "a":0.2, "b":0.2, "c":5.7,
            "H":get_H_DANSE(type_="RosslerSSMrn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t_rossler / decimation_factor_rossler,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "RosslerSSMn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":J_gen,
            "delta":delta_t_rossler,
            "a":0.2, "b":0.2, "c":5.7,
            "H":get_H_DANSE(type_="RosslerSSMn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t_rossler / decimation_factor_rossler,
            "decimate":True,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "use_Taylor":True,
            "normalize":False
        },
        "Lorenz96SSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "delta":delta_t_L96,
            "H":None, # By default, H is initialized to an identity matrix
            "delta_d":delta_t_L96 / 2,
            "decimate":False,
            "mu_w":np.zeros((n_obs,)),
            "method":'RK45',
            "F_mu":8.0
        },
        "Lorenz96SSMn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "delta":delta_t_L96,
            "H":get_H_DANSE(type_="Lorenz96SSMn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t_L96 / 2,
            "decimate":False,
            "mu_w":np.zeros((n_obs,)),
            "method":'RK45',
            "F_mu":8.0
        },
        "Lorenz96SSMrn{}".format(n_obs):{
            "n_states":n_states,
            "n_obs":n_obs,
            "delta":delta_t_L96,
            "H":get_H_DANSE(type_="Lorenz96SSMrn{}".format(n_obs), n_states=n_states, n_obs=n_obs), # By default, H is initialized to an identity matrix
            "delta_d":delta_t_L96 / 2,
            "decimate":False,
            "mu_w":np.zeros((n_obs,)),
            "method":'RK45',
            "F_mu":8.0
        },
    }

    estimators_dict={
        # Parameters of the DANSE estimator
        "danse":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,)),
            "C_w":None,
            "H":None,
            "mu_x0":np.zeros((n_states,)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":30,
                    "n_layers":1,
                    "lr":1e-2,
                    "num_epochs":2000,
                    "min_delta":5e-2,
                    "n_hidden_dense":32,
                    "device":device
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                }
            }
        },
        "danse_supervised":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,)),
            "C_w":None,
            "H":None,
            "mu_x0":np.zeros((n_states,)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":30,
                    "n_layers":1,
                    "lr":1e-3,
                    "num_epochs":2000,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                }
            }
        },
        "danse_semisupervised":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,)),
            "C_w":None,
            "kappa":0.10,
            "H":None,
            "mu_x0":np.zeros((n_states,)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":30,
                    "n_layers":1,
                    "lr":5e-4,
                    "num_epochs":2000,
                    "min_delta":2e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "n_hidden_dense":32,
                    "device":device
                }
            }
        },
        # Parameters of the Model-based filters - KF, EKF, UKF
        "KF":{
            "n_states":n_states,
            "n_obs":n_obs
        },
        "EKF":{
            "n_states":n_states,
            "n_obs":n_obs
        },
        "UKF":{
            "n_states":n_states,
            "n_obs":n_obs,
            "n_sigma":n_states*2,
            "kappa":0.0,
            "alpha":1e-3
        },
        "KNetUoffline":{
            "n_states":n_states,
            "n_obs":n_obs,
            "n_layers":1,
            "N_E":10_0,
            "N_CV":100,
            "N_T":10_0,
            "unsupervised":True,
            "data_file_specification":'Ratio_{}---R_{}---T_{}',
            "model_file_specification":'Ratio_{}---R_{}---T_{}---unsupervised_{}',
            "nu_dB":0.0,
            "lr":5e-4,
            "weight_decay":1e-6,
            "num_epochs":200,
            "batch_size":100,
            "device":device
        },
        "dmm":{
            "obs_dim":n_obs,  # Dimension of the observation / input to RNN
            "latent_dim":n_states, # Dimension of the latent state / output of RNN in case of state estimation
            "use_mean_field_q":False, # Flag to indicate the use of mean-field q(x_{1:T} \vert y_{1:T})
            "batch_size":128, # Batch size for training
            "rnn_model_type":'gru', # Sets the type of RNN
            "inference_mode":'st-l', # String to indicate the type of DMM inference mode (typically, we will use ST-L or MF-L)
            "combiner_dim":40, # Dimension of hidden layer of combiner network
            "train_emission":False, # Flag to indicate if emission network needs to be learned (True) or not (False)
            "H":None, # Measurement matrix, in case of nontrainable emission network with linear measurement
            "C_w":None, # Measurmenet noise cov. matrix, in case of nontrainable emission network with linear measurements
            "emission_dim":40, # Dimension of hidden layer for emission network
            "emission_num_layers":1, # No. of hidden layers for emission network
            "emission_use_binary_obs":False, # Flag to indicate the modeling of binary observations or not
            "train_transition":True, # Flag to indicate if transition network needs to be learned (True) or not (False)
            "transition_dim":60, # Dimension of hidden layer for transition network
            "transition_num_layers":2, # No. of hidden layers for transition network
            "train_initials":False, # Set if the initial states also are learned uring the optimization 
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru", # Type of RNN used (GRU / LSTM / RNN)
                    "input_size":n_obs, # Input size of the RNN
                    "output_size":n_states, # Output size of the RNN
                    "batch_first":True, # Flag to indicate the input tensor is of the form (batch_size x seq_length x input_dim)
                    "bias":True, # Use bias in RNNs
                    "n_hidden":40, # Dimension of the RNN latent space
                    "n_layers":1, # No. of RNN layers
                    "bidirectional":False, # In case of using DKS say then set this to True
                    "dropout":0.0, # In case of using RNN dropout
                    "device":device # Device to set the RNN
                },
            },
            "optimizer_params": {
                "type": "Adam",
                "args":{
                    "lr": 5e-4, # Learning rate 
                    "weight_decay": 0.0, # Weight decay 
                    "amsgrad": True, # AMS Grad mode to be used for RNN
                    "betas": [0.9, 0.999] # Betas for Adam
                },
                "num_epochs":3000, # Number of epochs
                "min_delta":5e-3, # Sets the delta to control the early stopping
            },
        }
    }

    return ssm_parameters_dict, estimators_dict
