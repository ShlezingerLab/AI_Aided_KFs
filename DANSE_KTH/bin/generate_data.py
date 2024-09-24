#####################################################
# Creator: Anubhab Ghosh 
# Nov 2023
#####################################################
# Importing the necessary libraries
import numpy as np
import sys
from os import path
# __file__ should be defined in this case
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import scipy
from scipy.linalg import block_diag
import torch
from torch import distributions
#import matplotlib.pyplot as plt
from scipy.linalg import expm
from utils.utils import save_dataset
from parameters_opt import get_parameters
from bin.ssm_models import LinearSSM, LorenzSSM, RosslerSSM, Lorenz96SSM
import argparse
from parse import parse
import os

def initialize_model(type_, parameters):

    print(type_)
    if type_ == "LinearSSM":

        model = LinearSSM(n_states=parameters["n_states"],
                        n_obs=parameters["n_obs"],
                        mu_e=parameters["mu_e"],
                        mu_w=parameters["mu_w"],
                        gamma=parameters["gamma"],
                        beta=parameters["beta"],
                        normalize=parameters["normalize"]
                        )

    elif type_ == "LorenzSSM" or type_ == "ChenSSM":

        model = LorenzSSM(
            n_states=parameters["n_states"],
            n_obs=parameters["n_obs"],
            J=parameters["J"],
            delta=parameters["delta"],
            delta_d=parameters["delta_d"],
            alpha=parameters["alpha"],
            decimate=parameters["decimate"],
            mu_e=parameters["mu_e"],
            mu_w=parameters["mu_w"],
            normalize=parameters["normalize"]
            )
       
    elif type_ == "RosslerSSM":

        model = RosslerSSM(
            n_states=parameters["n_states"],
            n_obs=parameters["n_obs"],
            J=parameters["J"],
            delta=parameters["delta"],
            delta_d=parameters["delta_d"],
            a=parameters["a"],
            b=parameters["b"],
            c=parameters["c"],
            decimate=parameters["decimate"],
            mu_e=parameters["mu_e"],
            mu_w=parameters["mu_w"],
            normalize=parameters["normalize"]
        )

    elif type_ == "RosslerSSMn2" or type_ == "RosslerSSMn1" or type_ == "RosslerSSMrn2" or type_ == "RosslerSSMrn3":
        
        model = RosslerSSM(
            n_states=parameters["n_states"],
            n_obs=parameters["n_obs"],
            J=parameters["J"],
            delta=parameters["delta"],
            delta_d=parameters["delta_d"],
            a=parameters["a"],
            b=parameters["b"],
            c=parameters["c"],
            H=parameters["H"],
            decimate=parameters["decimate"],
            mu_e=parameters["mu_e"],
            mu_w=parameters["mu_w"]

        )

    elif type_ == "LorenzSSMn2" or type_ == "LorenzSSMn1" or type_ == "LorenzSSMrn2" or type_ == "LorenzSSMrn3" or type_ == "ChenSSMn2" or type_ == "ChenSSMn1" or type_ == "ChenSSMrn2" or type_== "ChenSSMrn3":

        model = LorenzSSM(
            n_states=parameters["n_states"],
            n_obs=parameters["n_obs"],
            J=parameters["J"],
            delta=parameters["delta"],
            delta_d=parameters["delta_d"],
            alpha=parameters["alpha"],
            H=parameters["H"],
            decimate=parameters["decimate"],
            mu_e=parameters["mu_e"],
            mu_w=parameters["mu_w"],
            normalize=parameters["normalize"]
            )
        
    elif type_ == "Lorenz96SSM" or "Lorenz96SSMn" in type_ or "Lorenz96SSMrn" in type_:

        model = Lorenz96SSM(
            n_states=parameters["n_states"],
            n_obs=parameters["n_obs"],
            delta=parameters["delta"],
            delta_d=parameters["delta_d"],
            F_mu=parameters["F_mu"],
            method=parameters["method"],
            H=parameters["H"],
            decimate=parameters["decimate"],
            mu_w=parameters["mu_w"]
        )
    
    return model

def generate_SSM_data(model, T, sigma_e2_dB, smnr_dB):

    X_arr = np.zeros((T, model.n_states))
    Y_arr = np.zeros((T, model.n_obs))

    X_arr, Y_arr, Cw = model.generate_single_sequence(
                T=T,
                sigma_e2_dB = sigma_e2_dB,
                smnr_dB = smnr_dB
            )
        
    return X_arr, Y_arr, Cw

def generate_state_observation_pairs(type_, parameters, T=200, N_samples=1000, sigma_e2_dB=0.1, smnr_dB=10):

    Z_XY = {}
    Z_XY["num_samples"] = N_samples
    Z_XY_data_lengths = [] 

    #Z_XY_data = []
    Z_X_data = []
    Z_Y_data = []
    Z_Cw_data = []

    ssm_model = initialize_model(type_, parameters)
    Z_XY['ssm_model'] = ssm_model
    
    for i in range(N_samples):
        
        Xi, Yi, Cwi = generate_SSM_data(ssm_model, T, sigma_e2_dB, smnr_dB)
        Z_XY_data_lengths.append(T)
        #Z_XY_data.append([Xi, Yi, Cw])

        Z_X_data.append(Xi)
        Z_Y_data.append(Yi)
        Z_Cw_data.append(Cwi)

    #Z_XY["data"] = np.row_stack(Z_XY_data).astype(object)
    Z_XY["dataX"] = np.asarray(Z_X_data)
    Z_XY["dataY"] = np.asarray(Z_Y_data)
    Z_XY["dataCw"] = np.asarray(Z_Cw_data)
    #Z_pM["data"] = Z_pM_data
    Z_XY["trajectory_lengths"] = np.vstack(Z_XY_data_lengths)

    return Z_XY

def create_filename(T=100, N_samples=200, m=5, n=5, dataset_basepath="./data/", type_="LinearSSM", sigma_e2_dB=-10, smnr_dB=10, normalize=False):
    # Create the dataset based on the dataset parameters
    
    if normalize == False:
        datafile = "trajectories_m_{}_n_{}_{}_data_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB.pkl".format(m, n, type_, int(T), int(N_samples), sigma_e2_dB, smnr_dB)
    else:
        datafile = "trajectories_normalized_m_{}_n_{}_{}_data_T_{}_N_{}_sigmae2_{}dB_smnr_{}dB.pkl".format(m, n, type_, int(T), int(N_samples), sigma_e2_dB, smnr_dB)
    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath

def create_and_save_dataset(T, N_samples, filename, parameters, type_="LinearSSM", sigma_e2_dB=0.1, smnr_dB=10):

    #NOTE: Generates for pfixed theta estimation experiment
    # Currently this uses the 'modified' function
    #Z_pM = generate_trajectory_modified_param_pairs(N=N, 
    #                                                M=num_trajs, 
    #                                                P=num_realizations, 
    #                                                usenorm_flag=usenorm_flag)
    #np.random.seed(10) # This can be kept at a fixed step for being consistent
    Z_XY = generate_state_observation_pairs(type_=type_, parameters=parameters, T=T, N_samples=N_samples, sigma_e2_dB=sigma_e2_dB, smnr_dB=smnr_dB)
    save_dataset(Z_XY, filename=filename)

if __name__ == "__main__":
    
    usage = "Create datasets by simulating state space models \n"\
            "python generate_data.py --sequence_length T --num_samples N --dataset_type [LinearSSM/LorenzSSM] --output_path [output path name]\n"\
            "Creates the dataset at the location output_path"\
        
    parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for training RNNs")
    

    parser.add_argument("--n_states", help="denotes the number of states in the latent model", type=int, default=5)
    parser.add_argument("--n_obs", help="denotes the number of observations", type=int, default=5)
    parser.add_argument("--num_samples", help="denotes the number of trajectories to be simulated for each realization", type=int, default=500)
    parser.add_argument("--sequence_length", help="denotes the length of each trajectory", type=int, default=200)
    parser.add_argument("--sigma_e2_dB", help="denotes the process noise variance in dB", type=float, default=-10.0)
    parser.add_argument("--smnr_dB", help="denotes the signal-to-measurement noise in dB", type=float, default=20.0)
    parser.add_argument("--dataset_type", help="specify type of the SSM (LinearSSM / LorenzSSM)", type=str, default=None)
    parser.add_argument("--normalize", help="specify type of the SSM (LinearSSM / LorenzSSM)", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default=None)
    
    args = parser.parse_args() 

    n_states = args.n_states
    n_obs = args.n_obs
    T = args.sequence_length
    N_samples = args.num_samples
    type_ = args.dataset_type
    output_path = args.output_path
    sigma_e2_dB = args.sigma_e2_dB
    smnr_dB = args.smnr_dB
    normalize = args.normalize

    # Create the full path for the datafile
    datafilename = create_filename(T=T, N_samples=N_samples, m=n_states, n=n_obs, dataset_basepath=output_path, type_=type_, sigma_e2_dB=sigma_e2_dB, smnr_dB=smnr_dB, normalize=normalize)
    ssm_parameters, _ = get_parameters(n_states=n_states, n_obs=n_obs)
    
    if not "LinearSSM" in type_:
        if ssm_parameters[type_]["decimate"] == True:
            decimation_factor = int(ssm_parameters[type_]["delta"] / ssm_parameters[type_]["delta_d"])
        else:
            decimation_factor = int(1.0)
            print("Simulating for T_reqd: {}, with a decimation factor: {}/{} = {}".format(int(T * decimation_factor), ssm_parameters[type_]["delta"], 
                                                                                   ssm_parameters[type_]["delta_d"], decimation_factor))
    else:
            decimation_factor = int(1.0)

    T_reqd = int(T * decimation_factor)

    #print(normalize)
    if normalize == True:
        print("Applying normalization of state sequence ...")
        ssm_parameters[type_]["normalize"] = True
    else:
        print("No normalization applied")

    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(datafilename):
        print("Creating the data file: {}".format(datafilename))
        create_and_save_dataset(T=T_reqd, N_samples=N_samples, filename=datafilename, type_=type_, parameters=ssm_parameters[type_], sigma_e2_dB=sigma_e2_dB, smnr_dB=smnr_dB)
    else:
        print("Dataset {} is already present!".format(datafilename))
    
    print("Done...")
