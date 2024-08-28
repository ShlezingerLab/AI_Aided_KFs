#####################################################
# Creator: Anubhab Ghosh
# Feb 2023
#####################################################
# Import necessary libraries
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import argparse
from parse import parse
import numpy as np
import json
from utils.utils import NDArrayEncoder
import scipy

# import matplotlib.pyplot as plt
import torch
import pickle as pkl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from utils.utils import check_if_dir_or_file_exists, get_dataloaders, push_model

# Import the parameters
from parameters_opt import get_parameters, get_H_DANSE
# from utils.plot_functions import plot_measurement_data, plot_measurement_data_axes, plot_state_trajectory, plot_state_trajectory_axes

# Import estimator model and functions
from timeit import default_timer as timer
from src.danse import DANSE, train_danse  # , test_danse
from rtsnet_params import Q_structure, R_structure


def test_danse_ssm(
    Y_test,
    n_states,
    n_obs,
    Cw,
    H,
    model_file_saved_danse,
    Cw_test=None,
    rnn_type="gru",
    device="cpu",
):
    # Initialize the DANSE model parameters
    ssm_dict, est_dict = get_parameters(n_states=n_states, n_obs=n_obs, device=device)

    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=n_states,
        n_obs=n_obs,
        mu_w=np.zeros((n_states,)),
        C_w=Cw,
        batch_size=1,
        H=H,
        mu_x0=np.zeros((n_states,)),
        C_x0=np.eye(n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict["danse"]["rnn_params_dict"],
        device=device,
    )

    print("DANSE Model file: {}".format(model_file_saved_danse))

    start_time_danse = timer()
    danse_model.load_state_dict(torch.load(model_file_saved_danse, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():
        Y_test_batch = (
            Variable(Y_test, requires_grad=False).type(torch.FloatTensor).to(device)
        )
        Cw_test_batch = (
            Variable(Cw_test, requires_grad=False).type(torch.FloatTensor).to(device)
        )
        (
            X_estimated_pred,
            Pk_estimated_pred,
            X_estimated_filtered,
            Pk_estimated_filtered,
        ) = danse_model.compute_predictions(Y_test_batch, Cw_test_batch)

    time_elapsed_danse = timer() - start_time_danse

    return (
        X_estimated_pred,
        Pk_estimated_pred,
        X_estimated_filtered,
        Pk_estimated_filtered,
        time_elapsed_danse,
    )


class Series_Dataset_simplified(Dataset):
    def __init__(self, Z_XY_dict):
        self.data_dict = Z_XY_dict
        self.trajectory_lengths = Z_XY_dict["trajectory_lengths"]

    def __len__(self):
        return len(self.data_dict["dataX"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "noise_covs": np.expand_dims(self.data_dict["dataCw"][idx], axis=0),
            "measurements": np.expand_dims(self.data_dict["dataY"][idx], axis=0),
            "states": np.expand_dims(self.data_dict["dataX"][idx], axis=0),
        }

        return sample


def main():
    usage = (
        "Train DANSE using trajectories of SSMs \n"
        "python3.8 main_danse.py --mode [train/test] --model_type [gru/lstm/rnn] --dataset_mode [LinearSSM/LorenzSSM] \n"
        "--datafile [fullpath to datafile] --splits [fullpath to splits file]"
    )

    parser = argparse.ArgumentParser(
        description="Input a string indicating the mode of the script \n"
        "train - training and testing is done, test-only evlaution is carried out"
    )
    parser.add_argument("--mode", help="Enter the desired mode", type=str)
    parser.add_argument(
        "--rnn_model_type", help="Enter the desired model (rnn/lstm/gru)", type=str
    )
    parser.add_argument(
        "--dataset_type", help="Enter the type of dataset (pfixed/vars/all)", type=str
    )
    parser.add_argument(
        "--model_file_saved",
        help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--datafile", help="Enter the full path to the dataset", type=str
    )

    args = parser.parse_args()
    mode = args.mode
    model_type = args.rnn_model_type
    datafile = args.datafile
    dataset_type = args.dataset_type
    model_file_saved = args.model_file_saved

    print("datafile: {}".format(datafile))
    print(datafile.split("/")[-1])

    # Initialize parameters
    n_states = 3
    n_obs = 3
    r = torch.Tensor([1.0]).type(torch.FloatTensor)
    lambda_q = torch.Tensor([0.3873]).type(torch.FloatTensor)
    print("1/r2 [dB]: ", 10 * torch.log10(1 / r[0] ** 2))
    print("Search 1/q2 [dB]: ", 10 * torch.log10(1 / lambda_q[0] ** 2))
    q2_dB = 10 * torch.log10(lambda_q[0] ** 2)
    r2_dB = 10 * torch.log10(r[0] ** 2)
    Q = (lambda_q[0] ** 2) * Q_structure
    R = (r[0] ** 2) * R_structure

    ngpu = 1  # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )
    print("Device Used:{}".format(device))

    ssm_parameters_dict, est_parameters_dict = get_parameters(
        n_states=n_states, n_obs=n_obs, device=device
    )

    batch_size = est_parameters_dict["danse"]["batch_size"]  # Set the batch size
    estimator_options = est_parameters_dict[
        "danse"
    ]  # Get the options for the estimator

    if not os.path.isfile(datafile):
        print("Dataset is not present, please ensure right dataset is present!!")
        # plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:
        print("Dataset already present!")
        [
            train_input,
            train_target,
            cv_input_long,
            cv_target_long,
            test_input,
            test_target,
        ] = torch.load(datafile)

    # Create the Z_XY `dict' manually
    T = train_input.shape[-1]  # loaded shape is (batch size, dim, seq. length)

    # Calculate SMNR
    SMNR_train_dB = np.mean(10*np.log10(np.var(train_input.permute(0,2,1).numpy(), axis=(1,2)) / r.numpy()))
    SMNR_val_dB = np.mean(10*np.log10(np.var(cv_input_long.permute(0,2,1).numpy(), axis=(1,2)) / r.numpy()))
    SMNR_test_dB = np.mean(10*np.log10(np.var(test_input.permute(0,2,1).numpy(), axis=(1,2)) / r.numpy()))
    print("Computed SMNR - Train: {} dB, Val: {} dB, Test: {} dB".format(SMNR_train_dB, SMNR_val_dB, SMNR_test_dB))

    # Creating list of training, cross-avlidation and test indices
    tr_indices = np.array([i for i in range(train_input.shape[0])])
    val_indices = np.array([i for i in range(cv_input_long.shape[0])]) + len(tr_indices)
    test_indices = (
        np.array([i for i in range(test_input.shape[0])])
        + len(tr_indices)
        + len(val_indices)
    )
    
    tr_Cw = R.unsqueeze(0).repeat(train_input.shape[0],1,1)
    val_Cw = R.unsqueeze(0).repeat(cv_input_long.shape[0],1,1)
    test_Cw = R.unsqueeze(0).repeat(test_input.shape[0],1,1)

    Z_XY = {}
    Z_XY["dataY"] = np.concatenate(
        (
            train_input.transpose(1, 2).numpy(),
            cv_input_long.transpose(1, 2).numpy(),
            test_input.transpose(1, 2).numpy(),
        ),
        axis=0,
    )
    Z_XY["dataX"] = np.concatenate(
        (
            train_target.transpose(1, 2).numpy(),
            cv_target_long.transpose(1, 2).numpy(),
            test_target.transpose(1, 2).numpy(),
        ),
        axis=0,
    )
    Z_XY["dataCw"] = np.concatenate(
        (tr_Cw.numpy(), val_Cw.numpy(), test_Cw.numpy()), axis=0
    )
    Z_XY["trajectory_lengths"] = [
        Z_XY["dataY"].shape[1] for i in range(Z_XY["dataY"].shape[0])
    ]

    print(tr_indices, val_indices, test_indices, Z_XY["dataY"].shape, Z_XY["dataX"].shape, Z_XY["dataCw"].shape)
    N_samples = Z_XY["dataX"].shape[0]

    Z_XY_dataset = Series_Dataset_simplified(Z_XY_dict=Z_XY)
    estimator_options["C_w"] = (
        R.numpy()
    )  # Get the covariance matrix of the measurement noise from the model information
    estimator_options["H"] = get_H_DANSE(
        type_=dataset_type, n_states=n_states, n_obs=n_obs
    )  # Get the sensing matrix from the model info

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=Z_XY_dataset,
        batch_size=batch_size,
        tr_indices=tr_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    print(
        "No. of training, validation and testing batches: {}, {}, {}".format(
            len(train_loader), len(val_loader), len(test_loader)
        )
    )

    # ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    # print("Device Used:{}".format(device))

    logfile_path = "./log/"
    modelfile_path = "./models/"

    # NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_danse_opt_{}_m_{}_n_{}_T_{}_N_{}_q2_{:.1f}dB_r2_{:.1f}dB_run2".format(
        dataset_type, model_type, n_states, n_obs, T, N_samples, q2_dB, r2_dB
    )

    # print(params)
    tr_log_file_name = "training.log"
    te_log_file_name = "testing.log"

    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(
        os.path.join(logfile_path, main_exp_name), file_name=tr_log_file_name
    )

    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))

    flag_models_dir, _ = check_if_dir_or_file_exists(
        os.path.join(modelfile_path, main_exp_name), file_name=None
    )

    print("Is model-directory present:? - {}".format(flag_models_dir))
    # print("Is file present:? - {}".format(flag_file))

    tr_logfile_name_with_path = os.path.join(
        os.path.join(logfile_path, main_exp_name), tr_log_file_name
    )
    
    #te_logfile_name_with_path = os.path.join(
    #    os.path.join(logfile_path, main_exp_name), te_log_file_name
    #)

    if not flag_log_dir:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)

    if not flag_models_dir:
        print("Creating {}".format(os.path.join(modelfile_path, main_exp_name)))
        os.makedirs(os.path.join(modelfile_path, main_exp_name), exist_ok=True)

    modelfile_path = os.path.join(
        modelfile_path, main_exp_name
    )  # Modify the modelfile path to add full model file

    if mode.lower() == "train":
        model_danse = DANSE(**estimator_options)
        tr_verbose = True

        # Starting model training
        tr_losses, val_losses, _, _, _ = train_danse(
            model=model_danse,
            train_loader=train_loader,
            val_loader=val_loader,
            options=estimator_options,
            nepochs=model_danse.rnn.num_epochs,
            logfile_path=tr_logfile_name_with_path,
            modelfile_path=modelfile_path,
            save_chkpoints="some",
            device=device,
            tr_verbose=tr_verbose,
        )
        # if tr_verbose == True:
        #    plot_losses(tr_losses=tr_losses, val_losses=val_losses, logscale=False)

        losses_model = {}
        losses_model["tr_losses"] = tr_losses
        losses_model["val_losses"] = val_losses

        with open(
            os.path.join(
                os.path.join(logfile_path, main_exp_name),
                "danse_{}_losses_eps{}.json".format(
                    estimator_options["rnn_type"],
                    estimator_options["rnn_params_dict"][model_type]["num_epochs"],
                ),
            ),
            "w",
        ) as f:
            f.write(json.dumps(losses_model, cls=NDArrayEncoder, indent=2))

    elif mode.lower() == "test":

        (
            _,
            _,
            X_estimated_filtered,
            Pk_estimated_filtered,
            time_elapsed_danse,
        ) = test_danse_ssm(
            Y_test=test_input.permute(0,2,1),
            n_states=n_states,
            n_obs=n_obs,
            Cw=R.numpy(),
            H=estimator_options["H"],
            model_file_saved_danse=model_file_saved,
            Cw_test=test_Cw,
            rnn_type=model_type,
            device=device,
        )
        mse_loss_fn = nn.MSELoss(reduction='mean')
        MSE_danse_linear_arr = torch.empty(test_input.shape[0])# MSE [Linear]
        
        for j in range(0, test_input.shape[0]):   
            MSE_danse_linear_arr[j] = mse_loss_fn(X_estimated_filtered.permute(0,2,1)[j], test_target[j]).item()
            print("MSE for trajectory: {} is: {} dB".format(j+1, MSE_danse_linear_arr[j]))     
            
        MSE_danse_linear_avg = torch.mean(MSE_danse_linear_arr)
        MSE_danse_dB_avg = 10 * torch.log10(MSE_danse_linear_avg)
        # Standard deviation
        MSE_danse_linear_std = torch.std(MSE_danse_linear_arr, unbiased=True)

        # Confidence interval
        MSE_danse_dB_std = 10 * torch.log10(MSE_danse_linear_std + MSE_danse_linear_avg) - MSE_danse_dB_avg

        print("Test MSE {:.4f} ± {:.4f} dB on trajectories: {} ".format(
            MSE_danse_dB_avg, MSE_danse_dB_std, test_input.shape[0]
        ))

        test_results_file = os.path.join(os.path.join(logfile_path, main_exp_name),
                                        "test_results.pt")
        test_collection = [
            test_input, # Test set measurement trajectories,
            test_target, # Test set measurement targets / states,
            X_estimated_filtered.permute(0,2,1), # Test set filtered estimates using DANSE,
            Pk_estimated_filtered.permute(0, 2, 3, 1), # Test set (diagonal) covariance matrices for filtered estimates using DANSE,
            time_elapsed_danse, # CPU time for inference
            MSE_danse_linear_arr, # MSE linear computed between test set targets and estimates,
            MSE_danse_dB_avg, # Result metric in dB,
            MSE_danse_dB_std # Result metric in dB
        ]

        torch.save(obj=test_collection, f=test_results_file)

    return None


if __name__ == "__main__":
    main()
