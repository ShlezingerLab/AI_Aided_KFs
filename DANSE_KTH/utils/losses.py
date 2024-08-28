#########################################################################
# This script creates functions for the losses for DMMs. 

# [1]. Krishnan, Rahul, Uri Shalit, and David Sontag. 
# "Structured inference networks for nonlinear state space models." 
# Proceedings of the AAAI Conference on Artificial Intelligence. 
# Vol. 31. No. 1. 2017. 

# Parts of this implementation has been adapted from
# https://github.com/yjlolo/pytorch-deep-markov-model/blob/master/


# Creator: Anubhab Ghosh, Jan 2024
#########################################################################

# Import the necessary libraries
import sys
import math
from os import path
# __file__ should be defined in this case
PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
import torch.nn as nn
from utils.utils import create_diag
from torch.distributions import kl_divergence, MultivariateNormal

def kl_div_loss(mu_1, var_1, mu_2=None, var_2=None):
    """ This function computes the KL-divergence between 
    two Gaussian distributions. We assume the function is computing
    the following KL divergence = KL(N(mu_1, var_1) || N(mu_2, var_2))

    Args:
        mu_1 (torch.Tensor): Mean vector of the first Gaussian distribution (batch_size x latent_dim)
        var_1 (torch.Tensor): Variance vector of the first Gaussian distribution (diagonal covariance) (batch_size x latent_dim)
        mu_2 (torch.Tensor, optional): Mean of the second Gaussian distribution (batch_size x latent_dim). Defaults to None.
        var_2 (torch.Tensor, optional): Variance vector of the second Gaussian distribution (diagonal covariance) (batch_size x latent_dim). Defaults to None.

    Returns:
        kl_div_loss (torch.Tensor): The KL-divergence loss (for every sample in a batch)
    """
    if mu_2 is None:
        mu_2 = torch.zeros_like(mu_1)
    if var_2 is None:
        var_2 = torch.zeros_like(mu_1)

    cov_1 = create_diag(var_1)
    cov_2 = create_diag(var_2)

    mvn_dist1 = MultivariateNormal(loc=mu_1, covariance_matrix=cov_1)
    mvn_dist2 = MultivariateNormal(loc=mu_2, covariance_matrix=cov_2)

    kl_div_loss = kl_divergence(p=mvn_dist1, q=mvn_dist2)

    #std_mat_1 = torch.cholesky(cov_1)
    #std_mat_2 = torch.cholesky(cov_2)
    #kl_div_loss = torch.logdet(std_mat_2) - torch.logdet(std_mat_1) - 0.5 * mu_1.shape[-1] \
    #    - 0.5 * torch.norm(torch.bmm(torch.inverse(std_mat_2),(mu_1 - mu_2).unsqueeze(2)), p=2, dim=(1,2))**2 \
    #    + 0.5 * torch.sum(torch.bmm(torch.inverse(std_mat_2), std_mat_1)**2, dim=(1,2))

    return kl_div_loss

def ll_Gaussian(obs_curr_instant, mu_obs_curr_instant, var_obs_curr_instant):
    """ This function computes the log-likelihood loss as the 
    log-likelihood of a Gaussian distribution, assuming continuous random variables in the emission distribution. 
    In case of DMM, it is generally assumed that the emission distribution is a Gaussian distribution. 

    Args:
        obs_curr_instant (torch.Tensor): Observation sequence at the current time instant (batch_size, obs_dim)
        obs_mean_curr_instant (torch.Tensor): Mean of the observation sequence at the current time instant (batch_size, obs_dim)
        obs_var_curr_instant (torch.Tensor): Variance of the observation sequence at the current time instant (batch_size, obs_dim)
    
    Returns:
        ll_loss (torch.Tensor): Log-likelihood loss at the current instant (batch_size,)
    """
    batch_size, input_dim = obs_curr_instant.shape 
    assert (var_obs_curr_instant >= 0).all()== True, "Variances are not all non-negative!!"
    obs_cov_curr_instant = create_diag(x=var_obs_curr_instant)
    emission_dist = MultivariateNormal(loc=mu_obs_curr_instant, covariance_matrix=obs_cov_curr_instant)
    ll_loss = emission_dist.log_prob(obs_curr_instant)

    #ll_loss = - 0.5 * input_dim * math.log(math.pi*2) - 0.5 * torch.logdet(obs_cov_curr_instant) \
    #    - 0.5 * torch.einsum('ni,ni->n',
    #    (obs_curr_instant - mu_obs_curr_instant), 
    #    torch.einsum('nij,nj->ni',torch.inverse(obs_cov_curr_instant), (obs_curr_instant - mu_obs_curr_instant)))
    
    assert ll_loss.shape[0] == batch_size, "Error in nll calculation, batch dimensions do not match!!"
    return ll_loss

def dmm_loss_curr_instant(obs_curr_instant, 
                    mu_obs_curr_instant, 
                    var_obs_curr_instant, 
                    mu_latent_q_curr_instant,
                    var_latent_q_curr_instant,
                    mu_latent_curr_instant,
                    var_latent_curr_instant,
                    kl_annealing_factor=1.0, mask=None):
    
    """ This function computes the loss term at the current time instant for the deep Markov model (DMM). The function 
    computes the negative variational lower bound at the current time instant as the sum of negative log-likelihood loss and the KL-divegence. 

    Args:
        obs_curr_instant (torch.Tensor): Observations at the current time instant (batch size x obs dim)
        mu_obs_curr_instant (torch.Tensor): Mean of the observation at the current time instant (batch size x seq length x obs dim)
        var_obs_curr_instant (torch.Tensor): Variance of the observation at the current time instant  (batch size x seq length x obs dim)
        mu_latent_q_curr_instant (torch.Tensor): Mean of the latent at the current time instant sampled from approx. posterior (batch size x seq length x latent dim)
        var_latent_q_curr_instant (torch.Tensor): Variance of the latent at the current time instant sampled from approx. posterior (batch size x seq length x latent dim)
        mu_latent_curr_instant (torch.Tensor): Mean of the sampled latent at the current time instant for the transition distibution (batch size x seq length x latent dim)
        var_latent_curr_instant (torch.Tensor): Variance of the sampled latent at the current time instant for the transition distibution (batch size x seq length x latent dim)
        kl_annealing_factor (float, optional): Multiplicative factor for the KL-divergence. Defaults to 1.0.
        mask (bool, optional): _description_. Defaults to None.

    Returns:
        kl_div_loss_curr_instant (torch.Tensor): KL-divergence computed over batch size and at the current time instant (batch size,)
        ll_loss_curr_instant (torch.Tensor): Log-likelihood computed over batch size and at the current time instant (batch size,)
        n_vlb_curr_instant (torch.Tensor): Variational lower bound computed over batch size and at the current time instant (batch size,)
    """
    kl_div_loss_curr_instant = kl_div_loss(
        mu_1=mu_latent_q_curr_instant,
        var_1=var_latent_q_curr_instant,
        mu_2=mu_latent_curr_instant,
        var_2=var_latent_curr_instant
    )

    ll_loss_curr_instant = ll_Gaussian(
        obs_curr_instant=obs_curr_instant,
        mu_obs_curr_instant=mu_obs_curr_instant,
        var_obs_curr_instant=var_obs_curr_instant
    )

    assert kl_div_loss_curr_instant.shape[0] == obs_curr_instant.shape[0]
    assert ll_loss_curr_instant.shape[0] == obs_curr_instant.shape[0]

    n_vlb_curr_instant = - (ll_loss_curr_instant - kl_div_loss_curr_instant * kl_annealing_factor)

    return kl_div_loss_curr_instant, ll_loss_curr_instant, n_vlb_curr_instant

def dmm_loss_seq(obs_seq,
                 mu_obs_seq, 
                 var_obs_seq,
                 mu_latent_q_seq, 
                 var_latent_q_seq, 
                 mu_latent_seq, 
                 var_latent_seq):
    """ This function computes the loss term for the deep Markov model (DMM). The function takes in the 
    observation sequence, its mean and variance sequences (assuming diagonal covariances), the mean and variance
    sequences (assuming diagonal covariances) for the states sampled from the approximate posterior distribution, 
    and the mean and variance sequences (assuming diagonal covariances) for the transition distribution, and
    computes the negative of variational lower bound as the sum of the negative of the log-likelihood loss and 
    the KL-divegence. 

    Args:
        obs_seq (torch.Tensor): Observation sequence (batch size x seq length x obs dim)
        mu_obs_seq (torch.Tensor): Mean of the observation sequence (batch size x seq length x obs dim)
        var_obs_seq (torch.Tensor): Variance of the observation sequence  (batch size x seq length x obs dim)
        mu_latent_q_seq (torch.Tensor): Mean of the latent sequence sampled from approx. posterior (batch size x seq length x latent dim)
        var_latent_q_seq (torch.Tensor): Variance of the latent sequence sampled from approx. posterior (batch size x seq length x latent dim)
        mu_latent_seq (torch.Tensor): Mean of the sampled latent sequence for the transition distibution (batch size x seq length x latent dim)
        var_latent_seq (torch.Tensor): Variance of the sampled latent sequence for the transition distibution (batch size x seq length x latent dim)

    Returns:
        kl_loss_seq_mean (torch.Tensor): KL-divergence computed as a mean over batch size and seq length (scalar)
        ll_loss_seq_mean (torch.Tensor): Log-likelihood computed as a mean over batch size and seq length (scalar)
        n_vlb_loss_seq_mean (torch.Tensor): Negative of the Variational lower bound computed as a mean over batch size and seq length (scalar)
    """
    assert mu_obs_seq.shape[0] == mu_latent_q_seq.shape[0], "Unequal batch sizes of obs. and states"
    assert mu_obs_seq.shape[1] == mu_latent_q_seq.shape[1], "Unequal sequence lengths of obs. and states"

    batch_size, seq_len, obs_dim = obs_seq.shape

    n_vlb_seq_mean= 0.0
    kl_loss_seq_mean = 0.0
    ll_loss_seq_mean = 0.0

    for t in range(seq_len):

        kl_div_loss_curr_instant, ll_loss_curr_instant, n_vlb_curr_instant = dmm_loss_curr_instant(
            obs_curr_instant=obs_seq[:, t, :],
            mu_obs_curr_instant=mu_obs_seq[:, t, :],
            var_obs_curr_instant=var_obs_seq[:, t, :],
            mu_latent_q_curr_instant=mu_latent_q_seq[:, t, :],
            var_latent_q_curr_instant=var_latent_q_seq[:, t, :],
            mu_latent_curr_instant=mu_latent_seq[:, t, :],
            var_latent_curr_instant=var_latent_seq[:, t, :]
        ) # These computed losses are of dimensions (batch size x _ dim)

        kl_loss_seq_mean += kl_div_loss_curr_instant.mean(0) / seq_len
        ll_loss_seq_mean += ll_loss_curr_instant.mean(0) / seq_len
        n_vlb_seq_mean += n_vlb_curr_instant.mean(0) / seq_len
    
    return kl_loss_seq_mean, ll_loss_seq_mean, n_vlb_seq_mean

