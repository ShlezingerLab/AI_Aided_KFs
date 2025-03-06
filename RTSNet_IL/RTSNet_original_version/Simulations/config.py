"""This file contains the settings for the simulation"""
import torch
import argparse

def general_settings():
    ### Dataset settings
    parser = argparse.ArgumentParser(prog = 'RTSNet',description = 'Training parameters')
    parser.add_argument('--N_E', type=int, default=1000, metavar='trainset-size',
                        help='input training dataset size (# of sequences)')
    parser.add_argument('--N_CV', type=int, default=100, metavar='cvset-size',
                        help='input cross validation dataset size (# of sequences)')
    parser.add_argument('--N_T', type=int, default=200, metavar='testset-size',
                        help='input test dataset size (# of sequences)')
    parser.add_argument('--T', type=int, default=100, metavar='length',
                        help='input sequence length')
    parser.add_argument('--T_test', type=int, default=100, metavar='test-length',
                        help='input test sequence length')

    ### Training settings
    parser.add_argument('--n_steps', type=int, default=1000, metavar='N_steps',
                        help='number of training steps (default: 1000)')
    parser.add_argument('--n_batch', type=int, default=20, metavar='N_B',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    ### BRNN Network settings
    parser.add_argument('--in_mult_RNN', type=int, default=3, metavar='in_mult_RNN',
                        help='input dimension multiplier for the GRU of forward RNN (default: 3)')
    parser.add_argument('--out_mult_RNN', type=int, default=2, metavar='out_mult_RNN',
                        help='output dimension multiplier for the GRU of forward RNN (default: 2)')
    parser.add_argument('--nGRU_FW_RNN', type=int, default=3, metavar='nGRU_FW_RNN',
                        help='number hidden layers for GRU in the forward RNN (default: 3)')
    
    parser.add_argument('--in_mult_RNN_BW', type=int, default=3, metavar='in_mult_RNN_BW',
                        help='input dimension multiplier for the GRU of backward RNN (default: 3)')
    parser.add_argument('--out_mult_RNN_BW', type=int, default=2, metavar='out_mult_RNN',
                        help='output dimension multiplier for the GRU of backward RNN (default: 2)')
    parser.add_argument('--nGRU_BW_RNN', type=int, default=2, metavar='nGRU_FW_RNN',
                        help='number hidden layers for GRU in the backward RNN (default: 2)')

    ### RTSNet settings
    parser.add_argument('--in_mult_KNet', type=int, default=5, metavar='in_mult_KNet',
                        help='input dimension multiplier for KNet (FW pass of RTSNet)')
    parser.add_argument('--out_mult_KNet', type=int, default=40, metavar='out_mult_KNet',
                        help='output dimension multiplier for KNet (FW pass of RTSNet)')
    
    parser.add_argument('--in_mult_RTSNet', type=int, default=5, metavar='in_mult_RTSNet',
                        help='input dimension multiplier for RTSNet BW pass')
    parser.add_argument('--out_mult_RTSNet', type=int, default=40, metavar='out_mult_RTSNet',
                        help='output dimension multiplier for RTSNet BW pass')

    args = parser.parse_args()
    return args
