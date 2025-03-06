"""# **Class: Bi-directional RNN**
RNN with one forward filtering pass and one backward smoothing pass
"""

import torch
import torch.nn as nn

from RNN.RNN_forward import RNN_FW

class Vanilla_RNN(RNN_FW):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################

    def Build(self, args, SysModel, fully_agnostic = False):
        self.fully_agnostic = fully_agnostic

        # Set State Evolution Function
        self.f = SysModel.f
        self.m = SysModel.m

        # Set Observation Function
        self.h = SysModel.h
        self.n = SysModel.n

        self.InitSequence(SysModel.m1x_0, SysModel.T)

        # input dim for FW GRU
        input_dim_RNN = (self.m + self.n) * args.in_mult_RNN
        # Hidden Dimension for FW GRU
        self.hidden_dim = ((self.n * self.n) + (self.m * self.m)) * args.out_mult_RNN
        # FW GRU layers
        self.n_layers = args.nGRU_FW_RNN
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers
        
        self.InitRNN(input_dim_RNN)

        # input dim for BW GRU
        input_dim_RNN = (self.m + self.m) * args.in_mult_RNN_BW
        # Hidden Dimension for BW GRU
        self.hidden_dim_bw = 2 * self.m * self.m * args.out_mult_RNN_BW

        self.InitRNN_BW(args, input_dim_RNN)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitRNN_BW(self, args, input_dim_RNN):

        self.seq_len_input = 1
        self.batch_size = 1
        self.n_layers_bw = args.nGRU_BW_RNN
        # Hidden Sequence Length
        self.seq_len_hidden_bw = self.n_layers_bw

        # Initialize a Tensor for Hidden State
        self.hn_bw = torch.randn(self.seq_len_hidden_bw, self.batch_size, self.hidden_dim_bw)

        # GRUs
        self.rnn_GRU_bw = nn.GRU(input_dim_RNN, self.hidden_dim_bw, self.n_layers_bw)
        
        # Fully connected 3
        self.d_input_FC3 = self.m + self.m
        self.d_output_FC3 = input_dim_RNN
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.hidden_dim_bw
        self.d_output_FC4 = self.m 
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU())
        

    ####################################
    ### Initialize Backward Sequence ###
    ####################################
    def InitBackward(self, filter_x_T):
        self.s_m1x_nexttime = torch.squeeze(filter_x_T)


    def step_est_BW(self, filter_x):
        # If fully agnostic, xhat is the smoothed x_t. 
        # Else, xhat computes the fix to the prior of x_t.
        self.s_m1x_increment = self.BW_step(filter_x)

        # Reshape to a Matrix(linear case) or Squeeze (NL case)
        self.s_m1x_increment =  torch.squeeze(self.s_m1x_increment)

        # Add increment to the filtered x
        self.s_m1x = filter_x +  self.s_m1x_increment

    ########################
    ### Vanilla RNN Step ###
    ########################
    def BW_RNN_step(self, filter_x):
        
        self.step_est_BW(filter_x)

        self.s_m1x_nexttime = self.s_m1x

        return torch.squeeze(self.s_m1x)


    ########################
    ### Kalman Gain Step ###
    ########################
    def BW_step(self, filter_x):

        def expand_dim(x):
            x = torch.squeeze(x)
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded
        
        filter_x = expand_dim(filter_x)
        s_m1x_nexttime = expand_dim(self.s_m1x_nexttime)

        ####################
        ### Forward Flow ###
        ####################       
        # FC 3
        in_FC3 = torch.cat((filter_x, s_m1x_nexttime), 2)
        out_FC3 = self.FC3(in_FC3)
       
        # BW GRU        
        GRU_out, self.hn_bw = self.rnn_GRU_bw(expand_dim(out_FC3), self.hn_bw)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim_bw))

        # FC 4
        in_FC4 = GRU_out_reshape 
        out_FC4 = self.FC4(in_FC4)

        return out_FC4

    ###############
    ### Forward ###
    ###############
    def forward(self, y, filter_x, filter_x_nexttime, smoother_x_tplus2):
        if y is None:
            # BW pass
            return self.BW_RNN_step(filter_x)
        else:
            # FW pass
            y = torch.squeeze(y)

            return self.FW_RNN_step(y)

        

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        ### FW hn
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
        ### BW hn
        hidden = weight.new(self.n_layers_bw, self.batch_size, self.hidden_dim_bw).zero_()
        self.hn_bw = hidden.data



