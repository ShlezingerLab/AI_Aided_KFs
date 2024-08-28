#!/bin/bash
PYTHON="python3.8"
script_name="main_danse_opt.py"
mode="test"
model_type="gru"
datafile="../dataset/decimated_r0_Ttest3000.pt"
model_file_saved="./models/LorenzSSM_danse_opt_gru_m_3_n_3_T_3000_N_115_q2_-8.2dB_r2_0.0dB_run2/danse_gru_ckpt_epoch_671_best.pt"
dataset_type="LorenzSSM"

${PYTHON} ${script_name} \
	--mode ${mode} \
	--rnn_model_type ${model_type} \
	--dataset_type ${dataset_type} \
	--model_file_saved ${model_file_saved} \
	--datafile ${datafile}
