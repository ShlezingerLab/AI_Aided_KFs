import pandas as pd
import torch
import matplotlib.pyplot as plt

#  Load the CSV file into a DataFrame
df = pd.read_csv('data/mse_results.csv')

# Convert each column back to a PyTorch tensor
mse_CKF_true_datasets = torch.tensor(df['mse_CKF_true'].values)
mse_CKF_APBM_true_datasets = torch.tensor(df['mse_CKF_APBM_true'].values)
mse_CKF_PBM_datasets = torch.tensor(df['mse_CKF_PBM'].values)
mse_CKF_APBM_datasets = torch.tensor(df['mse_CKF_APBM'].values)

# Analyze data
# ===================================== #
# ============ CKF Ture =============== #
# ===================================== #
mse_CKF_true = torch.mean(mse_CKF_true_datasets)
mse_CKF_true_datasets_dB = 10 * torch.log10(mse_CKF_true_datasets)
mse_CKF_true_dB = 10 * torch.log10(mse_CKF_true)
mse_CKF_true_std = torch.std(mse_CKF_true_datasets, unbiased=True)
mse_CKF_true_std_dB = 10 * torch.log10(mse_CKF_true_std + mse_CKF_true) - mse_CKF_true_dB
print("CKF True - MSE:", mse_CKF_true_dB.item(), "[dB]")
print("CKF True - SE STD:", mse_CKF_true_std_dB.item(), "[dB]")
print(" ")

# ========================================== #
# ============ CKF APBM True =============== #
# ========================================== #
mse_CKF_APBM_true = torch.mean(mse_CKF_APBM_true_datasets)
mse_CKF_APBM_true_datasets_dB = 10 * torch.log10(mse_CKF_APBM_true_datasets)
mse_CKF_APBM_true_dB = 10 * torch.log10(mse_CKF_APBM_true)
mse_CKF_APBM_true_std = torch.std(mse_CKF_APBM_true_datasets, unbiased=True)
mse_CKF_APBM_true_std_dB = 10 * torch.log10(mse_CKF_APBM_true_std + mse_CKF_APBM_true) - mse_CKF_APBM_true_dB
print("CKF APBM_true - MSE:", mse_CKF_APBM_true_dB.item(), "[dB]")
print("CKF APBM_true - SE STD:", mse_CKF_APBM_true_std_dB.item(), "[dB]")
print(" ")

# ===================================== #
# ============ CKF PBM =============== #
# ===================================== #
mse_CKF_PBM = torch.mean(mse_CKF_PBM_datasets)
mse_CKF_PBM_datasets_dB = 10 * torch.log10(mse_CKF_PBM_datasets)
mse_CKF_PBM_dB = 10 * torch.log10(mse_CKF_PBM)
mse_CKF_PBM_std = torch.std(mse_CKF_PBM_datasets, unbiased=True)
mse_CKF_PBM_std_dB = 10 * torch.log10(mse_CKF_PBM_std + mse_CKF_PBM) - mse_CKF_PBM_dB
print("CKF PBM - MSE:", mse_CKF_PBM_dB.item(), "[dB]")
print("CKF PBM - SE STD:", mse_CKF_PBM_std_dB.item(), "[dB]")
print(" ")

# ===================================== #
# ============ CKF APBM =============== #
# ===================================== #
mse_CKF_APBM = torch.mean(mse_CKF_APBM_datasets)
mse_CKF_APBM_datasets_dB = 10 * torch.log10(mse_CKF_APBM_datasets)
mse_CKF_APBM_dB = 10 * torch.log10(mse_CKF_APBM)
mse_CKF_APBM_std = torch.std(mse_CKF_APBM_datasets, unbiased=True)
mse_CKF_APBM_std_dB = 10 * torch.log10(mse_CKF_APBM_std + mse_CKF_APBM) - mse_CKF_APBM_dB
print("CKF APBM - MSE:", mse_CKF_APBM_dB.item(), "[dB]")
print("CKF APBM - SE STD:", mse_CKF_APBM_std_dB.item(), "[dB]")
print(" ")

# Plot result
data_plot = {
    'Ture': mse_CKF_true_datasets_dB.numpy(),
    # 'APBM_True': mse_CKF_APBM_true_datasets_dB.numpy(),
    'PBM': mse_CKF_PBM_datasets_dB.numpy(),
    'APBM': mse_CKF_APBM_datasets_dB.numpy()
}
df_plot = pd.DataFrame(data_plot)
plt.figure(figsize=(8, 6))
df_plot.boxplot()

plt.title('Boxplot of MSE Results')
plt.ylabel('MSE [dB]')
plt.grid(True)
plt.show()
