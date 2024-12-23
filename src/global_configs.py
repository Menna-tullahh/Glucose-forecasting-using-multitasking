import torch

window_size = 12
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask_value = 0
input_folder = 'data/ohio-data/processed'
output_folder = 'data/ohio-data/processed/cleaned'  # Create a subfolder for processed files
# # output_folder_test = 'data/ohio-data/processed/cleaned_test'
output_folder_train = 'data/ohio-data/processed/cleaned'  # Create a subfolder for processed files
patients_list = ['559', '563', '570', '575', '588', '591', '540', '544', '552', '567', '584', '596']
# prediction_horizons = [3, 6, 9, 12, 15, 18]
prediction_horizons = [3, 6, 9, 12,15, 18, 24]
# 15, 30, 45, 60, 90, 120
