import torch

window_size = 6
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask_value = 0
input_folder = 'data/ohio-data/processed'
output_folder = 'data/ohio-data/processed/cleaned'  # Create a subfolder for processed files
# # output_folder_test = 'data/ohio-data/processed/cleaned_test'
output_folder_train = 'data/ohio-data/processed/cleaned'  # Create a subfolder for processed files
