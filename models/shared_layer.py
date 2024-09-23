import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging

# Initialize logger
logging.basicConfig(
    filename='training.log',  # File where logs will be saved
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    # filemode='w'  # Overwrite log file on each run
)
logger = logging.getLogger()

# class PersonalizedLSTMModel(nn.Module):
#     def __init__(self, input_shape, output_shape, hidden_units=128, dropout_rate=0.2, l2_reg=0.001, mask_value=-1):
#         """
#         Initialize the LSTM model.

#         Parameters:
#         - input_shape: Shape of the input (sequence length, features).
#         - output_shape: Shape of the output.
#         - hidden_units: Number of units in the LSTM layer.
#         - dropout_rate: Dropout rate for regularization.
#         - l2_reg: L2 regularization factor.
#         - mask_value: The value to be masked in the input sequences (default: -1).
#         """
#         super(PersonalizedLSTMModel, self).__init__()
        
#         self.mask_value = mask_value  # Value to be masked
        
#         # LSTM layer shared among all branches
#         self.shared_lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=hidden_units, batch_first=True)
        
#         # Dense layer after LSTM
#         self.dense = nn.Linear(hidden_units, 64)
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(dropout_rate)
        
#         # Output layer
#         self.output_layer = nn.Linear(output_shape[0] * 64, output_shape[0])
        
#         # L2 regularization factor
#         self.l2_reg = l2_reg

#     def apply_mask(self, x):
#         """
#         Apply masking to the input tensor by replacing masked values with zeros.
#         Parameters:
#         - x: Input tensor of shape (batch_size, sequence_length, features)
#         Returns:
#         - masked_x: The input tensor where masked values are set to zero.
#         """
#         mask = (x != self.mask_value).float()  # Create a mask where 1 indicates valid data
#         masked_x = x * mask  # Apply the mask by zeroing out the masked values
#         return masked_x, mask

#     def forward(self, x):
#         """
#         Forward pass through the model.

#         Parameters:
#         - x: A list of input tensors, one for each output dimension (12 branches in your case).

#         Returns:
#         - A tensor of shape (batch_size, output_shape[0]).
#         """
#         shared_outputs = []
        
#         # Process each input with the shared LSTM
#         for branch_input in x:
#             # Apply masking
#             masked_input, mask = self.apply_mask(branch_input)
            
#             # Pass through shared LSTM
#             lstm_out, _ = self.shared_lstm(masked_input)
            
#             # Take the last time step output, considering only valid timesteps (masked with 1)
#             lstm_out = lstm_out[:, -1, :]
            
#             # Pass through Dense and Dropout
#             output = F.relu(self.dense(lstm_out))
#             output = self.dropout(output)
            
#             shared_outputs.append(output)
        
#         # Concatenate all branch outputs
#         merged_output = torch.cat(shared_outputs, dim=1)
        
#         # Pass through final output layer
#         output = self.output_layer(merged_output)
#         # output = torch.sigmoid(output)
#         return output

class PersonalizedLSTMModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PersonalizedLSTMModel, self).__init__()

                # CNN Layer before LSTM
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_shape[-1], out_channels=64, kernel_size=4, stride=2, padding=1 , dilation=2) 
            for _ in range(output_shape[0])
        ])
        

        # Shared LSTM layer
        self.shared_lstm = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        # self.shared_lstm = nn.GRU(input_size=input_shape[-1], hidden_size=128, batch_first=True)
        # self.batch_norm = nn.BatchNorm1d(128)
        # Personalized layers
        self.personalized_fc = nn.ModuleList([nn.Linear(128, 64) for _ in range(output_shape[0])])
        self.dropout = nn.Dropout(0.2)

        # Output layer
        self.output_layer = nn.Linear(64 * output_shape[0], output_shape[0])

    def apply_mask(self, x):
        """
        Apply masking to the input tensor by replacing masked values with zeros.
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, features)
        Returns:
        - masked_x: The input tensor where masked values are set to zero.
        """
        mask = (x != -2.5).float()  # Create a mask where 1 indicates valid data
        masked_x = x * mask  # Apply the mask by zeroing out the masked values
        return masked_x, mask
    
    def forward(self, x):
        # Assuming x is a list of input tensors, one for each output
        shared_outputs = []
        for i in range(len(x)):
            masked_input, mask = self.apply_mask(x[i])
            # print(masked_input)
            
            conv_out = self.conv_layers[i](masked_input.transpose(1, 2))
            
            # # LSTM expects input in the shape [batch_size, sequence_length, features], so we transpose back
            lstm_input = conv_out.transpose(1, 2)

            lstm_out, _ = self.shared_lstm(lstm_input)
            # normalized_out = self.batch_norm(lstm_out[:, -1, :]) 
            # Take the last output of the LSTM (corresponding to the last timestep)
            shared_outputs.append(lstm_out[:, -1, :])  

        # Personalized dense and dropout layers
        personalized_outputs = []
        for i, shared_out in enumerate(shared_outputs):
            personal_out = F.relu(self.personalized_fc[i](shared_out))
            personal_out = self.dropout(personal_out)
            personalized_outputs.append(personal_out)

        # Concatenate personalized outputs
        merged_personalized_output = torch.cat(personalized_outputs, dim=1)

        # Output layer
        output = self.output_layer(merged_personalized_output)
        # output = torch.sigmoid(output)
        return output

# class PersonalizedLSTMModel(nn.Module):
#     def __init__(self, input_shape, output_shape, hidden_units=128, dropout_rate=0.2, l2_reg=0.001, mask_value=-1):
#         """
#         Initialize the LSTM model.

#         Parameters:
#         - input_shape: Shape of the input (sequence length, features).
#         - output_shape: Shape of the output.
#         - hidden_units: Number of units in the LSTM layer.
#         - dropout_rate: Dropout rate for regularization.
#         - l2_reg: L2 regularization factor.
#         - mask_value: The value to be masked in the input sequences (default: -1).
#         """
#         super(PersonalizedLSTMModel, self).__init__()
        
#         self.mask_value = mask_value  # Value to be masked
        
#         # LSTM layer shared among all branches
#         self.shared_lstm = nn.LSTM(input_size=input_shape[-1], hidden_size=hidden_units, batch_first=True)
        
#         # Dense layer after LSTM
#         self.dense = nn.Linear(hidden_units, 64)
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(dropout_rate)
        
#         # Output layer
#         self.output_layer = nn.Linear(output_shape[0] * 64, output_shape[0])
        
#         # L2 regularization factor
#         self.l2_reg = l2_reg

#     def apply_mask(self, x):
#         """
#         Apply masking to the input tensor by creating a binary mask for valid values.

#         Parameters:
#         - x: Input tensor of shape (batch_size, sequence_length, features)
#         Returns:
#         - masked_x: The input tensor where masked values are set to zero.
#         - mask: A binary mask where valid values are 1 and masked values are 0.
#         """
#         # Create a mask: valid values (not equal to mask_value) are set to 1, masked values are set to 0
#         mask = (x != self.mask_value).float()  # Shape: (batch_size, sequence_length, features)
        
#         # Replace masked values (-1) with zeros in the input tensor
#         masked_x = x.masked_fill(x == self.mask_value, 0)
        
#         return masked_x, mask

#     def forward(self, x):
#         """
#         Forward pass through the model.

#         Parameters:
#         - x: A list of input tensors, one for each output dimension (12 branches in your case).

#         Returns:
#         - A tensor of shape (batch_size, output_shape[0]).
#         """
#         shared_outputs = []
        
#         # Process each input with the shared LSTM
#         for branch_input in x:
#             # Apply masking
#             masked_input, mask = self.apply_mask(branch_input)  # mask has shape (batch_size, seq_len, features)
            
#             # Initialize hidden states (h_0, c_0) manually
#             batch_size, seq_len, _ = masked_input.size()
#             h_0 = torch.zeros(1, batch_size, self.shared_lstm.hidden_size, device=masked_input.device)
#             c_0 = torch.zeros(1, batch_size, self.shared_lstm.hidden_size, device=masked_input.device)
            
#             # Pass through shared LSTM step by step with masking
#             lstm_hidden_out = []
#             for t in range(seq_len):
#                 input_step = masked_input[:, t, :].unsqueeze(1)  # Get input at time step t
#                 mask_step = mask[:, t, :]  # Mask at time step t
                
#                 lstm_out, (h_0, c_0) = self.shared_lstm(input_step, (h_0, c_0))
                
#                 # Apply the mask to the hidden states, ensuring that masked timesteps don't update the hidden state
#                 h_0 = h_0 * mask_step.unsqueeze(0)
#                 c_0 = c_0 * mask_step.unsqueeze(0)
                
#                 lstm_hidden_out.append(lstm_out)
            
#             lstm_hidden_out = torch.cat(lstm_hidden_out, dim=1)
            
#             # Take the last valid time step output
#             lstm_out = lstm_hidden_out[:, -1, :]
            
#             # Pass through Dense and Dropout
#             output = F.relu(self.dense(lstm_out))
#             output = self.dropout(output)
            
#             shared_outputs.append(output)
        
#         # Concatenate all branch outputs
#         merged_output = torch.cat(shared_outputs, dim=1)
        
#         # Pass through final output layer
#         output = self.output_layer(merged_output)
        
#         return output


def validate_model(model, val_loader, criterion, second_criterion, device):
    """
    Validate the LSTM model on the validation set.

    Parameters:
    - model: The model to validate.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function to evaluate performance.
    - device: Device to perform computations on (CPU or GPU).

    Returns:
    - avg_val_loss: The average validation loss over the validation dataset.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    running_val_loss = 0.0

    running_val_loss2 = 0.0
    
    with torch.no_grad():  # Disable gradient calculations during validation
        for inputs, targets in val_loader:
            # Move data to the appropriate device (GPU or CPU)
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()

            loss2 = second_criterion(outputs, targets)
            running_val_loss2 += loss2.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_loss_2 = running_val_loss2 / len(val_loader)

    return avg_val_loss, avg_val_loss_2


def train_model(model, train_loader, val_loader, epochs, learning_rate):
    """
    Train the LSTM model using a custom training loop with tqdm progress bars.
    Also evaluates the model on validation data after each epoch.
    
    Logs key training events and results to both the console and a log file.
    
    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - epochs: Number of epochs to train for.
    - learning_rate: Learning rate for optimizer.
    
    Returns:
    - model: The trained model.
    """
    # Define the optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.HuberLoss()  # huber Error
    criterion = nn.L1Loss()  # Mean Absolute Error
    criterion2 = nn.MSELoss()  # Mean Absolute Error

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Starting training for {epochs} epochs with learning rate {learning_rate}.")

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Add tqdm progress bar for each epoch
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device (GPU or CPU)
                inputs = [inp.to(device) for inp in inputs]  # List of tensors, one for each input branch
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # Update progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}")

        # Validate the model after each epoch
        avg_val_loss, avg_val_loss2 = validate_model(model, val_loader, criterion, criterion2, device)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Validation MAE: {avg_val_loss:.4f}, Validation RMSE: {np.sqrt(avg_val_loss2):.4f}")
        print(f"Epoch [{epoch + 1}/{epochs}], Validation MAE: {avg_val_loss:.4f}, Validation RMSE: {np.sqrt(avg_val_loss2):.4f}")
    
    # return model  # Return the trained model

def evaluate_test_data(model, test_loader):
    """
    Evaluate the model on the test data and calculate RMSE.
    
    Parameters:
    - model: The trained model to evaluate.
    - test_loader: DataLoader for the test data.
    
    Returns:
    - rmse: The calculated Root Mean Squared Error (RMSE) on the test data.
    """
    model.eval()  # Set the model to evaluation mode
    criterion = nn.MSELoss()  # We use MSELoss and take the square root for RMSE
    running_test_loss = 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, targets in test_loader:
            # Move data to device (GPU or CPU)
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute MSE loss
            loss = criterion(outputs, targets)
            running_test_loss += loss.item()
    
    # Calculate the average MSE over all batches and take the square root for RMSE
    avg_mse = running_test_loss / len(test_loader)
    rmse = np.sqrt(avg_mse)
    
    print(f"Test RMSE: {rmse:.4f}")
    return rmse

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        """
        A Dataset class to load time series data for LSTM training.

        Parameters:
        - inputs: A NumPy array of shape (12, num_samples, sequence_length), where:
          - 12 is the number of input branches.
          - num_samples is the number of samples (e.g., 13618).
          - sequence_length is the number of time steps per sample (e.g., 6).
        - targets: A NumPy array of shape (12, num_samples), where:
          - 12 is the number of predicted values.
          - num_samples is the number of samples (e.g., 13618).
        """
        self.inputs = inputs  # Inputs shape: (12, num_samples, sequence_length)
        self.targets = targets  # Targets shape: (12, num_samples)

    def __len__(self):
        return self.inputs.shape[1]  # num_samples (e.g., 13618)

    def __getitem__(self, idx):
        # For inputs, return a list of tensors, one for each branch (12 branches)
        input_batch = [torch.tensor(self.inputs[i][idx], dtype=torch.float32).unsqueeze(1) for i in range(self.inputs.shape[0])]
        # Targets
        target_batch = torch.tensor(self.targets[:, idx], dtype=torch.float32)
        # input_batch = [self.inputs[i][idx] for i in range(self.inputs.shape[0])]
        # # Targets
        # target_batch = self.targets[:, idx]
        return input_batch, target_batch
