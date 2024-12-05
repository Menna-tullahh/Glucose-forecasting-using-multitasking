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

class SharedLayerModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SharedLayerModel, self).__init__()

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
        mask = (x != 0.0).float()  # Create a mask where 1 indicates valid data
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    """
    A single temporal block for the TCN.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return self.relu(out + x)  # Residual connection

class SharedLayerModelWithAttention(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=3, num_tcn_blocks=2, dropout=0.2):
        super(SharedLayerModelWithAttention, self).__init__()

        # Personalized TCN layers
        self.pre_shared_personalized_tcn = nn.ModuleList([
            nn.Sequential(
                *[TemporalBlock(input_shape[-1] if i == 0 else 64, 64, kernel_size, stride=1, dilation=2 ** i, padding=(kernel_size - 1) * (2 ** i) // 2, dropout=dropout)
                  for i in range(num_tcn_blocks)]
            )
            for _ in range(output_shape[0])
        ])

        # Personalized GRU layers
        self.pre_shared_personalized_gru = nn.ModuleList([
            nn.GRU(input_size=64, hidden_size=64, batch_first=True)
            for _ in range(output_shape[0])
        ])

        # Task-specific multi-head attention layers
        self.task_specific_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
            for _ in range(output_shape[0])
        ])

        # Shared LSTM layer
        self.shared_lstm = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

        # Residual transformation for the shared LSTM input
        self.shared_residual_transform = nn.Linear(64, 128)

        # Personalized fully connected layers after shared layer
        self.personalized_fc = nn.ModuleList([
            nn.Linear(128, 64) for _ in range(output_shape[0])
        ])
        self.residual_transform = nn.ModuleList([
            nn.Linear(128, 64) for _ in range(output_shape[0])
        ])
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
        mask = (x != 0.0).float()  # Create a mask where 1 indicates valid data
        masked_x = x * mask  # Apply the mask by zeroing out the masked values
        return masked_x, mask

    def forward(self, x):
        shared_inputs = []

        # Process each task's input through the personalized TCN+GRU with multi-head attention
        for i in range(len(x)):
            masked_input, mask = self.apply_mask(x[i])

            # Personalized TCN layer with residual connection
            personal_tcn_out = self.pre_shared_personalized_tcn[i](masked_input.transpose(1, 2))  # (batch, channels, seq)
            personal_tcn_out = personal_tcn_out.transpose(1, 2)  # Convert back to (batch, seq, features)

            # Personalized GRU layer with residual connection
            personal_gru_out, _ = self.pre_shared_personalized_gru[i](personal_tcn_out)  # (batch, seq, features)

            # Residual connection for GRU
            if personal_tcn_out.size(-1) == personal_gru_out.size(-1):
                personal_gru_out = personal_gru_out + personal_tcn_out

            # Task-specific multi-head attention
            attention_out, _ = self.task_specific_attention[i](personal_gru_out, personal_gru_out, personal_gru_out)  # (batch, seq, features)

            # Take the mean over the sequence as the context vector
            context_vector = attention_out.mean(dim=1)  # (batch, features)

            # Add context vector as input to shared layers
            shared_inputs.append(context_vector)

        # Stack the context vectors for shared processing
        shared_inputs = torch.stack(shared_inputs, dim=1)  # (batch, num_tasks, features)

        # Shared LSTM layer with residual connection
        shared_lstm_out, _ = self.shared_lstm(shared_inputs)

        # Transform the input for residual addition
        shared_residual = self.shared_residual_transform(shared_inputs)

        # Add residual connection for the shared LSTM
        shared_lstm_out = shared_lstm_out + shared_residual

        # Personalized dense layers after shared layer
        personalized_outputs = []
        for i in range(shared_lstm_out.size(1)):  # Iterate over task outputs
            shared_out = shared_lstm_out[:, i, :]

            # Transform the shared LSTM output to match the dense layer output
            residual_input = self.residual_transform[i](shared_out)

            # Pass through the dense layer
            personal_out = F.relu(self.personalized_fc[i](shared_out))
            personal_out = self.dropout(personal_out)

            # Add residual connection
            personal_out = personal_out + residual_input
            personalized_outputs.append(personal_out)

        # Concatenate task-specific outputs
        merged_personalized_output = torch.cat(personalized_outputs, dim=1)

        # Final output layer
        output = self.output_layer(merged_personalized_output)
        return output


 #BEST VERSION   
# class SharedLayerModelWithAttention(nn.Module):
#     def __init__(self, input_shape, output_shape):
#         super(SharedLayerModelWithAttention, self).__init__()

#         # Personalized CNN and GRU layers before the shared layers
#         self.pre_shared_personalized_cnn = nn.ModuleList([
#             nn.Conv1d(in_channels=input_shape[-1], out_channels=64, kernel_size=4, stride=2, padding=1, dilation=2)
#             for _ in range(output_shape[0])
#         ])
#         self.pre_shared_personalized_gru = nn.ModuleList([
#             nn.GRU(input_size=64, hidden_size=64, batch_first=True)
#             for _ in range(output_shape[0])
#         ])

#         # Task-specific multi-head attention layers
#         self.task_specific_attention = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
#             for _ in range(output_shape[0])
#         ])

#         # Shared LSTM layer
#         self.shared_lstm = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

#         # Residual transformation for the shared LSTM input
#         self.shared_residual_transform = nn.Linear(64, 128)

#         # Personalized fully connected layers after the shared layer
#         self.personalized_fc = nn.ModuleList([
#             nn.Linear(128, 64) for _ in range(output_shape[0])
#         ])
#         self.residual_transform = nn.ModuleList([
#             nn.Linear(128, 64) for _ in range(output_shape[0])
#         ])
#         self.dropout = nn.Dropout(0.2)

#         # Output layer
#         self.output_layer = nn.Linear(64 * output_shape[0], output_shape[0])

#     def apply_mask(self, x):
#         """
#         Apply masking to the input tensor by replacing masked values with zeros.
#         Parameters:
#         - x: Input tensor of shape (batch_size, sequence_length, features)
#         Returns:
#         - masked_x: The input tensor where masked values are set to zero.
#         """
#         mask = (x != 0.0).float()  # Create a mask where 1 indicates valid data
#         masked_x = x * mask  # Apply the mask by zeroing out the masked values
#         return masked_x, mask

#     def forward(self, x):
#         shared_inputs = []

#         # Process each task's input through the personalized CNN+GRU with multi-head attention
#         for i in range(len(x)):
#             masked_input, mask = self.apply_mask(x[i])

#             # Personalized CNN layer with residual connection
#             personal_cnn_out = self.pre_shared_personalized_cnn[i](masked_input.transpose(1, 2))  # (batch, channels, seq)
#             personal_cnn_out = personal_cnn_out.transpose(1, 2)  # Convert back to (batch, seq, features)

#             # Residual connection for CNN
#             if masked_input.size(-1) == personal_cnn_out.size(-1):
#                 personal_cnn_out = personal_cnn_out + masked_input

#             # Personalized GRU layer with residual connection
#             personal_gru_out, _ = self.pre_shared_personalized_gru[i](personal_cnn_out)  # (batch, seq, features)

#             # Residual connection for GRU
#             if personal_cnn_out.size(-1) == personal_gru_out.size(-1):
#                 personal_gru_out = personal_gru_out + personal_cnn_out

#             # Task-specific multi-head attention
#             attention_out, _ = self.task_specific_attention[i](personal_gru_out, personal_gru_out, personal_gru_out)  # (batch, seq, features)

#             # Take the mean over the sequence as the context vector
#             context_vector = attention_out.mean(dim=1)  # (batch, features)

#             # Add context vector as input to shared layers
#             shared_inputs.append(context_vector)

#         # Stack the context vectors for shared processing
#         shared_inputs = torch.stack(shared_inputs, dim=1)  # (batch, num_tasks, features)

#         # Shared LSTM layer with residual connection
#         shared_lstm_out, _ = self.shared_lstm(shared_inputs)

#         # Transform the input for residual addition
#         shared_residual = self.shared_residual_transform(shared_inputs)

#         # Add residual connection for the shared LSTM
#         shared_lstm_out = shared_lstm_out + shared_residual

#         # Personalized dense layers after shared layer
#         personalized_outputs = []
#         for i in range(shared_lstm_out.size(1)):  # Iterate over task outputs
#             shared_out = shared_lstm_out[:, i, :]

#             # Transform the shared LSTM output to match the dense layer output
#             residual_input = self.residual_transform[i](shared_out)

#             # Pass through the dense layer
#             personal_out = F.relu(self.personalized_fc[i](shared_out))
#             personal_out = self.dropout(personal_out)

#             # Add residual connection
#             personal_out = personal_out + residual_input
#             personalized_outputs.append(personal_out)

#         # Concatenate task-specific outputs
#         merged_personalized_output = torch.cat(personalized_outputs, dim=1)

#         # Final output layer
#         output = self.output_layer(merged_personalized_output)
#         return output


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

import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np

def validate_model(model, val_loader, criterion_mae, criterion_mse, device, model_type):
    """
    Validate the model on the validation set.

    Parameters:
    - model: The model to validate.
    - val_loader: DataLoader for validation data.
    - criterion_mae: Mean Absolute Error loss function.
    - criterion_mse: Mean Squared Error loss function.
    - device: Device to perform computations on (CPU or GPU).

    Returns:
    - avg_val_mae: The average validation MAE loss.
    - avg_val_rmse: The average validation RMSE loss.
    """
    model.eval()  # Set model to evaluation mode
    running_mae_loss = 0.0
    running_mse_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():  # Disable gradient calculation during validation
        for inputs, targets in val_loader:
            if model_type == "generalized":
                # print("generalized")
                inputs = [inp for inp in inputs]
                inputs = torch.tensor(np.array(inputs)).to(device)
                targets = targets.to(device)
            
                # Forward pass
                outputs = model(inputs)
            else:
            # Move data to the appropriate device (GPU or CPU)
                inputs = [inp.to(device) for inp in inputs]
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
            # Calculate MAE and MSE loss
            mae_loss = criterion_mae(outputs, targets)
            mse_loss = criterion_mse(outputs, targets)
            
            running_mae_loss += mae_loss.item()
            running_mse_loss += mse_loss.item()

    avg_val_mae = running_mae_loss / num_batches
    avg_val_rmse = np.sqrt(running_mse_loss / num_batches)  # RMSE is sqrt of MSE

    return avg_val_mae, avg_val_rmse

# def lr_lambda(epoch):
#     return 1 - ( 4.5e-8 * (epoch // 100))

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_type):
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
    - history: Dictionary containing the training and validation MAE and RMSE for each epoch.
    """
    # Define the optimizer and the loss functions (MAE and MSE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mae = torch.nn.L1Loss()  # Mean Absolute Error
    criterion_mse = torch.nn.MSELoss()  # Mean Squared Error (for RMSE calculation)
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize lists to store the history of losses
    history = {
        'train_mae': [],
        'train_rmse': [],
        'val_mae': [],
        'val_rmse': []
    }

    logger.info(f"Starting training for {epochs} epochs with learning rate {learning_rate}.")

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_mae_loss = 0.0
        running_mse_loss = 0.0
        num_batches = len(train_loader)

        # Add tqdm progress bar for each epoch
        with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device (GPU or CPU)
                if model_type == "generalized":
                    # print("generalized")
                    inputs = [inp for inp in inputs]
                    inputs = torch.tensor(np.array(inputs)).to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                
                    # Forward pass
                    outputs = model(inputs)
                elif model_type == "shared-layer":
                    inputs = [inp.to(device) for inp in inputs]
                    targets = targets.to(device)
                    # Zero the parameter gradients
                    optimizer.zero_grad()
            
                    # Forward pass
                    outputs = model(inputs)

                else:
                    # print(inputs)
                    # print("Not")
                    inputs = [inp.to(device) for inp in inputs]
                    # inputs = torch.tensor(np.array(inputs)).to(device)
                    targets = targets.to(device)
                    # Zero the parameter gradients
                    optimizer.zero_grad()
            
                    # Forward pass
                    outputs = model(inputs)
                
                # Compute losses
                mae_loss = criterion_mae(outputs, targets)
                mse_loss = criterion_mse(outputs, targets)
                
                # Backward pass and optimization
                mae_loss.backward()  
                optimizer.step()
                # scheduler.step()
                # print(f"Epoch {epoch+1}: Learning rate: {scheduler.get_last_lr()}")
                
                running_mae_loss += mae_loss.item()
                running_mse_loss += mse_loss.item()

                # Update progress bar
                pbar.set_postfix(mae_loss=mae_loss.item())
                pbar.update(1)
        
        # Calculate average training loss for the epoch
        avg_train_mae = running_mae_loss / num_batches
        avg_train_rmse = np.sqrt(running_mse_loss / num_batches)
        history['train_mae'].append(avg_train_mae)
        history['train_rmse'].append(avg_train_rmse)
        # Log training loss
        
        print(f"Epoch [{epoch + 1}/{epochs}], Training MAE: {avg_train_mae:.4f}, Training RMSE: {avg_train_rmse:.4f}")

        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training MAE: {avg_train_mae:.4f}, Training RMSE: {avg_train_rmse:.4f}")

        if val_loader != None:
            # Validate the model after each epoch
            avg_val_mae, avg_val_rmse = validate_model(model, val_loader, criterion_mae, criterion_mse, device,model_type)
            
            # Log validation loss
            logger.info(f"Epoch [{epoch + 1}/{epochs}], Validation MAE: {avg_val_mae:.4f}, Validation RMSE: {avg_val_rmse:.4f}")
            
            # Print validation loss
            print(f"Epoch [{epoch + 1}/{epochs}], Validation MAE: {avg_val_mae:.4f}, Validation RMSE: {avg_val_rmse:.4f}")
            
            # Store the losses in history

            history['val_mae'].append(avg_val_mae)
            history['val_rmse'].append(avg_val_rmse)

    # Return the model and the history of losses
    return model, history

# def evaluate_test(model,test_loader, device ,scaler, mask_value, model_type):
#     with torch.no_grad():  # Disable gradient calculation for evaluation
#         for inputs, targets in test_loader:
#             # Move data to device (GPU or CPU)

#             if model_type == "personalized":
#                 inputs = [inp.to(device) for inp in inputs]
#                 targets = targets.to(device)
#                 # print(targets[0])
#                 # print(targets)
#                 # Forward pass
#                 outputs = model(inputs)
#                 # print([[outputs[0]]])
#                 # # # Convert to numpy arrays for masking
#                 outputs_np = outputs.cpu().numpy()
#                 targets_np = targets.cpu().numpy()
#                 abs_patients_errors =  []
#                 squared_patients_errors = []

#                 # for i in range(len(targets_np)): #this prints a list of 12 values of each patient
#                 for j in range(len(targets_np)):
#                     if targets[j] != mask_value:
#                         sub_output = scaler.inverse_transform([outputs_np[j]])[0][0]
#                         sub_target = scaler.inverse_transform([targets_np[j]])[0][0]
#                         abs_patients_errors.append(abs(sub_output-sub_target))
#                         squared_patients_errors.append((sub_output-sub_target) ** 2)
#                 mae = np.mean(abs_patients_errors)
#                 rmse = np.sqrt(np.mean(squared_patients_errors))
#                 each_patient_mae, each_patient_rmse = None, None
#             elif model_type == "shared-layer" or model_type == "generalized":
#                 inputs = [inp for inp in inputs]
#                 inputs = torch.tensor(np.array(inputs)).to(device)
#                 targets = targets.to(device)
#                 # print(targets[0])
#                 # print(targets)
#                 # Forward pass
#                 outputs = model(inputs)
#                 # print([[outputs[0]]])
#                 # # # Convert to numpy arrays for masking
#                 outputs_np = outputs.cpu().numpy()
#                 targets_np = targets.cpu().numpy()

#                 abs_patients_errors =  {key: [] for key in range(13)}
#                 squared_patients_errors =  {key: [] for key in range(13)}
#                 for i in range(len(targets_np)): #this prints a list of 12 values of each patient
#                     for j in range(len(targets_np[i])):
#                         if targets[i][j] != mask_value:
#                             sub_output = scaler.inverse_transform([[outputs_np[i][j]]])[0][0]
#                             sub_target = scaler.inverse_transform([[targets_np[i][j]]])[0][0]
#                             abs_patients_errors[j].append(abs(sub_output-sub_target))
#                             squared_patients_errors[j].append((sub_output-sub_target) ** 2)
#                 each_patient_mae = []
#                 each_patient_rmse = []
#                 for i in range(len(squared_patients_errors)-1):
#                     mae = np.mean(abs_patients_errors[i])
#                     rmse = np.sqrt(np.mean(squared_patients_errors[i]))
#                     each_patient_mae.append(mae)
#                     each_patient_rmse.append(rmse)
#                 mae = np.mean(each_patient_mae)
#                 rmse = np.mean(each_patient_rmse)


#                     # print(f"{patients_list[i]}: RMSE: {rmse}, MAE: {mae}")
#                 # mae, rmse = abs_patients_errors, squared_patients_errors

#     return mae, rmse, each_patient_mae, each_patient_rmse


# def model_prediction(model,test_loader, device, model_type):
#     with torch.no_grad():  # Disable gradient calculation for evaluation
#         outputs_all_batches = []
#         targets_all_batches = []

#         for inputs, targets in test_loader:
#             # Move data to device (GPU or CPU)
#             if model_type == "personalized":
#                 inputs = [inp.to(device) for inp in inputs]

#             elif model_type == "shared-layer" or model_type == "generalized":
#                 inputs = [inp for inp in inputs]
#                 inputs = torch.tensor(np.array(inputs)).to(device)

#             targets = targets.to(device)
#             outputs = model(inputs)
#             targets_all_batches.append(targets)
#             outputs_all_batches.append(outputs)
#     return outputs_all_batches, targets_all_batches

def model_prediction(model,test_loader, device, model_type):
    with torch.no_grad():  # Disable gradient calculation for evaluation
        outputs_all_batches = []
        targets_all_batches = []
        
        # outputs_all_batches =  {key: [] for key in range(12)}
        # targets_all_batches =  {key: [] for key in range(12)}
        if model_type == "personalized":
            for inputs, targets in test_loader:
                # Move data to device (GPU or CPU)
                
                inputs = [inp.to(device) for inp in inputs]
                targets = targets.to(device)
                outputs = model(inputs)
                
                targets = targets.to(device)
                outputs = model(inputs)
                targets_all_batches.append(targets)
                outputs_all_batches.append(outputs)

        elif model_type == "shared-layer" or model_type == "generalized":
            for inputs, targets in test_loader:

                inputs = [inp for inp in inputs]
                inputs = torch.tensor(np.array(inputs)).to(device)

                targets = targets.to(device)
                outputs = model(inputs)
                targets_all_batches.append(targets)
                outputs_all_batches.append(outputs)

    return outputs_all_batches, targets_all_batches
                # print([[outputs[0]]])
                # # # Convert to numpy arrays for masking
                # outputs_np = outputs.cpu().numpy()
                # targets_np = targets.cpu().numpy()
                # abs_patients_errors =  []
                # squared_patients_errors = []
def evaluate_test(model,test_loader, device ,scaler, mask_value, model_type):
    outputs_all_batches, targets_all_batches = model_prediction(model,test_loader, device, model_type)

    if model_type == "personalized":
        abs_patients_errors =  []
        squared_patients_errors = []
        for one_batch in range(len(outputs_all_batches)):
            outputs_np = outputs_all_batches[one_batch].cpu().numpy()
            targets_np = targets_all_batches[one_batch].cpu().numpy()


            for j in range(len(targets_np)):
                    if targets_np[j] != mask_value:
                        sub_output = np.exp(scaler.inverse_transform([outputs_np[j]])[0][0])
                        sub_target = np.exp(scaler.inverse_transform([targets_np[j]])[0][0])
                        abs_patients_errors.append(abs(sub_output-sub_target))
                        squared_patients_errors.append((sub_output-sub_target) ** 2)
        mae = np.mean(abs_patients_errors)
        rmse = np.sqrt(np.mean(squared_patients_errors))
        return mae, rmse
    
    elif model_type == "shared-layer" or model_type == "generalized":
        abs_patients_errors =  {key: [] for key in range(12)}
        squared_patients_errors =  {key: [] for key in range(12)}
        for one_batch in range(len(outputs_all_batches)):
            outputs_np = outputs_all_batches[one_batch].cpu().numpy()
            targets_np = targets_all_batches[one_batch].cpu().numpy()
            for i in range(len(targets_np)): #this prints a list of 12 values of each patient
                for j in range(len(targets_np[i])):
                    if targets_np[i][j] != mask_value:
                        # outputs_np[i][j] 
                        sub_output = np.exp(scaler.inverse_transform([[outputs_np[i][j]]])[0][0])
                        sub_target = np.exp(scaler.inverse_transform([[targets_np[i][j]]])[0][0])
                        # print(sub_output)
                        # print(sub_target)
                        # print(sub_output)
                        # print(sub_target)
                        abs_patients_errors[j].append(abs(sub_output-sub_target))
                        squared_patients_errors[j].append((sub_output-sub_target) ** 2)
                        # print(abs_patients_errors)
        for patient in range(len(squared_patients_errors)):
            abs_patients_errors[patient] = np.mean(abs_patients_errors[patient])
            squared_patients_errors[patient] = np.sqrt(np.mean(squared_patients_errors[patient]))
        return abs_patients_errors, squared_patients_errors


# def evaluate_test_data(model, test_loader):
#     """
#     Evaluate the model on the test data and calculate RMSE.
    
#     Parameters:
#     - model: The trained model to evaluate.
#     - test_loader: DataLoader for the test data.
    
#     Returns:
#     - rmse: The calculated Root Mean Squared Error (RMSE) on the test data.
#     """
#     model.eval()  # Set the model to evaluation mode
#     criterion = nn.MSELoss()  # We use MSELoss and take the square root for RMSE
#     running_test_loss = 0.0
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
    
#     with torch.no_grad():  # Disable gradient calculation for evaluation
#         for inputs, targets in test_loader:
#             # Move data to device (GPU or CPU)
#             inputs = [inp.to(device) for inp in inputs]
#             targets = targets.to(device)
            
#             # Forward pass
#             outputs = model(inputs)
            
#             # Compute MSE loss
#             loss = criterion(outputs, targets)
#             running_test_loss += loss.item()
    
#     # Calculate the average MSE over all batches and take the square root for RMSE
#     avg_mse = running_test_loss / len(test_loader)
#     rmse = np.sqrt(avg_mse)
    
#     print(f"Test RMSE: {rmse:.4f}")
#     return rmse

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
