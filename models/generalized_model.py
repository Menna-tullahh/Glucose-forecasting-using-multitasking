from models.shared_layer import *
from src.data_preprocessing import *
from models.shared_layer import *
from src.data_preprocessing import *

class GeneralizedModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        """
        Generalized LSTM model for multiple patients.
        
        Parameters:
        - input_shape: Shape of the input data (sequence_length, features).
        - output_shape: Shape of the output data (number of outputs, typically the number of patients).
        """
        super(GeneralizedModel, self).__init__()

        # CNN Layer before LSTM
        self.conv_layer = nn.Conv1d(
            in_channels=input_shape[-1], out_channels=64, kernel_size=4, stride=2, padding=1, dilation=2
        )
        
        # Shared LSTM layer (across all patients)
        self.shared_lstm = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

        # Fully connected layer for final output (shared for all patients)
        self.fc = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)

        # Output layer for predicting for all patients in one go
        self.output_layer = nn.Linear(64, output_shape[0])  # One output layer for all patients

    def apply_mask(self, x):
        """
        Apply masking to the input tensor by replacing masked values with zeros.
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, features)
        Returns:
        - masked_x: The input tensor where masked values are set to zero.
        """
        # print(np.array(x).shape)
        if not isinstance(x, torch.Tensor):
            # x = torch.from_numpy(x.detach().cpu().numpy())
            x = torch.from_numpy(np.array(x))

        mask = (x != -2.5).float()  # Create a mask where 1 indicates valid data
        masked_x = x * mask  # Apply the mask by zeroing out the masked values
        return masked_x, mask

    def forward(self, x):
        """
        Forward pass for the model.
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, features)
        Returns:
        - output: Tensor of shape (batch_size, num_patients)
        """
        # Apply mask to the input
        masked_input, mask = self.apply_mask(x)
        # print(masked_input.shape)

        conv_input = masked_input.squeeze(-1).permute(1, 0, 2) # (batch_size, num_features, sequence_length)
        # print(conv_input.shape)

        # Convolution layer
        conv_out = self.conv_layer(conv_input)  # (batch_size, 64, sequence_length after conv)
        
        # LSTM layer: transpose back to (batch_size, sequence_length, features) after conv
        lstm_input = conv_out.transpose(1, 2)  # (batch_size, sequence_length, 64)
        lstm_out, _ = self.shared_lstm(lstm_input)

        # Take the last output of the LSTM (corresponding to the last timestep)
        lstm_out_last = lstm_out[:, -1, :]  # (batch_size, 128)

        # Fully connected layer
        fc_out = F.relu(self.fc(lstm_out_last))  # (batch_size, 64)
        fc_out = self.dropout(fc_out)

        # Output layer for predictions
        output = self.output_layer(fc_out)  # (batch_size, num_patients)

        return output
    
# class GeneralizedModel(nn.Module):
#     def __init__(self, input_shape, output_shape):
#         """
#         Generalized LSTM model for multiple patients.
        
#         Parameters:
#         - input_shape: Shape of the input data (sequence_length, features).
#         - output_shape: Shape of the output data (number of outputs, typically the number of patients).
#         """
#         super(GeneralizedModel, self).__init__()

#         # CNN Layer before LSTM
#         self.conv_layer = nn.Conv1d(
#             in_channels=input_shape[-1], out_channels=64, kernel_size=4, stride=2, padding=1, dilation=2
#         )
        
#         # Shared LSTM layer (across all patients)
#         self.shared_lstm = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

#         # Fully connected layer for final output (shared for all patients)
#         self.fc = nn.Linear(128, 64)
#         self.dropout = nn.Dropout(0.2)

#         # Output layer for predicting for all patients in one go
#         self.output_layer = nn.Linear(64, output_shape[0])  # One output layer for all patients

#     def apply_mask(self, x):
#         """
#         Apply masking to the input tensor by replacing masked values with zeros.
#         Parameters:
#         - x: Input tensor of shape (batch_size, sequence_length, features)
#         Returns:
#         - masked_x: The input tensor where masked values are set to zero.
#         """
#         # print(np.array(x).shape)
#         if not isinstance(x, torch.Tensor):
#             x = torch.from_numpy(np.array(x))
#         mask = (x != -2.5).float()  # Create a mask where 1 indicates valid data
#         masked_x = x * mask  # Apply the mask by zeroing out the masked values
#         return masked_x, mask

#     def forward(self, x):
#         """
#         Forward pass for the model.
#         Parameters:
#         - x: Input tensor of shape (batch_size, sequence_length, features)
#         Returns:
#         - output: Tensor of shape (batch_size, num_patients)
#         """
#         # Apply mask to the input
#         masked_input, mask = self.apply_mask(x)
#         # print(masked_input.shape)

#         conv_input = masked_input.squeeze(-1).permute(1, 0, 2) # (batch_size, num_features, sequence_length)
#         # print(conv_input.shape)

#         # Convolution layer
#         conv_out = self.conv_layer(conv_input)  # (batch_size, 64, sequence_length after conv)
        
#         # LSTM layer: transpose back to (batch_size, sequence_length, features) after conv
#         lstm_input = conv_out.transpose(1, 2)  # (batch_size, sequence_length, 64)
#         lstm_out, _ = self.shared_lstm(lstm_input)

#         # Take the last output of the LSTM (corresponding to the last timestep)
#         lstm_out_last = lstm_out[:, -1, :]  # (batch_size, 128)

#         # Fully connected layer
#         fc_out = F.relu(self.fc(lstm_out_last))  # (batch_size, 64)
#         fc_out = self.dropout(fc_out)

#         # Output layer for predictions
#         output = self.output_layer(fc_out)  # (batch_size, num_patients)

#         return output