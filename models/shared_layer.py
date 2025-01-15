import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from models.temporal_block import *

# Initialize logger
logging.basicConfig(
    filename='training.log',  # File where logs will be saved
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    # filemode='w'  # Overwrite log file on each run
)
logger = logging.getLogger()



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




