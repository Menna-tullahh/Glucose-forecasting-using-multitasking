
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.temporal_block import *

class PersonalizedModelWithAttention(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=3, num_tcn_blocks=2, dropout=0.2):
        super(PersonalizedModelWithAttention, self).__init__()

        # Generalized TCN layers
        self.generalized_tcn = nn.Sequential(
            *[TemporalBlock(input_shape[-1] if i == 0 else 64, 64, kernel_size, stride=1, dilation=2 ** i, padding=(kernel_size - 1) * (2 ** i) // 2, dropout=dropout)
              for i in range(num_tcn_blocks)]
        )

        # Generalized GRU layer
        self.generalized_gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)

        # Generalized multi-head attention layer
        self.generalized_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Shared LSTM layer
        self.shared_lstm = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

        # Residual transformation for the shared LSTM input
        self.shared_residual_transform = nn.Linear(64, 128)

        # Fully connected layers
        self.fc = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.output_layer = nn.Linear(64, output_shape[0])

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
        x = x.squeeze(-1).squeeze(-1).permute(1,2,0)
        # Apply masking
        masked_input, mask = self.apply_mask(x)

        # TCN layer with residual connection
        tcn_out = self.generalized_tcn(masked_input.transpose(1, 2))  # (batch, channels, seq)
        tcn_out = tcn_out.transpose(1, 2)  # Convert back to (batch, seq, features)

        # GRU layer with residual connection
        gru_out, _ = self.generalized_gru(tcn_out)  # (batch, seq, features)

        # Residual connection for GRU
        if tcn_out.size(-1) == gru_out.size(-1):
            gru_out = gru_out + tcn_out

        # Multi-head attention
        attention_out, _ = self.generalized_attention(gru_out, gru_out, gru_out)  # (batch, seq, features)

        # Take the mean over the sequence as the context vector
        context_vector = attention_out.mean(dim=1)  # (batch, features)

        # Shared LSTM layer with residual connection
        shared_lstm_out, _ = self.shared_lstm(context_vector.unsqueeze(1))  # (batch, 1, features)
        shared_lstm_out = shared_lstm_out.squeeze(1)  # (batch, features)

        # Transform the input for residual addition
        shared_residual = self.shared_residual_transform(context_vector)  # (batch, features)

        # Ensure shapes match before adding
        if shared_lstm_out.size(-1) != shared_residual.size(-1):
            shared_residual = F.pad(shared_residual, (0, shared_lstm_out.size(-1) - shared_residual.size(-1)))

        # Add residual connection for the shared LSTM
        shared_lstm_out = shared_lstm_out + shared_residual

        # Fully connected layer
        fc_out = F.relu(self.fc(shared_lstm_out))
        fc_out = self.dropout(fc_out)

        # Final output layer
        output = self.output_layer(fc_out)
        return output

