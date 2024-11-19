import torch
import torch.nn as nn
from torch.nn import functional as F

class PersonalizedModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PersonalizedModel, self).__init__()

        # CNN Layer before LSTM
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_shape[-1], out_channels=64, kernel_size=4, stride=2, padding=1 , dilation=2) 
            for _ in range(output_shape[0])
        ])

        # Shared LSTM layer
        self.shared_lstm = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

        # Personalized layers
        self.personalized_fc = nn.ModuleList([nn.Linear(128, 64) for _ in range(output_shape[0])])
        self.dropout = nn.Dropout(0.2)

        # Output layer
        self.output_layer = nn.Linear(64 * output_shape[0], output_shape[0])

    def apply_mask(self, x):
        # print(x)
        x = x[0]
        mask = (x != -2.5).float() # Create a mask where 1 indicates valid data
        # print(mask.shape)
        # print(x.shape)
        masked_x = x * mask  # Apply the mask by zeroing out the masked values
        return masked_x, mask
    
    def forward(self, x):
        shared_outputs = []
        # print(x[0])
        masked_input, mask = self.apply_mask(x)
        # print(masked_input)
        # Ensure input has 3 dimensions for Conv1d
        if masked_input.dim() == 2:  # If input is (batch_size, sequence_length)
            masked_input = masked_input.unsqueeze(-1)  # Add a channel dimension to become (batch_size, sequence_length, 1)
        # if masked_input.dim() > 3:  # If input is (batch_size, sequence_length)
        #     masked_input = masked_input.squeeze(-1)  # Add a channel dimension to become (batch_size, sequence_length, 1)

        for i in range(len(self.conv_layers)):
            # Transpose to match Conv1d expected input (batch_size, channels, sequence_length)
            conv_out = self.conv_layers[i](masked_input.transpose(1, 2))
            
            # LSTM expects input in the shape [batch_size, sequence_length, features], so we transpose back
            lstm_input = conv_out.transpose(1, 2)
            lstm_out, _ = self.shared_lstm(lstm_input)
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
        return output