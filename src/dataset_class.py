import torch

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
