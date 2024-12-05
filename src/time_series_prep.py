import numpy as np
import pandas as pd
import os
import logging
import torch
from models.shared_layer import *

# Configure logging to save logs to a file and stream to console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging.log"),
        logging.StreamHandler()
    ]
)

def prepare_time_series_data(df, target_col, window_size, prediction_horizon):
    """
    Prepares time series data for forecasting by generating input-output pairs (X, y)
    using a sliding window approach.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the time series data.
    target_col (str): The name of the column to be predicted.
    window_size (int): The number of time steps in each input window (X).
    prediction_horizon (int): The number of time steps into the future to predict (y).

    Returns:
    X (np.ndarray): The feature matrix containing windows of time series data.
    y (np.ndarray): The target vector with corresponding future values.
    """
    X, y = [], []

    # Check if the target column exists
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found in DataFrame.")
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Extract the target series from the DataFrame
    target_series = df[target_col].values

    # Generate windows of data
    for i in range(len(target_series) - window_size - prediction_horizon + 1):
        X.append(target_series[i:i + window_size])  # Input window
        y.append(target_series[i + window_size + prediction_horizon - 1])  # Target value at prediction horizon

    # Convert lists to numpy arrays for model input
    X = np.array(X)
    y = np.array(y)

    logging.info(f"Prepared time series data with {len(X)} samples.")
    # print(X.shape)
    # print(y.shape)
    return X, y

def pad_sequences(sequences, max_len, pad_value=0.0):
    """
    Pads a list of sequences to ensure they all have the same length.

    Parameters:
    sequences (list of np.ndarray): The list of sequences or target vectors to be padded.
    max_len (int): The maximum length to pad the sequences to.
    pad_value (int or float): The value to pad with. Default is -1.

    Returns:
    np.ndarray: A padded array where all sequences are the same length.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad the sequence with pad_value to match the max_len
            if len(seq.shape) == 1:  # For 1D targets (y)
                padding = np.full((max_len - len(seq)), pad_value)
                padded_seq = np.concatenate((seq, padding))
            else:  # For 2D feature windows (X)
                padding = np.full((max_len - len(seq), seq.shape[1]), pad_value)
                padded_seq = np.vstack((seq, padding))
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)


def aggregate_patients(input_folder, target_col, window_size = 5, prediction_horizon = 1, test = True, patient_no = 12):
    """
    Aggregates time series data from multiple CSV files in a folder into feature matrix X and target vector y.
    All patients' data will be padded to the same length (with -1) based on the longest time series.

    Parameters:
    input_folder (str): The folder containing CSV files for different patients.
    target_col (str): The column name to be used as the target for prediction.
    window_size (int): The number of time steps to use in each input window (X). Default is 5.
    prediction_horizon (int): The number of time steps into the future to predict (y). Default is 1.

    Returns:
    X (np.ndarray): Combined feature matrix from all files, padded to the maximum sequence length.
    y (np.ndarray): Combined target vector from all files, padded to the maximum sequence length.
    """
    all_X, all_y = [], []
    max_len_X, max_len_y = 0, 0  # To track the maximum length of X and y

    if not os.path.exists(input_folder):
        logging.error(f"The folder {input_folder} does not exist.")
        raise FileNotFoundError(f"The folder {input_folder} does not exist.")
    counter = 0
    # Traverse through all folders and files in the input directory
    for top, _, files in os.walk(input_folder):
        if test == False:
            if os.path.basename(top) =='test':
                continue
        else:
            if os.path.basename(top) =='train':
                continue
        if files:
            csv_files = [file for file in os.listdir(top) if file.endswith(".csv")]
            if not csv_files:
                logging.warning(f"No CSV files found in {top}.")
                continue

            for file in csv_files:
                file_path = os.path.join(top, file)
                logging.info(f"Processing file: {file_path}")
                
                try:
                    if counter == patient_no: 
                        break
                    counter+=1
                    df = pd.read_csv(file_path)
                    
                    # Prepare time series data for this file
                    X, y = prepare_time_series_data(df, target_col, window_size, prediction_horizon)

                    # Append to the global lists
                    all_X.append(X)
                    all_y.append(y)

                    # Update the maximum length of sequences
                    max_len_X = max(max_len_X, len(X))
                    max_len_y = max(max_len_y, len(y))
                
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {e}")
                    continue

    # Pad all sequences to the same length
    if all_X:
        all_X = pad_sequences(all_X, max_len_X)
        all_y = pad_sequences(all_y, max_len_y)  # Reuse the same pad_sequences function for y

    logging.info(f"Aggregated data from {len(csv_files)} files with a total of {len(all_X)} samples.")

    return all_X, all_y

def sequential_split(dataset, train_ratio=0.8):
    """
    Splits the dataset into a sequential training and validation set based on the specified ratio.
    
    Parameters:
    - dataset: A TimeSeriesDataset instance.
    - train_ratio: The proportion of the dataset to include in the train split.
    
    Returns:
    - train_dataset: The training subset.
    - val_dataset: The validation subset.
    """
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size
    
    # Assuming the dataset is already sorted chronologically
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, total_samples)))
    
    return train_dataset, val_dataset


def prepare_data_loader(window_size,BATCH_SIZE, prediction_horizon, model_type, split_ratio = 0.8, df=None, df_test=None, output_folder_train = None, shuffle = True, patient_no = 12):

    if model_type == 'personalized':

        X, y = prepare_time_series_data(df, 'value', window_size, prediction_horizon)
        X = X.reshape(1,X.shape[0], X.shape[1])
        y = y.reshape(1,y.shape[0])
        print("Shape of X (features):", X.shape)
        print("Shape of y (targets):", y.shape)

        X_test, y_test = prepare_time_series_data(df_test, 'value', window_size, prediction_horizon)
        X_test = X_test.reshape(1,X_test.shape[0], X_test.shape[1])
        # print(y.shape)
        y_test = y_test.reshape(1,y_test.shape[0])
        print("Shape of X_test (features):", X_test.shape)
        print("Shape of y_test (targets):", y_test.shape)
        input_shape = (window_size, X.shape[1], 1)
        input_shape_test = (window_size, X_test.shape[1], 1)
        output_shape, output_shape_test = (1,), (1,)


    else:
        X, y = aggregate_patients(output_folder_train, 'value', window_size, prediction_horizon, test = False, patient_no = patient_no)
        print("Shape of X (features):", X.shape)
        print("Shape of y (targets):", y.shape)

        X_test, y_test = aggregate_patients(output_folder_train, 'value', window_size, prediction_horizon, test = True, patient_no = patient_no)
        print("Shape of X_test (features):", X_test.shape)
        print("Shape of y_test (targets):", y_test.shape)
        if model_type == 'shared-layer':
            input_shape = (window_size, X.shape[1], 1)
            input_shape_test = (window_size, X_test.shape[1], 1)
        elif model_type == 'generalized':
            input_shape = (window_size, X.shape[1], patient_no)
            input_shape_test = (window_size, X_test.shape[1], patient_no)
        output_shape, output_shape_test = (patient_no,), (patient_no,)



    dataset = TimeSeriesDataset(X, y)
    train_dataset, val_dataset = sequential_split(dataset, train_ratio=split_ratio)

        # Create a DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    all_train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


    test_dataset = TimeSeriesDataset(X_test, y_test)
    # Create a DataLoader for batching
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


    return train_loader, validation_loader, all_train_loader, test_loader, input_shape, input_shape_test, output_shape, output_shape_test


