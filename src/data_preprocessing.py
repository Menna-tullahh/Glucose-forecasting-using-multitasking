import pandas as pd
import os
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shutil
import numpy as np
from sklearn.impute import KNNImputer

# Configure logging to save logs to a file and stream to console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging.log"),
        logging.StreamHandler()
    ]
)

def timestamp_type(df, timestamp_col):
    """
    Converts the specified timestamp column to a datetime object and sets it as the DataFrame index.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    timestamp_col (str): The name of the column containing the timestamps.

    Returns:
    pd.DataFrame: DataFrame with the timestamp column converted and set as the index.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame.")
    
    # Convert timestamp column to datetime
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col].astype(str), format="%d-%m-%Y %H:%M:%S")
        # Set the timestamp column as index
        df.set_index(timestamp_col, inplace=True)
    except ValueError as e:
        raise ValueError(f"Error converting timestamp column to datetime: {e}")

    return df


def resample_data(df, freq, agg_func):
    """
    Resamples the data based on the specified frequency and aggregation function.

    Parameters:
    df (pd.DataFrame): The DataFrame to resample.
    freq (str): The resampling frequency (e.g., '5T' for 5 minutes).
    agg_func (str): The aggregation function to use ('mean', 'sum', 'median').

    Returns:
    pd.DataFrame: Resampled DataFrame.
    """
    agg_map = {
        'mean': df.resample(freq).mean,
        'sum': df.resample(freq).sum,
        'median': df.resample(freq).median
    }
    try:
        df = agg_map[agg_func]()  # Resample and apply aggregation
    except KeyError:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")

    return df

def normalize_data(df):
    """
    Normalizes the DataFrame values to be between 0 and 1 using Min-Max normalization.

    Parameters:
    df (pd.DataFrame): The DataFrame to normalize.

    Returns:
    pd.DataFrame: Normalized DataFrame.
    """
    # Apply log scaling
    df = df.map(np.log)

    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    df[df.columns] = scaler.fit_transform(df)  # Normalize all columns
    # print('Min value:',df['value'].min())
    return df, scaler

# def fill_na(df):
#     """
#     Fills missing values in the DataFrame with 0.

#     Parameters:
#     df (pd.DataFrame): The DataFrame with potential missing values.

#     Returns:
#     pd.DataFrame: DataFrame with missing values filled.
#     """
#     df.fillna(0, inplace=True)
#     return df

def imputation(df):
    df = df.interpolate(method='linear')
    return df

# def fill_na_with_knn(df, column, n_neighbors=2, test = False):
#     """
#     Fills missing values in the specified column of the DataFrame using KNN
#     if the gap is less than 3 readings; otherwise, fills with 0.

#     Parameters:
#     df (pd.DataFrame): The DataFrame with potential missing values.
#     column (str): The column name in the DataFrame to apply the method.
#     n_neighbors (int): Number of neighbors to consider for KNN.

#     Returns:
#     pd.DataFrame: DataFrame with missing values filled as per the specified method.
#     """
#     # Work on a copy of the DataFrame
#     df_copy = df.copy()
#     if test:
#         df_copy.fillna(0, inplace=True)
#         return df_copy
#     # Identify gaps and their sizes
#     is_missing = df_copy[column].isna()
#     missing_groups = (is_missing != is_missing.shift()).cumsum()
    
#     for group, size in is_missing.groupby(missing_groups).sum().items():
#         if size < 3 and size > 0:  # Apply KNN if gap is less than 3
#             mask = missing_groups == group
#             observed_data = df_copy[[column]].copy()
            
#             # Use KNNImputer to fill the small gaps
#             imputer = KNNImputer(n_neighbors=n_neighbors)
#             observed_data_filled = imputer.fit_transform(observed_data)
#             df_copy.loc[mask, column] = observed_data_filled[mask]
#         elif size >= 3:  # Fill with 0 if gap is 3 or more
#             df_copy.loc[missing_groups == group, column] = 0
    

#     return df_copy

def interpolate_small_gaps(df, column, max_gap=12):
    """
    Interpolates gaps in the specified column of the DataFrame
    if the number of consecutive NaNs is less than or equal to max_gap.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    max_gap (int): Maximum number of consecutive NaNs to interpolate.
    
    Returns:
    pd.DataFrame: DataFrame with small gaps interpolated.
    """
    data = df[column]
    is_nan = data.isna()
    
    # Identify gaps smaller than or equal to max_gap
    gap_sizes = is_nan.groupby((~is_nan).cumsum()).cumsum()
    to_interpolate = (is_nan & (gap_sizes <= max_gap))
    
    # Interpolate only the small gaps
    data[to_interpolate] = data.interpolate(method='linear', limit_direction='forward', limit=max_gap).loc[to_interpolate]
    data.fillna(0, inplace=True)
    return df

def fill_na(df, test):
    """
    Fills missing values in the DataFrame with 0.

    Parameters:
    df (pd.DataFrame): The DataFrame with potential missing values.

    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """
    if test:
        df.fillna(0, inplace=True)
    else:
        df = interpolate_small_gaps(df, 'value')
    return df

def preprocessing_df(df, timestamp_col, freq, agg_func, test):
    """
    Applies a series of preprocessing steps: converts timestamp to datetime, resamples the data, normalizes values, and fills missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    timestamp_col (str): The name of the timestamp column.
    freq (str): The resampling frequency (e.g., '5T' for 5 minutes).
    agg_func (str): The aggregation function to apply during resampling ('mean', 'sum', 'median').

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = timestamp_type(df, timestamp_col)
    df = resample_data(df, freq, agg_func)
    df, scaler= normalize_data(df)
    df = fill_na(df, test)
    # df = imputation(df)
    return df, scaler



    
def preprocessing_df_old(df, timestamp_col, freq, agg_func, test):
    """
    Applies a series of preprocessing steps: converts timestamp to datetime, resamples the data, normalizes values, and fills missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.
    timestamp_col (str): The name of the timestamp column.
    freq (str): The resampling frequency (e.g., '5T' for 5 minutes).
    agg_func (str): The aggregation function to apply during resampling ('mean', 'sum', 'median').

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = timestamp_type(df, timestamp_col)
    df = resample_data(df, freq, agg_func)
    if test == False:
        df, scaler= normalize_data(df)
        df = fill_na(df)
        return df, scaler
    else:
        df = fill_na(df)
        return df, None

def process_all_csv_files(input_folder, output_folder, timestamp_col='ts', freq='5T', agg_func='mean'):
    """
    Processes all CSV files in the specified input folder, applies preprocessing, and saves the processed files to the output folder.

    Parameters:
    input_folder (str): The folder containing the CSV files to process.
    output_folder (str): The folder where processed CSV files will be saved.
    timestamp_col (str): The name of the column containing the timestamps. Default is 'ts'.
    freq (str): The resampling frequency. Default is '5T' (5 minutes).
    agg_func (str): The aggregation function to apply during resampling. Default is 'mean'.

    Returns:
    int: Status code, 1 for success.
    """
    try:
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"The folder {input_folder} does not exist.")
        
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            # raise FileNotFoundError(f"The folder {output_folder} already exists.")

        # Traverse through all folders and files in the input directory
        for top, _, files in os.walk(input_folder):
            if files:
                sub_path = os.path.relpath(top, input_folder)
                full_save_folder_path = os.path.join(output_folder, sub_path)
                logging.info(f"Processing folder {top}, saving to {full_save_folder_path}")

                # Create output directories if they don't exist
                os.makedirs(full_save_folder_path, exist_ok=True)

                csv_files = [file for file in os.listdir(top) if file.endswith(".csv")]
                if not csv_files:
                    logging.warning(f"No CSV files found in {top}.")
                    continue

                for file in csv_files:
                    file_path = os.path.join(top, file)
                    df = pd.read_csv(file_path)

                    # Skip empty CSV files
                    if df.empty:
                        logging.warning(f"The file {file} is empty and was skipped.")
                        continue
                    
                    logging.info(f"Processing file {file}...")

                    # Apply preprocessing
                    try:
                        if top[-4] == "test":
                            print("Test")
                            processed_df, scaler= preprocessing_df(df, timestamp_col=timestamp_col, freq=freq, agg_func=agg_func, test = True)
                        else:
                            processed_df, scaler= preprocessing_df(df, timestamp_col=timestamp_col, freq=freq, agg_func=agg_func, test = False)

                        # Save the processed DataFrame
                        output_file_path = os.path.join(full_save_folder_path, file)
                        processed_df.to_csv(output_file_path, index=True)  # Ensure index is saved
                        logging.info(f"Processed and saved {file} to {output_file_path}.")
                        
                    except Exception as e:
                        logging.error(f"Error processing file {file}: {e}")
        
        return scaler
    except Exception as e:
        logging.error(f"Error processing CSV files in {input_folder}: {e}")
        return 0
    
def process_all_csv_files_personalized(input_folder, output_folder, timestamp_col='ts', freq='5 min', agg_func='mean'):
    """
    Processes all CSV files in the specified input folder, applies preprocessing, and saves the processed files to the output folder.

    Parameters:
    input_folder (str): The folder containing the CSV files to process.
    output_folder (str): The folder where processed CSV files will be saved.
    timestamp_col (str): The name of the column containing the timestamps. Default is 'ts'.
    freq (str): The resampling frequency. Default is '5T' (5 minutes).
    agg_func (str): The aggregation function to apply during resampling. Default is 'mean'.

    Returns:
    int: Status code, 1 for success.
    """
    try:
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"The folder {input_folder} does not exist.")
        
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            # raise FileNotFoundError(f"The folder {output_folder} already exists.")
        train_patients = {}
        test_patients = {}
        # Traverse through all folders and files in the input directory
        for top, _, files in os.walk(input_folder):
            if files:
                sub_path = os.path.relpath(top, input_folder)
                full_save_folder_path = os.path.join(output_folder, sub_path)
                logging.info(f"Processing folder {top}, saving to {full_save_folder_path}")

                # Create output directories if they don't exist
                os.makedirs(full_save_folder_path, exist_ok=True)

                csv_files = [file for file in os.listdir(top) if file.endswith(".csv")]
                if not csv_files:
                    logging.warning(f"No CSV files found in {top}.")
                    continue

                for file in csv_files:
                    patient_code = file[:3]
                    # print(patient_code)
                    file_path = os.path.join(top, file)

                    df = pd.read_csv(file_path)
                    # print(sub_path.split("\\")[1])
                    
                    # Skip empty CSV files
                    if df.empty:
                        logging.warning(f"The file {file} is empty and was skipped.")
                        continue
                    
                    logging.info(f"Processing file {file}...")

                    # Apply preprocessing
                    try:
                        if top[-4] == "test":
                            print("Test")
                            processed_df, scaler= preprocessing_df(df, timestamp_col=timestamp_col, freq=freq, agg_func=agg_func, test = True)
                        else:
                            processed_df, scaler= preprocessing_df(df, timestamp_col=timestamp_col, freq=freq, agg_func=agg_func, test = False)

                        if sub_path.split("\\")[1] == "train":
                            train_patients[patient_code] =  processed_df
                        else:
                            test_patients[patient_code] =  processed_df
                        # Save the processed DataFrame
                        output_file_path = os.path.join(full_save_folder_path, file)
                        processed_df.to_csv(output_file_path, index=True)  # Ensure index is saved
                        logging.info(f"Processed and saved {file} to {output_file_path}.")
                        
                    except Exception as e:
                        logging.error(f"Error processing file {file}: {e}")
        
        return scaler, train_patients, test_patients
    except Exception as e:
        logging.error(f"Error processing CSV files in {input_folder}: {e}")
        return 0
    
    

if __name__ == "__main__":
    input_folder = 'data/ohio-data/processed'
    output_folder_train = 'data/ohio-data/processed/cleaned'  # Create a subfolder for processed files
    # output_folder_test = 'data/ohio-data/processed/cleaned_test'
    scaler = process_all_csv_files(input_folder, output_folder_train, timestamp_col='ts', freq='5T', agg_func='mean')
    # process_all_csv_files(input_folder, output_folder_test, timestamp_col='ts', freq='5T', agg_func='mean', test = True)
