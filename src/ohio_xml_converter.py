import os
import pandas as pd
import logging

# Configure logging to save logs to a file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging.log"),
        logging.StreamHandler()
    ]
)

def convert_xml_subfolder_to_csv(folder_path: str, saved_path: str) -> int:
    """
    Converts all XML files in a specified folder to CSV files.
    
    Parameters:
    folder_path (str): The folder containing the dataset.
    saved_path (str): The folder where CSV files will be saved.
    
    Returns:
    int: Status code, 1 for success.
    """
    try:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist.")
        
        os.makedirs(saved_path, exist_ok=True)

        xml_files = [file for file in os.listdir(folder_path) if file.endswith(".xml")]
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {folder_path}.")

        for file in xml_files:
            file_path = os.path.join(folder_path, file)
            data = pd.read_xml(file_path, xpath="//glucose_level//event")
            
            filename = os.path.splitext(file)[0]
            patient_id = filename.split("-")[0]
            data_type = filename.split("-")[2]

            output_file_path = os.path.join(saved_path, f"{patient_id}_{data_type}.csv")
            data.to_csv(output_file_path, index=False)
            logging.info(f"Converted {file} to CSV.")
        
        return 1
    except Exception as e:
        logging.error(f"Error converting XML files in {folder_path}: {e}")
        return 0

def convert_all_xml_to_csv(raw_data_path: str, processed_data_path: str) -> int:
    """
    Converts all XML files in a raw data directory and its subdirectories to CSV files.
    
    Parameters:
    raw_data_path (str): The root directory containing raw XML data.
    processed_data_path (str): The root directory where processed CSV files will be saved.
    
    Returns:
    int: Status code, 1 for success.
    """
    try:
        for top, _, files in os.walk(raw_data_path):
            if files:
                sub_path = os.path.relpath(top, raw_data_path)
                full_save_folder_path = os.path.join(processed_data_path, sub_path)
                logging.info(f"Processing folder {top}, saving to {full_save_folder_path}")
                convert_xml_subfolder_to_csv(top, full_save_folder_path)
        
        return 1
    except Exception as e:
        logging.error(f"Error processing XML files: {e}")
        return 0

if __name__ == "__main__":
    # Example usage:
    main_folder_path = 'data/ohio-data/raw'
    save_folder_path = 'data/ohio-data/processed'
    convert_all_xml_to_csv(main_folder_path, save_folder_path)
