
import torch.optim as optim
import torch
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
            if model_type == 'personalized':
                inputs = [inp for inp in inputs]
                inputs = torch.tensor(np.array(inputs)).to(device)
                targets = targets.to(device)
            
                # Forward pass
                outputs = model(inputs)
            else:
            # Move data to the appropriate device (GPU or CPU)
                # inputs = torch.tensor([inp.to(device) for inp in inputs])
                # targets = targets.to(device)
                inputs = [inp.cpu() for inp in inputs]
                inputs = torch.tensor(np.array(inputs)).to(device)
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
                    inputs = [inp for inp in inputs]
                    inputs = torch.tensor(np.array(inputs)).to(device)
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


def model_prediction(model,test_loader, device, model_type):
    with torch.no_grad():  # Disable gradient calculation for evaluation
        outputs_all_batches = []
        targets_all_batches = []
        
        # outputs_all_batches =  {key: [] for key in range(12)}
        # targets_all_batches =  {key: [] for key in range(12)}
        if model_type == "personalized":
            for inputs, targets in test_loader:
                # Move data to device (GPU or CPU)
                
                # inputs = [inp.to(device) for inp in inputs]
                # targets = targets.to(device)
                # outputs = model(inputs)
                
                # targets = targets.to(device)
                # outputs = model(inputs)
                
                inputs = [inp for inp in inputs]
                inputs = torch.tensor(np.array(inputs)).to(device)

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
