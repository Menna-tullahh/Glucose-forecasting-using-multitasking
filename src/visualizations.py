import matplotlib.pyplot as plt

def plot_loss_curves(history):
    """
    Plot the training and validation loss curves for MAE and RMSE.
    
    Parameters:
    - history: Dictionary containing lists of 'train_mae', 'train_rmse', 'val_mae', and 'val_rmse'.
    """
    epochs = range(1, len(history['train_mae']) + 1)

    # Plot MAE
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_mae'], 'o-', label='Training MAE')
    plt.plot(epochs, history['val_mae'], '*-', label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Generalized MAE')
    plt.legend()

    # Plot RMSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_rmse'], 'o-', label='Training RMSE')
    plt.plot(epochs, history['val_rmse'], '*-', label='Validation RMSE')
    # plt.title('')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Generalized RMSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_loss_curve(history):
    """
    Plot the training and validation loss curves for MAE and RMSE.
    
    Parameters:
    - history: Dictionary containing lists of 'train_mae', 'train_rmse', 'val_mae', and 'val_rmse'.
    """
    epochs = range(1, len(history['train_mae']) + 1)

    # Plot MAE
    plt.style.use('ggplot')
    plt.figure(figsize=(6, 5), dpi=600)
    
    # plt.subplot(1, 1, 1)
    plt.plot(epochs, history['train_mae'], 'o-', label='Training')
    plt.plot(epochs, history['val_mae'], '*-', label='Testing')
    plt.xlabel('Epochs', fontsize=16, color='black')
    plt.ylabel('MAE', fontsize=16, color='black')
    
    # plt.title('Generalized MAE')
    plt.legend()
    # plt.yscale('log')

    # Plot RMSE
    # plt.subplot(1, 1, 1)
    # plt.plot(epochs, history['train_rmse'], 'o-', label='Training RMSE')
    # plt.plot(epochs, history['val_rmse'], '*-', label='Validation RMSE')
    # # plt.title('')
    # plt.xlabel('Epochs')
    # plt.ylabel('RMSE')
    # plt.title('Generalized RMSE')
    # plt.legend()
    plt.tick_params(axis='x', labelsize=14, labelcolor='black')  # X-axis ticks
    plt.tick_params(axis='y', labelsize=14, labelcolor='black')  # Y-axis ticks
    ax = plt.gca()  # Get current axes
    for spine in ax.spines.values():
        spine.set_edgecolor('black')  # Set border color to black
        spine.set_linewidth(1)        # Set border line width to 2
    legend = plt.legend(fontsize=12, frameon=True)  # Enable legend frame
    legend.get_frame().set_edgecolor('black')      # Set legend border color to black
    legend.get_frame().set_linewidth(1)   
    plt.minorticks_on()
    plt.grid( linestyle='-', linewidth=0.7)
    plt.grid( which='minor', linestyle=':', linewidth=0.5)
    plt.tight_layout()
         # Set legend border line width to 2

    plt.show()