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
