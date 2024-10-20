# This file was automatically generatedimport matplotlib.pyplot as plt
import numpy as np

def plot_training_history(train_losses, val_losses, val_accuracies):
    """
    Plot the training history including train loss, validation loss, and validation accuracy.
    
    :param train_losses: List of training losses per epoch
    :param val_losses: List of validation losses per epoch
    :param val_accuracies: List of validation accuracies per epoch
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions(true_prices, predicted_prices, num_samples=100):
    """
    Plot a sample of true prices vs predicted prices.
    
    :param true_prices: Array of true price values
    :param predicted_prices: Array of predicted price values
    :param num_samples: Number of samples to plot (default: 100)
    """
    # Sample a subset of the data
    indices = np.random.choice(len(true_prices), num_samples, replace=False)
    true_sample = true_prices[indices]
    pred_sample = predicted_prices[indices]

    plt.figure(figsize=(12, 6))
    plt.scatter(true_sample, pred_sample, alpha=0.5)
    plt.plot([true_sample.min(), true_sample.max()], [true_sample.min(), true_sample.max()], 'r--', lw=2)
    plt.xlabel('True Prices')
    plt.ylabel('Predicted Prices')
    plt.title('True vs Predicted Prices')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for the given model.
    
    :param model: Trained model with feature_importances_ attribute
    :param feature_names: List of feature names
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def plot_price_trend(dates, prices, window_size=30):
    """
    Plot the price trend with a moving average.
    
    :param dates: Array of dates
    :param prices: Array of prices
    :param window_size: Size of the moving average window (default: 30)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label='Price')
    plt.plot(dates, np.convolve(prices, np.ones(window_size), 'valid') / window_size, 
             label=f'{window_size}-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Trend with Moving Average')
    plt.legend()
    plt.tight_layout()
    plt.show()