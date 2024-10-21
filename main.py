import os
import torch
from torch.utils.data import DataLoader
from src.data import PriceVolumeDataset, create_or_load_scalers
from src.models import PriceVolumePredictor
from src.utils import estimate_iterations, calculate_random_guess_stats
from scripts.train import train_model
from src.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_probability_histogram,
)
from src.visualization import plot_training_history
import config

# Use the logger from config.py
logger = config.logger

def main():
    try:
        # Print configuration
        config.print_config()

        # Create or load scalers
        scalers = create_or_load_scalers(
            config.PROCESSED_DIR, update_scalers=config.UPDATE_SCALERS
        )

        # Estimate iterations
        estimated_iterations = estimate_iterations(
            f"{config.PROCESSED_DIR}/train/train.csv",
            config.SEQUENCE_LENGTH,
            config.BATCH_SIZE,
            sample_ratio=1.0,
        )

        # Create datasets and data loaders
        train_dataset = PriceVolumeDataset(
            f"{config.PROCESSED_DIR}/train/train.csv",
            config.SEQUENCE_LENGTH,
            scalers=scalers,
        )
        val_dataset = PriceVolumeDataset(
            f"{config.PROCESSED_DIR}/train/val.csv",
            config.SEQUENCE_LENGTH,
            scalers=scalers,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.TRAIN_NUM_WORKERS,
            prefetch_factor=config.PREFETCH_FACTOR,
            pin_memory=config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.VAL_NUM_WORKERS,
            prefetch_factor=config.PREFETCH_FACTOR,
            pin_memory=config.PIN_MEMORY,
        )

        logger.info("Datasets and loaders created successfully.")

        # Initialize model
        model = PriceVolumePredictor(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            output_size=config.OUTPUT_SIZE,
        ).to(config.DEVICE)

        # Train the model
        train_losses, val_losses, val_accuracies = train_model(
            model,
            train_loader,
            val_loader,
            config.NUM_EPOCHS,
            config.LEARNING_RATE,
            config.DEVICE,
            config.PATIENCE,
            estimated_iterations,
        )

        # Plot training history
        plot_training_history(train_losses, val_losses, val_accuracies)

        # Evaluate the model
        model.eval()
        all_targets = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch_seq, batch_tgt in val_loader:
                batch_seq = batch_seq.to(config.DEVICE)
                batch_tgt = batch_tgt.to(config.DEVICE)

                outputs = model(batch_seq)
                probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_targets.extend(batch_tgt.squeeze().cpu().numpy())
                all_probs.extend(probs)
                all_preds.extend(preds)

        # Compute and print metrics
        metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
        logger.info("Classification Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Plot evaluation results
        plot_confusion_matrix(all_targets, all_preds)
        plot_roc_curve(all_targets, all_probs)
        plot_probability_histogram(all_targets, all_probs)

        # Calculate and print random guess stats
        random_guess_stats = calculate_random_guess_stats(val_dataset)
        logger.info("Random Guess Statistics:")
        for stat, value in random_guess_stats.items():
            logger.info(f"{stat}: {value}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
