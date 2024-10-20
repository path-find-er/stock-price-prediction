import torch
import torch.nn as nn
from tqdm import tqdm
import os
import config
import logging


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    patience: int,
    estimated_iterations: int,
) -> tuple[list[float], list[float], list[float]]:
    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0
    epochs_no_improve: int = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    checkpoint_dir: str = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    prev_train_loss: float = float("inf")
    prev_val_loss: float = float("inf")
    prev_val_accuracy: float = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss: float = 0.0
        batch_count: int = 0

        # Training phase
        train_loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=estimated_iterations,
        )
        for batch_seq, batch_tgt in train_loop:
            batch_seq = batch_seq.to(device, non_blocking=True)
            batch_tgt = batch_tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            output: torch.Tensor = model(batch_seq)
            loss: torch.Tensor = criterion(output.squeeze(), batch_tgt.squeeze())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            train_loop.set_postfix(loss=total_loss / batch_count)

        avg_train_loss: float = total_loss / batch_count
        train_losses.append(avg_train_loss)
        train_loss_diff: float = (
            ((avg_train_loss - prev_train_loss) / prev_train_loss * 100)
            if epoch > 0
            else 0
        )

        # Validation phase
        model.eval()
        val_loss: float = 0
        val_batch_count: int = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_seq, batch_tgt in val_loader:
                batch_seq = batch_seq.to(device, non_blocking=True)
                batch_tgt = batch_tgt.to(device, non_blocking=True)

                output: torch.Tensor = model(batch_seq)
                val_loss += criterion(output.squeeze(), batch_tgt.squeeze()).item()
                val_batch_count += 1

                probs: torch.Tensor = torch.sigmoid(output.squeeze())
                preds: torch.Tensor = (probs > 0.5).float()
                all_preds.append(preds.cpu())
                all_targets.append(batch_tgt.squeeze().cpu())

        avg_val_loss: float = val_loss / val_batch_count
        all_preds: torch.Tensor = torch.cat(all_preds)
        all_targets: torch.Tensor = torch.cat(all_targets)
        val_accuracy: float = (all_preds == all_targets).float().mean().item()

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        val_loss_diff: float = (
            ((avg_val_loss - prev_val_loss) / prev_val_loss * 100) if epoch > 0 else 0
        )
        val_accuracy_diff: float = (
            ((val_accuracy - prev_val_accuracy) / prev_val_accuracy * 100)
            if epoch > 0
            else 0
        )

        logging.info("-" * 50)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Train Loss: {avg_train_loss:.6f} (Diff: {train_loss_diff:.2f}%)")
        logging.info(f"Val Loss: {avg_val_loss:.6f} (Diff: {val_loss_diff:.2f}%)")
        logging.info(
            f"Val Accuracy: {val_accuracy:.4f} (Diff: {val_accuracy_diff:.2f}%)"
        )

        # Early Stopping and Checkpoint Saving
        should_save_checkpoint: bool = False
        if val_accuracy > best_val_accuracy:
            logging.info("New best validation accuracy!")
            best_val_accuracy = val_accuracy
            should_save_checkpoint = True
        if avg_val_loss < best_val_loss:
            logging.info("New best validation loss!")
            best_val_loss = avg_val_loss
            should_save_checkpoint = True
        if avg_train_loss < prev_train_loss:
            logging.info("New best training loss!")
            should_save_checkpoint = True

        if should_save_checkpoint:
            checkpoint_path: str = os.path.join(
                checkpoint_dir, f"model_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info("No improvement in monitored metrics.")
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered.")
                break
        logging.info("-" * 50)

        # Update previous values for next epoch
        prev_train_loss = avg_train_loss
        prev_val_loss = avg_val_loss
        prev_val_accuracy = val_accuracy

    logging.info("Training complete.")
    return train_losses, val_losses, val_accuracies
