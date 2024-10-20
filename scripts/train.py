import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, patience, estimated_iterations):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        # Training phase
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=estimated_iterations)
        for batch_seq, batch_tgt in train_loop:
            batch_seq = batch_seq.to(device, non_blocking=True)
            batch_tgt = batch_tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(batch_seq)
            loss = criterion(output.squeeze(), batch_tgt.squeeze())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            train_loop.set_postfix(loss=total_loss / batch_count)

        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_seq, batch_tgt in val_loader:
                batch_seq = batch_seq.to(device, non_blocking=True)
                batch_tgt = batch_tgt.to(device, non_blocking=True)

                output = model(batch_seq)
                val_loss += criterion(output.squeeze(), batch_tgt.squeeze()).item()
                val_batch_count += 1

                predictions = (torch.sigmoid(output.squeeze()) > 0.5).float()
                correct_predictions += (predictions == batch_tgt.squeeze()).sum().item()
                total_predictions += batch_tgt.size(0)

        avg_val_loss = val_loss / val_batch_count
        val_accuracy = correct_predictions / total_predictions

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses, val_accuracies