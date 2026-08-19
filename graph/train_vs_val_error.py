import matplotlib.pyplot as plt

def plot_training_progress(losses):
    """
    Plots the training and validation losses over epochs.

    Args:
        losses: Training history as {"train": [...], "val": [...]}. The old
                alternating list format is still accepted for compatibility.
    """
    if isinstance(losses, dict):
        train_losses = losses.get("train", [])
        val_losses = losses.get("val", [])
    else:
        raise TypeError("losses must be a dictionary with 'train' and 'val' keys.")

    epochs = len(train_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    if val_losses:
        plt.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label='Validation Loss',
            marker='o'
        )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
