import matplotlib.pyplot as plt

def plot_training_progress(losses, show: bool = False):
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

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    if val_losses:
        axes.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label='Validation Loss',
            marker='o'
        )
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Loss')
    axes.set_title('Training and Validation Loss Over Epochs')
    axes.legend()
    axes.grid(True)
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes
