import matplotlib.pyplot as plt

from plots.plot_theme import BASELINE, DASH, SPLIT_COLORS, apply_theme, legend_top_left, set_title


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
    last_epoch = max(epochs, len(val_losses))

    apply_theme()
    figure, axes = plt.subplots(figsize=(6, 4))
    # Markers help on a 6-epoch smoke run and bury the line on a 500-epoch one.
    marker = 'o' if last_epoch <= 40 else None
    axes.plot(
        range(1, epochs + 1),
        train_losses,
        label='Training loss',
        marker=marker,
        color=SPLIT_COLORS["train"],
    )
    if val_losses:
        axes.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label='Validation loss',
            marker=marker,
            color=SPLIT_COLORS["validation"],
        )
        # Early stopping restores this epoch's weights, so it is the epoch the
        # reported test numbers actually come from. It goes in the legend rather
        # than in an arrow on the plot: one more line, no new place to look.
        best_epoch = min(range(len(val_losses)), key=val_losses.__getitem__) + 1
        axes.axvline(
            best_epoch, color=BASELINE, linestyle=DASH,
            label=f'Best epoch ({best_epoch})',
        )

    _style_epoch_axis(axes, last_epoch)
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    set_title(axes, 'Training and Validation Loss Over Epochs')
    legend_top_left(axes)
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


def _style_epoch_axis(axes, last_epoch):
    """Epochs are whole numbers, and a one-epoch run has no range to divide.

    Left to autoscale, a single point makes matplotlib invent ticks at 0.945,
    0.960, … around it, which reads as a broken figure rather than a short run.
    """
    if last_epoch <= 1:
        axes.set_xlim(0.5, 1.5)
        axes.set_xticks([1])
        return
    axes.set_xlim(1 - 0.02 * last_epoch, last_epoch + 0.02 * last_epoch)
    # Ticks counted from epoch 1, not from a round zero: the first epoch is a real
    # point on the line, and a run that starts at an unlabelled 1 reads as cropped.
    step = max(1, -(-last_epoch // 9))
    axes.set_xticks(range(1, last_epoch + 1, step))
