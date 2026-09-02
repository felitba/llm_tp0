import matplotlib.pyplot as plt

from plots.plot_theme import BODY, SPLIT_COLORS, apply_theme, legend_top_left


def plot_training_progress(
    losses,
    title: str | None = None,
    hyperparameters: str | None = None,
    show: bool = False,
):
    """
    Plots the training and validation losses over epochs.

    Args:
        losses: Training history as {"train": [...], "val": [...]}. The old
                alternating list format is still accepted for compatibility.
        title: Figure title shown above the loss plot.
        hyperparameters: Compact, optionally multi-line annotation below the plot.
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
        label='Pérdida en entrenamiento',
        marker=marker,
        color=SPLIT_COLORS["train"],
    )
    if val_losses:
        axes.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label='Pérdida en validación',
            marker=marker,
            color=SPLIT_COLORS["validation"],
        )
        # CHANGED (2026-09-01): aca iba una vertical en la epoca del checkpoint.
        # Sale de la figura -- la regla de seleccion vive en sus propias filminas,
        # no encima de la curva de perdida.

    _style_epoch_axis(axes, last_epoch)
    axes.set_xlabel('Época')
    axes.set_ylabel('Pérdida')
    legend_top_left(axes)
    figure.suptitle(title or 'Pérdida: entrenamiento vs. validación', y=0.98, fontweight='semibold')
    footer_height = 0.13 if hyperparameters else 0.04
    figure.tight_layout(rect=(0, footer_height, 1, 0.90))
    if hyperparameters:
        figure.text(
            0.5, 0.025, hyperparameters,
            ha='center', va='bottom', color=BODY, fontsize=8, linespacing=1.45,
        )
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
