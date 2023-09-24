import seaborn as sns
import torch
from matplotlib import pyplot as plt


def plot_2d_weights(weights: torch.Tensor, title: str, figsize=(10, 10), cmap="viridis"):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(weights.cpu().detach().numpy(), ax=ax, cmap=cmap)
    ax.set_title(title)
    plt.close(fig)
    return fig
