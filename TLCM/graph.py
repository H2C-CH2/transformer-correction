from pathlib import Path

import matplotlib.pyplot as plt
from jaxtyping import Float
from torch import Tensor
from utils import Config, make_path


def plot_4_1(
    res: dict[tuple[str, str], Float[Tensor, "n_layers n_layers"]],
    cfg: Config,
    clamp: float = 0.2,
) -> None:
    filename: Path = make_path("figures", cfg)

    res = res[("layer", "layer")].cpu().numpy().clip(-clamp, clamp)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(res, vmin=-clamp, vmax=clamp, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title(f"Figure 1: {cfg.model_name}")
    plt.tight_layout()

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Experiment 1 graph (Figure 1) saved to:", filename)
