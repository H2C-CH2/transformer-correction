from __future__ import annotations

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge

from TLCM.utils import DataConfig, ExperimentConfig


def self_cossim(
    x: Float[Tensor, "n_layers batch seq d_model"],
) -> Float[Tensor, "n_layers n_layers"]:
    """Cosine similarity of same contributions"""

    seq_len = x.shape[-2]

    x_norm = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    sim_sum = einops.einsum(x_norm, x_norm, "l1 b s d, l2 b s d -> b l1 l2")
    sim_sum = sim_sum.mean(dim=0) / seq_len
    sim_sum.fill_diagonal_(0.0)
    return sim_sum


def layer_diff(
    bridge, tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers batch seq d_model"]:
    """
    Returns only the contribution of every layer

    Contribution of layer i =
        blocks.{i}.hook_out - blocks.{i}.hook_in

    Note blocks.{i}.hook_in is equivalent to blocks.{i-1}.hook_out
    """
    n_layers = bridge.cfg.n_layers

    hook_names = {f"blocks.{i}.hook_out" for i in range(n_layers)}
    hook_names.add("blocks.0.hook_in")

    _, cache = bridge.run_with_cache(
        tokens,
        names_filter=lambda name: name in hook_names,
    )

    resid_pre = [cache["blocks.0.hook_in"]] + [
        cache[f"blocks.{i}.hook_out"] for i in range(n_layers - 1)
    ]
    resid_post = [cache[f"blocks.{i}.hook_out"] for i in range(n_layers)]

    return torch.stack([resid_post[i] - resid_pre[i] for i in range(n_layers)])


def _tlcm_layer(bridge, tokens, cfg):
    """Returns cosine similarities between layers' contributions"""
    accumulate = torch.zeros(
        bridge.cfg.n_layers, bridge.cfg.n_layers, device=cfg.device
    )  # [n_layers n_layers]

    for start in tqdm(range(0, len(tokens), cfg.batch_size)):
        batch = tokens[start : start + cfg.batch_size].to(cfg.device)  # [batch, seq]
        layer_contrib = layer_diff(bridge, batch)
        if cfg.debug:
            print("\nLayer contrib:", layer_contrib.shape)
        accumulate += self_cossim(layer_contrib)

    return {("layer", "layer"): accumulate / (len(tokens) // cfg.batch_size)}


def compute_tlcm(
    bridge: TransformerBridge,
    expcfg: ExperimentConfig,
    dcfg: DataConfig,
    tokens: Int[Tensor, "n_docs seq"],
    run_both: bool,
) -> dict[tuple[str, str], Float[Tensor, "n_layers n_layers"]]:
    """
    Returns mappings of pair-wise cosine similarity of layer/sublayer contributions
    to said pairs of layer/sublayer components
    """

    if (expcfg.experiment == "4.1" and run_both) or expcfg.experiment == "4.4":
        raise NotImplementedError()
        return _tlcm_components(bridge, tokens, dcfg, run_both)
    else:
        return _tlcm_layer(bridge, tokens, dcfg)
