import argparse
import os
import sys

import torch
from transformer_lens.utilities.devices import get_device

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "TLCM")
)

from diff import compute_tlcm
from utils import Config, get_tokens, load_model

# from graph import plot_4_1, plot_4_4


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Perform Section 4.1 Experiment: cosine similarity between layer contributions"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace/TransformerLens model name",
    )
    parser.add_argument(
        "--revision", type=str, default="main", help="HuggingFace checkpoint"
    )
    parser.add_argument(
        "--both",
        type=bool,
        default=False,
        help="Run Section 4.4 Experiment (sublayer contribution cosine similarity) simultaneously",
    )
    parser.add_argument("--seqlen", type=int, default=128, help="seq len of prompts")
    parser.add_argument(
        "--n_docs", type=int, default=50, help="Max amount of prompts of seqlen"
    )
    parser.add_argument("--batch", type=int, default=4, help="batch size")

    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()

    return Config(
        experiment="4.1",
        model_name=args.model,
        revision=args.revision,
        device=get_device(),
        run_both=args.both,
        seq_len=args.seqlen,
        n_docs=args.n_docs,
        batch_size=args.batch,
        plot=args.plot,
        save=args.save,
    )


def main() -> None:
    cfg = parse_args()

    torch.set_grad_enabled(False)

    bridge = load_model(cfg, revision=cfg.revision)
    all_tokens = get_tokens(bridge, cfg)

    results = compute_tlcm(bridge=bridge, cfg=cfg, tokens=all_tokens)

    if cfg.save:
        os.makedirs("data", exist_ok=True)
        torch.save(results, "data/4_1.pt")

    if cfg.plot:
        raise NotImplementedError("Next Commit")


if __name__ == "__main__":
    main()
