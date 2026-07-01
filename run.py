import argparse
from dataclasses import replace

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens.utilities.devices import get_device

from TLCM.diff import compute_tlcm
from TLCM.graph import plot_cossims
from TLCM.utils import (
    DataConfig,
    ExperimentConfig,
    get_experiment,
    get_tokens,
    load_model,
    save_results,
)


# Section 4.1
def run_layer_contrib(run_both: bool, expcfg: ExperimentConfig, dcfg: DataConfig):
    bridge = load_model(expcfg, revision=expcfg.revision)
    all_tokens = get_tokens(bridge, dcfg)

    results: Float[Tensor, "n_layers n_layers"] = compute_tlcm(
        bridge=bridge, expcfg=expcfg, dcfg=dcfg, tokens=all_tokens, run_both=run_both
    )[("layer", "layer")]

    if dcfg.save:
        save_results(results, expcfg)
    if dcfg.plot:
        plot_cossims(res=[results], cfg=[expcfg], clamp=0.2, file_name="fig1")


# Section 4.2
def run_emergence(revisions: list[str], expcfg: ExperimentConfig, dcfg: DataConfig):
    """
    The expcfg in expcfgs should all be identical EXCEPT for the revision
    """

    expcfgs: list[ExperimentConfig] = [replace(expcfg, revision=rv) for rv in revisions]

    bridge = load_model(expcfg, revision=expcfg.revision)
    all_tokens = get_tokens(bridge, dcfg)

    results: list[Float[Tensor, "n_layers n_layers"]] = [
        compute_tlcm(
            bridge=bridge, expcfg=expcfg, dcfg=dcfg, tokens=all_tokens, run_both=False
        )[("layer", "layer")]
    ]

    for i in range(1, len(revisions)):
        bridge = load_model(expcfgs[i], revision=expcfgs[i].revision)
        results.append(
            compute_tlcm(
                bridge=bridge,
                expcfg=expcfgs[i],
                dcfg=dcfg,
                tokens=all_tokens,
                run_both=False,
            )[("layer", "layer")]
        )

    if dcfg.save:
        for i in range(len(expcfgs)):
            save_results(results[i], expcfgs[i])

    if dcfg.plot:
        plot_cossims(
            res=[r for r in results],
            cfg=expcfgs,
            clamp=0.2,
            file_name="figA9",
            title=[expcfg.revision for expcfg in expcfgs],
        )


# Section 4.4
def run_sublayer_contrib(run_both: bool, expcfg: ExperimentConfig, dcfg: DataConfig):
    bridge = load_model(expcfg, revision=expcfg.revision)
    all_tokens = get_tokens(bridge, dcfg)

    results: dict[tuple[str, str], Float[Tensor, "n_layers n_layers"]] = compute_tlcm(
        bridge=bridge, expcfg=expcfg, dcfg=dcfg, tokens=all_tokens, run_both=run_both
    )

    if dcfg.save:
        save_results(results, expcfg)
    if dcfg.plot:
        plot_cossims(
            [results[("attn", "attn")], results[("mlp", "mlp")]],
            cfg=[expcfg, expcfg],
            clamp=0.3,
            file_name="fig2a",
            title=["$M_{attn}$", "$M_{mlp}$"],
        )
        plot_cossims(
            [results[("attn", "mlp")]],
            cfg=[expcfg],
            clamp=0.3,
            file_name="figA6",
            title=["$M_{attn\\times MLP}$"],
            axis_names=[("Attn Index", "MLP Index")],
        )
        if run_both:
            plot_cossims(
                [results[("layer", "layer")]], cfg=[expcfg], clamp=0.2, file_name="fig1"
            )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, help="HuggingFace/TransformerLens model name"
    )
    parser.add_argument("--ckpt", action="extend", nargs="+", default=[])

    # Experiments
    parser.add_argument("--layer_contrib", action="store_true")
    parser.add_argument("--emergence", action="store_true")
    parser.add_argument("--token_act", action="store_true")
    parser.add_argument("--sublayer_contrib", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--eigen", action="store_true")

    # Section 4.1
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run layer_contrib and sublayer_contrib experiments simultaneously, using only one forward pass",
    )

    # Data Config
    parser.add_argument("--seq", type=int, default=120, help="Sequence length")
    parser.add_argument(
        "--n_docs",
        type=int,
        default=50,
        help="Maximum number of documents of sequence length",
    )
    parser.add_argument("--batch", type=int, default=4, help="Batch size")

    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--plot", action="store_true", default=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    expcfg = ExperimentConfig(
        experiment=get_experiment(args),
        model_name=args.model,
        revision=args.ckpt[0] if len(args.ckpt) > 0 else "main",
        device=get_device(),
    )

    dcfg = DataConfig(
        plot=args.plot,
        save=args.save,
        seq_len=args.seq,
        n_docs=args.n_docs,
        batch_size=args.batch,
        device=get_device(),
    )

    if args.layer_contrib:
        print(
            "Running Section 4.1 Experiment: cosine similarity between layer contributions"
        )
        run_layer_contrib(run_both=args.both, expcfg=expcfg, dcfg=dcfg)
    elif args.emergence:
        print("Running Section 4.2 Experiment: TLCM emergence over model checkpoints")
        run_emergence(revisions=args.ckpt, expcfg=expcfg, dcfg=dcfg)
    elif args.token_act:
        raise NotImplementedError
    elif args.sublayer_contrib:
        print(
            "Running Section 4.4 Experiment: cosine similarity between sublayer contributions"
        )
        run_sublayer_contrib(run_both=args.both, expcfg=expcfg, dcfg=dcfg)
    elif args.perturb:
        raise NotImplementedError
    elif args.eigen:
        raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
