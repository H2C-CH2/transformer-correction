from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from datasets import load_dataset
from jaxtyping import Int
from torch import Tensor
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    experiment: Literal["4.1", "4.2", "4.3", "4.4", "5.1", "5.2"]
    model_name: str
    revision: str
    device: str


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    seq_len: int
    n_docs: int
    batch_size: int
    device: str
    plot: bool
    save: bool
    debug: bool = True


def load_model(cfg: ExperimentConfig, revision: str = "main") -> TransformerBridge:
    """Load model (checkpoint) from HuggingFace and convert into a TransformerBridge"""

    # not entirely sure if TL directly allows for creating revisions, so this is the workaround

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, revision=revision, trust_remote_code=True
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=cfg.device,
        low_cpu_mem_usage=True,
    )

    bridge = TransformerBridge.boot_transformers(
        cfg.model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
    )

    bridge.enable_compatibility_mode()
    return bridge


def get_tokens(bridge: TransformerBridge, cfg: DataConfig) -> Int[Tensor, "n_docs seq"]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"] for row in dataset if len(row["text"]) > 100][: cfg.n_docs]

    token_seqs: list[Tensor] = []
    for text in texts:
        toks = bridge.to_tokens(text, prepend_bos=True)[0]
        if toks.shape[0] >= cfg.seq_len:
            token_seqs.append(toks[: cfg.seq_len])

    if len(token_seqs) == 0:
        raise ValueError(f"No documents had >= {cfg.seq_len} tokens.")

    return torch.stack(token_seqs).to(cfg.device)


# Saving and Loading Below
def encode_model_name(repo: str) -> str:
    return repo.replace("/", "__")


def decode_model_name(name: str) -> str:
    return name.replace("__", "/", 1)


def make_path(src: str, cfg: ExperimentConfig, file_name: str = "") -> Path:

    path = Path(
        f"{src}/{cfg.experiment}/{file_name if file_name != '' else encode_model_name(cfg.model_name)}{'.pt' if src == 'data' else '.png'}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def save_results(
    result: Any, cfg: ExperimentConfig, extra_metadata: Optional[dict] = None
) -> Path:
    path = make_path("data", cfg, "")
    metadata = {
        "cfg": cfg,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra_metadata or {}),
    }
    torch.save({"data": result, "metadata": metadata}, path)
    return path


def load_results(path: Path, map_location: str = "cpu") -> Any:
    print("Loading data from:", path)
    return torch.load(path, map_location=map_location, weights_only=False)


def get_experiment(args) -> Literal["4.1", "4.2", "4.3", "4.4", "5.1", "5.2"]:
    if args.layer_contrib:
        return "4.1"
    elif args.emergence:
        return "4.2"
    elif args.token_act:
        return "4.3"
    elif args.sublayer_contrib:
        return "4.4"
    elif args.perturb:
        return "5.1"
    elif args.eigen:
        return "5.2"
    else:
        raise NameError("Experiment not found")
