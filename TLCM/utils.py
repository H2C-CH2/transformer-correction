from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from datasets import load_dataset
from jaxtyping import Int
from torch import Tensor
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Config:
    experiment: Literal["4.1", "4.2", "4.3", "4.4", "5.1", "5.2"]
    model_name: str
    revision: str
    device: str

    save: bool
    plot: bool

    run_both: bool

    # data loading
    seq_len: int = 128
    n_docs: int = 50
    batch_size: int = 1

    # 4.1
    debug: bool = True


def load_model(cfg: Config, revision: str = "main") -> TransformerBridge:
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


def get_tokens(bridge: TransformerBridge, cfg: Config) -> Int[Tensor, "n_docs seq"]:
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
