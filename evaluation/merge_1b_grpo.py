"""Merge a Llasa-1B GRPO single-GPU FSDP checkpoint into HF model (CPU)."""

import os
import sys
import argparse
import torch
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "HKUSTAudio/Llasa-1B-Multilingual"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory containing actor/model_world_size_1_rank_0.pt")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    actor_dir = os.path.join(args.ckpt_dir, "actor")
    shard = os.path.join(actor_dir, "model_world_size_1_rank_0.pt")
    if not os.path.exists(shard):
        print(f"Missing: {shard}"); sys.exit(1)

    print(f"Loading base HF structure: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)

    print(f"Loading shard: {shard}")
    sd = torch.load(shard, map_location="cpu", weights_only=False)

    # FSDP single-GPU: tensors are stored as DTensor or regular tensor; if DTensor,
    # extract _local_tensor (which is the full tensor since world_size=1)
    clean = OrderedDict()
    for k, v in sd.items():
        t = v._local_tensor if hasattr(v, "_local_tensor") else v
        clean[k] = t

    model.load_state_dict(clean, strict=True)
    del sd, clean

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving HF model -> {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
