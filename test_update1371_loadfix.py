#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_update1371_loadfix.py - tests for ROBUST-HF-LOAD (13.7.1).

What is pinned down:
  t1  load_hf_model_robust: a tiny native-arch HF model round-trips through
      save_pretrained -> from_config + state-dict shards (strict, no
      missing/unexpected), weights bit-equal to the from_pretrained load;
  t2  the dtype path: dtype=float32 is honored by the robust load;
  t3  _bad_nonpersistent_buffers detects a NaN non-persistent buffer and
      ignores healthy/persistent ones (the 13.7.1 gate);
  t4  the CLI dense default (13.7.1 companion change) - parser default
      bank_init == "dense".

Run: /path/to/python test_update1371_loadfix.py
"""
import os
import sys
import tempfile

EP13 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EP13)

import torch  # noqa: E402

import hf_cli  # noqa: E402
from hf_field_transform import (  # noqa: E402
    _bad_nonpersistent_buffers, load_hf_model_robust)

FAILED = []


def ok(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)


def main():
    from transformers import AutoModelForCausalLM
    from transformers.models.opt.configuration_opt import OPTConfig

    # t1: robust load round-trip on a tiny native model -------------------
    cfg = OPTConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, ffn_dim=32,
                    max_position_embeddings=32)
    tiny = AutoModelForCausalLM.from_config(cfg)
    tiny.eval()
    with tempfile.TemporaryDirectory() as td:
        tiny.save_pretrained(td, safe_serialization=True)
        robust = load_hf_model_robust(td, dtype=torch.float32,
                                      trust_remote_code=False)
        fp = AutoModelForCausalLM.from_pretrained(td, dtype=torch.float32)
        w_rob = next(robust.parameters())
        w_fp = next(fp.parameters())
        ok(torch.equal(w_rob, w_fp),
           "t1 robust load: weights bit-equal to from_pretrained")
        ok(not _bad_nonpersistent_buffers(robust),
           "t1 robust load: no bad non-persistent buffers")

        # t2: dtype honored ------------------------------------------------
        ok(str(next(robust.parameters()).dtype) == "torch.float32",
           "t2 dtype=float32 honored by the robust load")

    # t3: the gate ----------------------------------------------------------
    probe = AutoModelForCausalLM.from_config(cfg)
    ok(not _bad_nonpersistent_buffers(probe),
       "t3 gate: healthy model passes")
    # register a NaN NON-persistent buffer on the embedding module
    emb = probe.get_input_embeddings()
    emb.register_buffer("nan_buf", torch.tensor([float("nan"), 1.0]),
                        persistent=False)
    emb.register_buffer("ok_buf", torch.zeros(2), persistent=False)
    emb.register_buffer("nan_persistent", torch.tensor([float("nan")]))
    bad = _bad_nonpersistent_buffers(probe)
    ok(any(b.endswith(".nan_buf") for b in bad)
       and not any("ok_buf" in b for b in bad)
       and not any("nan_persistent" in b for b in bad),
       f"t3 gate: NaN non-persistent detected, persistent ignored ({bad})")

    # t4: the companion CLI default ----------------------------------------
    ap = hf_cli.build_parser("x")
    ok(ap.parse_args([]).bank_init == "dense",
       "t4 CLI: --bank-init default is dense (13.7.1)")

    print()
    if FAILED:
        print("FAILED:", len(FAILED))
        for m in FAILED:
            print("  -", m)
        return 1
    print("ALL PASSED (4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
