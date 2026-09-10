#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_update136_cli.py - tests for CLI-CLARITY (13.6, hf_cli.py).

What is pinned down:
  t1  FLAG PARITY: the grouped parser carries EXACTLY the 76 flags of the old
      flat argparse block (same names) + the 2 new info flags
  t2  DEFAULT/CHOICES parity for every flag whose default is not None
      (spot-hardcoded from the pre-refactor parser)
  t3  parse of a real command: dests land where the pipeline expects them;
      bad choices still SystemExit
  t4  file_facts: sha256 matches a direct hashlib pass; VER constants
      scraped without importing (SVD_INIT_VER / FIELD_ENGINE_VER /
      WHBANK_T2_VER); print_manifest/print_version/print_stage_table run
  t5  canonical_command: bare defaults -> empty; mutated args -> the exact
      flags; re-parsing the emitted command reproduces the values
  t6  STAGE metadata: 9 stages, unique ids, table prints
  t7  subprocess: --version / --list-flags / --list-stages / --help exit 0
      on a python WITHOUT torch (info commands must not bootstrap)

Run: /path/to/python test_update136_cli.py
"""
import os
import subprocess
import sys

EP13 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EP13)

import hashlib  # noqa: E402
import hf_cli   # noqa: E402

PASS = []


def ok(cond, msg):
    PASS.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + msg)


# the 76 flags of the pre-refactor flat parser, verbatim
OLD_FLAGS = set("""--model --gguf-quant --gguf-file --gguf --gguf-out
--gguf-base-repo --local-path --auto --rank --banks --bank-init --field-mode
--u-mode --u-rank --out --device --dtype --profile --calib-file --eval-file
--calib-dataset --text-cap --calib-windows --calib-bsz --calib-ctx
--per-layer-cap --pool-recalibrate --fit-steps --fit-bs --fit-lr --fit-method
--fit-autocast --muon-max-dim --muon-ns-steps --fit-train --fit-init
--refresh-init --refresh-refine --fit-jitter --fit-preset --fit-workers
--fit-early-stop --fit-guard-warmup --strict-fit-guard --fit-lr-warmup
--fit-router --router-steps --router-lr --router-anchor --refine-rounds
--io-threads --prefetch --io-cache --verify-topk --iron-prompt --eval-chunks
--kl-chunks --eval-ctx --gen-tokens --gen-rep-pen --max-shard --save-backbone
--threads --low-mem --cleanup --full-dequant --keep-dequant --skip-fit-guard
--skip-reload-check --test-only --force-topk --stages --skip --list-stages
--no-cache-verify --smoke""".split())

# dest -> expected default (non-None defaults of the old parser)
OLD_DEFAULTS = {
    "model": "mradermacher/OLMoE-1B-7B-0924-GGUF", "gguf_quant": "Q4_K_M",
    "rank": 32, "banks": 1, "bank_init": "svd", "field_mode": "preact",
    "u_mode": "none", "u_rank": 4, "device": "auto", "dtype": "auto",
    "profile": "auto", "text_cap": 3_000_000, "calib_windows": 3,
    "calib_ctx": 512, "per_layer_cap": 8192, "fit_init": "svd",
    "fit_jitter": 0.0, "fit_workers": 1, "fit_early_stop": 0,
    "fit_guard_warmup": -1, "fit_lr_warmup": 0, "fit_router": "off",
    "router_steps": 80, "router_anchor": 0.03, "refine_rounds": 0,
    "prefetch": 1, "eval_chunks": 50, "kl_chunks": 16, "eval_ctx": 512,
    "gen_tokens": 48, "gen_rep_pen": 1.15, "max_shard": "4GB",
    "save_backbone": "keep", "force_topk": 0, "verify_topk": "",
    "iron_prompt": "", "muon_max_dim": 512, "muon_ns_steps": 5,
    "fit_autocast": "auto",
}
OLD_CHOICES = {
    "banks": (1, 2), "bank_init": ["svd", "t2"],
    "field_mode": ["preact", "postact"],
    "u_mode": ["none", "rank1", "rank4", "full"],
    "device": ["auto", "cuda", "cpu"],
    "dtype": ["auto", "bfloat16", "float16", "float32"],
    "profile": ["auto", "low", "high"], "fit_autocast": ["auto", "on", "off"],
    "fit_train": ["all", "core", "cores"], "fit_init": ["svd", "random"],
    "fit_router": ["off", "after", "joint"], "io_cache": ["disk", "ram"],
    "save_backbone": ["keep", "bf16"],
}


def main():
    ap = hf_cli.build_parser("x")

    # t1: flag parity --------------------------------------------------------
    got = {a.option_strings[0] for a in ap._actions if a.option_strings} \
          - {"-h", "--help"}
    new_flags = {"--list-flags", "--version"}
    ok(got == OLD_FLAGS | new_flags,
       "t1 flag parity: %d flags, old 76 all present, extras = %s"
       % (len(got), sorted(got - OLD_FLAGS)))

    # t2: defaults + choices parity ------------------------------------------
    d = {a.dest: a.default for a in ap._actions if hasattr(a, "dest")}
    bad = {k: (d.get(k), v) for k, v in OLD_DEFAULTS.items() if d.get(k) != v}
    ok(not bad, "t2 defaults parity (%d checked)%s"
       % (len(OLD_DEFAULTS), "; BAD: %s" % bad if bad else ""))
    c = {a.dest: a.choices for a in ap._actions if hasattr(a, "dest")}
    badc = {k: (c.get(k), v) for k, v in OLD_CHOICES.items() if c.get(k) != v}
    ok(not badc, "t2b choices parity (%d checked)%s"
       % (len(OLD_CHOICES), "; BAD: %s" % badc if badc else ""))

    # t3: parse behavior ------------------------------------------------------
    a = ap.parse_args(["--rank", "64", "--u-mode", "rank4", "--u-rank", "2",
                       "--stages", "fit,save,verify", "--verify-topk", "1,4",
                       "--fit-method", "muon", "--banks", "2", "--auto"])
    ok(a.rank == 64 and a.u_mode == "rank4" and a.u_rank == 2
       and a.stages == "fit,save,verify" and a.banks == 2 and a.auto
       and a.fit_train is None and a.out is None,
       "t3 parse: dests match the old layout")
    for bad_argv in (["--rank", "x"], ["--u-mode", "rank8"], ["--banks", "3"]):
        try:
            ap.parse_args(bad_argv)
            ok(False, "t3b bad argv %r did not fail" % bad_argv)
        except SystemExit:
            ok(True, "t3b bad argv %r -> SystemExit" % bad_argv)

    # t4: fingerprints without importing --------------------------------------
    f = hf_cli.file_facts(EP13, "hf_field_transform.py")
    import io
    raw = io.open(os.path.join(EP13, "hf_field_transform.py"), "rb").read()
    ok(f["sha"] == hashlib.sha256(raw).hexdigest()[:12],
       "t4 sha256 matches a direct hashlib pass")
    vers = dict(kv.split("=", 1) for kv in f["vers"])
    ok(vers.get("SVD_INIT_VER") == "joint-v1"
       and vers.get("FIELD_ENGINE_VER") == "update-13-whrank",
       "t4b VER constants scraped from source: %s" % vers)
    wb = dict(kv.split("=", 1)
              for kv in hf_cli.file_facts(EP13, "whbank_build.py")["vers"])
    ok(wb.get("WHBANK_T2_VER") == "whrank-t2-v1", "t4c WHBANK_T2_VER found")
    ok(all(hf_cli.file_facts(EP13, c["file"]) is not None
           for c in hf_cli.COMPONENTS if c["core"]),
       "t4d every core component file exists")
    ok(isinstance(hf_cli.shadow_copies(EP13), list),
       "t4e shadow-copy scan runs")
    hf_cli.print_manifest(EP13)
    hf_cli.print_version(EP13)
    print("PASS  t4f print_manifest/print_version ran (see banner above)")

    # t5: canonical command ---------------------------------------------------
    b = hf_cli.build_parser("x")
    bare = b.parse_args([])
    ok(hf_cli.canonical_command(bare) == "",
       "t5 bare defaults -> empty canonical command")
    a.verify_topk = [1, 4]           # post-parse mutation, as main() does
    a.fit_steps = 300                # preset/legacy collapse
    a.u_mode, a.auto = "rank4", False
    cmd = hf_cli.canonical_command(a)
    ok("--fit-steps 300" in cmd and "--u-mode rank4" in cmd
       and "--verify-topk 1,4" in cmd and "--rank 64" in cmd
       and "--iron-prompt" not in cmd and "--out" not in cmd
       and "--fit-train" not in cmd,
       "t5b canonical shows effective deltas only: %s" % cmd)
    re_parsed = b.parse_args(cmd.split())
    ok(re_parsed.rank == 64 and re_parsed.u_mode == "rank4"
       and re_parsed.fit_steps == 300 and re_parsed.verify_topk == "1,4",
       "t5c canonical command round-trips through the parser")

    # t6: stages ---------------------------------------------------------------
    ok(hf_cli.STAGE_ORDER == ["download", "texts", "base", "calibrate", "fit",
                              "refine", "save", "verify", "report"]
       and len(set(hf_cli.STAGE_ORDER)) == 9,
       "t6 stage order intact (9 unique ids)")
    ok(set(hf_cli.STAGE_DESCR) == set(hf_cli.STAGE_ORDER)
       and all(hf_cli.STAGE_DESCR[s] for s in hf_cli.STAGE_ORDER),
       "t6b STAGE_DESCR covers every stage")
    hf_cli.print_stage_table()
    print("PASS  t6c stage table ran (see table above)")

    # t7: info commands on a torch-free python --------------------------------
    for flag, needle in [("--version", "RUNTIME COMPONENTS"),
                         ("--list-flags", "EVERY FLAG"),
                         ("--list-stages", "Pipeline stages"),
                         ("--help", "required - model source")]:
        p = subprocess.run([sys.executable, os.path.join(EP13, "hf_pipeline.py"),
                            flag], capture_output=True, text=True, timeout=120)
        ok(p.returncode == 0 and needle in p.stdout,
           "t7 %s exits 0, contains %r%s" % (flag, needle,
                                             "" if p.returncode == 0
                                             else " OUT: " + p.stdout[-400:]))

    print("\n%d/%d passed" % (sum(PASS), len(PASS)))
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
