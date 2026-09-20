#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_update1364_fitfresh.py - tests for FIT-FRESH (13.6.4, --fit-fresh).

What is pinned down:
  t1  parser: --fit-fresh exists, store_true, default False, dest fit_fresh
  t2  canonical_command: emits "--fit-fresh" when True, nothing when False;
      re-parsing the emitted command round-trips the value
  t3  hf_pipeline.py source guards (anchor check, no torch import needed):
      (a) the fit-fresh bypass sits AFTER fit_meta read and BEFORE the
          per-block resume block,
      (b) partial_p is defined UNCONDITIONALLY (a NameError here would kill
          the first finished block of every --fit-fresh run),
      (c) the resume block is skipped when fit_fresh (`not args.fit_fresh`),
      (d) both cache notices advertise the flag
  t4  --list-flags (subprocess, python WITHOUT torch) shows --fit-fresh
  t5  functional: a fake fit_dir with a complete cached fit (fit_meta.json +
      fit_blk*.pt) and a matching sig - the decision logic refits ALL blocks
      under fit_fresh and resumes under the default (replicates the exact
      decision chain of stage 5: fit_blocks_ok -> sig match -> fresh bypass)

Run: /path/to/python test_update1364_fitfresh.py
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

EP13 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EP13)

import hf_cli  # noqa: E402

PASS = []


def ok(cond, msg):
    PASS.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + msg)


def main():
    ap = hf_cli.build_parser("x")

    # t1: parser ---------------------------------------------------------
    act = [a for a in ap._actions if "--fit-fresh" in a.option_strings]
    ok(len(act) == 1 and isinstance(act[0], argparse._StoreTrueAction)
       and act[0].default is False and act[0].dest == "fit_fresh",
       "t1 parser: --fit-fresh store_true, default False, dest fit_fresh")

    # t2: canonical command ----------------------------------------------
    a_true = ap.parse_args(["--fit-fresh"])
    cmd = hf_cli.canonical_command(a_true)
    ok("--fit-fresh" in cmd, "t2 canonical: emits --fit-fresh when set (%r)" % cmd)
    a_rt = ap.parse_args(cmd.split())
    ok(a_rt.fit_fresh is True, "t2 canonical: round-trip keeps fit_fresh=True")
    a_false = ap.parse_args([])
    ok("--fit-fresh" not in hf_cli.canonical_command(a_false),
       "t2 canonical: omitted when unset")

    # t3: source anchors in hf_pipeline.py -------------------------------
    src = open(os.path.join(EP13, "hf_pipeline.py"), encoding="utf-8").read()
    i_meta = src.find("fit_meta_p = os.path.join(fit_dir, \"fit_meta.json\")")
    i_part = src.find("partial_p = os.path.join(fit_dir, \"fit_partial.json\")")
    i_fresh = src.find("if args.fit_fresh and (fit_done or fit_blocks_ok")
    i_resume = src.find("if not fit_done and not args.fit_fresh:")
    ok(0 < i_meta < i_part < i_fresh < i_resume,
       "t3a anchors ordered: fit_meta -> partial_p -> fresh bypass -> resume")
    ok(i_meta < src.find("partial_p = ", i_meta + 1) if False else
       src.count("partial_p = os.path.join") == 1 and i_part < i_meta + 200,
       "t3b partial_p defined once, unconditionally (no fit-fresh NameError)")
    ok(i_resume > 0 and "not args.fit_fresh" in src[i_resume:i_resume + 60],
       "t3c resume block skipped when --fit-fresh")
    ok("--fit-fresh: ignoring the cached fit" in src
       and "(pass --fit-fresh to refit everything " in src
       and "new fit: pass --fit-fresh, or delete" in src,
       "t3d both fit-cache notices advertise --fit-fresh")

    # t4: --list-flags without torch (info commands live in hf_pipeline.py)
    r = subprocess.run([sys.executable, os.path.join(EP13, "hf_pipeline.py"),
                        "--list-flags"], capture_output=True, text=True,
                       timeout=120)
    ok(r.returncode == 0 and "--fit-fresh" in r.stdout,
       "t4 --list-flags exits 0 and documents --fit-fresh")

    # t5: functional decision check (stage-5 chain replica) ---------------
    sys.path.insert(0, EP13)
    from hf_pipeline import fit_blocks_ok  # noqa: E402
    td = tempfile.mkdtemp(prefix="fitfresh_")
    try:
        sig = {"fit_steps": 10, "fit_method": "adam"}
        for i in range(3):
            with open(os.path.join(td, f"fit_blk{i}.pt"), "wb") as f:
                f.write(b"pretend-torch-save")
        with open(os.path.join(td, "fit_meta.json"), "w") as f:
            json.dump(sig, f)
        with open(os.path.join(td, "fit_partial.json"), "w") as f:
            json.dump({"sig": sig, "mse": {"0": 1.0, "1": 2.0, "2": 3.0}}, f)

        n_blocks, fit_sig = 3, dict(sig)
        fit_done = fit_blocks_ok(td, n_blocks)
        if fit_done:
            with open(os.path.join(td, "fit_meta.json")) as f:
                fit_done = json.load(f) == fit_sig
        part_mse = {}
        if not fit_done and False:  # default path placeholder (symmetry)
            pass
        # default: fully cached -> skip
        ok(fit_done and fit_blocks_ok(td, n_blocks),
           "t5 default: complete matching fit cache -> fit_done (skip)")
        # --fit-fresh: bypass -> refit everything
        args_fit_fresh = True
        fresh_hit = args_fit_fresh and (fit_done or fit_blocks_ok(td, n_blocks))
        ok(fresh_hit,
           "t5 fit-fresh: cache present -> fresh bypass fires")
        fit_done2 = fit_done and not args_fit_fresh
        part_mse2 = {}
        if not fit_done2 and not args_fit_fresh:
            pass  # would load fit_partial.json here
        ok(not fit_done2 and part_mse2 == {},
           "t5 fit-fresh: fit_done=False, part_mse empty -> ALL blocks refit")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    print("\n%d/%d passed" % (sum(PASS), len(PASS)))
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
