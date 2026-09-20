#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_update137_dense.py - tests for DENSE-INIT (13.7, --bank-init dense).

What is pinned down:
  t1  CLI: --bank-init accepts "dense" (choices svd|t2|dense, default dense
      since 13.7.1); canonical_command emits the explicit non-default flag;
      canonical_command round-trips the value
  t2  svd_init_ver helper: the four stamp combinations; the default stays
      "joint-v2" (bit-identical cache namespace since 13.6.x)
  t3  functional expert_basis_init on a synthetic structured block:
      (a) dense mode returns C (E, r, r), diag returns (E, r)
      (b) SAME bases: U/V bitwise equal between the modes (the core choice
          must not perturb the basis machinery)
      (c) diag C == diagonal of the dense C (the legacy values, unchanged)
      (d) dense recon error <= diag recon error (the diagonal is a subset
          of the full core in the same basis)
      (e) stamps: svd_ver "joint-v2-dense" / "joint-v2"; dense capture ==
          capture_proj; diag capture < capture_proj (structured deltas)
  t4  FieldSparseMoe transfer + forward: apply_core(geom, True) + dense init
      -> Cgu/Cdn are (E, r, r), forward is finite; banks=2 + dense -> C2*
      stay DIAGONAL (E, r) (the container rule) and U2/V2 transfer
  t5  hf_pipeline.py source anchors: svd_ver_want via svd_init_ver, the
      fit_r<rank>dense dir, the core= argument in _save_svd_init, the three
      apply_core gates on ("t2", "dense"), the artifact/accounting cores,
      bank_init in the field_meta profile; no stale "== \"t2\"" gates left
      outside the two legitimate spots (polish profile + t2 svd_ver)
  t6  --list-flags (subprocess, torch-free) shows the dense choice

Run: /path/to/python test_update137_dense.py
"""
import math
import os
import subprocess
import sys

EP13 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EP13)

import torch  # noqa: E402

import hf_cli  # noqa: E402
from hf_field_transform import (FieldSparseMoe, apply_core,  # noqa: E402
                                expert_basis_init, svd_init_ver,
                                SVD_INIT_VER)
from test_update1363_rangefix import FakeBlock, make_block  # noqa: E402

torch.set_num_threads(2)
FAILED = []


def ok(cond, name):
    print(("PASS  " if cond else "FAIL  ") + name, flush=True)
    if not cond:
        FAILED.append(name)


def main():
    rank = 8

    # t1: CLI -------------------------------------------------------------
    ap = hf_cli.build_parser("x")
    act = [a for a in ap._actions if "--bank-init" in a.option_strings]
    ok(len(act) == 1 and list(act[0].choices) == ["svd", "t2", "dense"]
       and act[0].default == "dense",
       "t1 parser: --bank-init choices svd|t2|dense, default dense (13.7.1)")
    ok(ap.parse_args([]).bank_init == "dense",
       "t1 default: empty parse -> bank_init=dense (zero-shot first)")
    a = ap.parse_args(["--bank-init", "svd"])
    cmd = hf_cli.canonical_command(a)
    ok("--bank-init svd" in cmd,
       "t1 canonical: emits explicit --bank-init svd (%r)" % cmd)
    ok(ap.parse_args(cmd.split()).bank_init == "svd",
       "t1 canonical: round-trip keeps bank_init=svd")

    # t2: stamp helper -----------------------------------------------------
    ok(svd_init_ver() == "joint-v2" == SVD_INIT_VER,
       "t2 svd_init_ver default is joint-v2 (13.6.x namespace unchanged)")
    ok(svd_init_ver("diag", 2) == "joint-v2+b2",
       "t2 svd_init_ver(diag, banks=2) = joint-v2+b2")
    ok(svd_init_ver("dense", 1) == "joint-v2-dense"
       and svd_init_ver("dense", 2) == "joint-v2-dense+b2",
       "t2 svd_init_ver dense stamps (-dense[+b2])")

    # t3: functional init --------------------------------------------------
    Wgu, Wdn = make_block(d=96, dff=48, n_exp=16, seed=100, r_true=10)
    blk = FakeBlock(Wgu, Wdn)
    mgu, mdn = Wgu.mean(0), Wdn.mean(0)
    ini_d = expert_basis_init(blk, mgu.clone(), mdn.clone(), rank,
                              log_prefix="t3-diag")
    ini_s = expert_basis_init(blk, mgu.clone(), mdn.clone(), rank,
                              log_prefix="t3-dense", core="dense")
    ok(ini_d["Cgu"].shape == (16, rank) and ini_d["Cdn"].shape == (16, rank)
       and ini_s["Cgu"].shape == (16, rank, rank)
       and ini_s["Cdn"].shape == (16, rank, rank),
       "t3a core shapes: diag (E,r), dense (E,r,r)")
    ok(torch.equal(ini_d["Ugu"], ini_s["Ugu"])
       and torch.equal(ini_d["Vgu"], ini_s["Vgu"])
       and torch.equal(ini_d["Udn"], ini_s["Udn"])
       and torch.equal(ini_d["Vdn"], ini_s["Vdn"]),
       "t3b SAME bases: U/V bitwise equal between the modes")
    ok(torch.equal(ini_d["Cgu"],
                   torch.diagonal(ini_s["Cgu"], dim1=-2, dim2=-1))
       and torch.equal(ini_d["Cdn"],
                       torch.diagonal(ini_s["Cdn"], dim1=-2, dim2=-1)),
       "t3c diag C == diagonal of the dense C (legacy values unchanged)")

    def recon_err(ini, dense, side):
        W = Wgu if side == "gu" else Wdn
        m = mgu if side == "gu" else mdn
        U, V, C = ini[f"U{side}"], ini[f"V{side}"], ini[f"C{side}"]
        num = 0.0
        for e in range(W.shape[0]):
            rec = (U @ C[e]) @ V.t() if dense else (U * C[e]) @ V.t()
            num += float(((rec - (W[e] - m)) ** 2).sum())
        return num / max(float(((W - m.unsqueeze(0)) ** 2).sum()), 1e-12)

    ok(all(recon_err(ini_s, True, s) <= recon_err(ini_d, False, s) + 1e-9
           for s in ("gu", "dn")),
       "t3d dense recon <= diag recon on the same bases (gu %.4f vs %.4f, "
       "dn %.4f vs %.4f)"
       % (recon_err(ini_s, True, "gu"), recon_err(ini_d, False, "gu"),
          recon_err(ini_s, True, "dn"), recon_err(ini_d, False, "dn")))
    ok(ini_s["svd_ver"] == "joint-v2-dense" and ini_d["svd_ver"] == "joint-v2"
       and ini_s["capture_metric"] == "full core (13.7)"
       and ini_d["capture_metric"] == "diag coords",
       "t3e stamps: svd_ver + capture_metric per mode")
    ok(all(abs(ini_s["capture"][s] - ini_s[f"capture_proj_{s}"]) < 1e-12
           for s in ("gu", "dn"))
       and all(ini_d["capture"][s] < ini_d[f"capture_proj_{s}"]
               for s in ("gu", "dn")),
       "t3f dense capture == capture_proj; diag capture < capture_proj")

    # t4: container transfer + forward -------------------------------------
    geom = dict(n_exp=16, d_model=96, d_ff=48, top_k=4, norm_topk=True,
                field_mode="postact", u_mode="none",
                router_kind="softmax")
    mod = FieldSparseMoe(apply_core(geom, True), rank,
                         gate_w=torch.randn(16, 96), init=dict(ini_s),
                         dtype=torch.float32)
    ok(tuple(mod.Cgu.shape) == (16, rank, rank)
       and tuple(mod.Cdn.shape) == (16, rank, rank)
       and float(mod.Cgu.detach().norm()) > 0,
       "t4a dense module holds Cgu/Cdn (E,r,r) from the init")
    x = torch.randn(2, 16, 96)
    with torch.no_grad():
        y = mod(x)
    ok(torch.isfinite(y).all() and y.shape == (2, 16, 96),
       "t4b dense-core forward is finite")
    ini_b2 = expert_basis_init(blk, mgu.clone(), mdn.clone(), rank,
                               log_prefix="t4-b2", banks=2, core="dense")
    b2_init = dict(ini_s)
    for k in ini_b2:
        if k.startswith(("U2", "V2", "C2")):
            b2_init[k] = ini_b2[k]
    mod2 = FieldSparseMoe(apply_core(geom, True), rank,
                          gate_w=torch.randn(16, 96), banks=2,
                          init=b2_init, dtype=torch.float32)
    ok(tuple(mod2.C2gu.shape) == (16, rank) and tuple(mod2.C2dn.shape) == (16, rank)
       and float(mod2.U2gu.detach().norm()) > 0
       and float(mod2.C2gu.detach().norm()) > 0,
       "t4c banks=2 + dense: bank 2 stays DIAGONAL (E,r); C2 starts at the "
       "residual capture (UPDATE-12), U2/V2 transferred")
    with torch.no_grad():
        y2 = mod2(x)
    ok(torch.isfinite(y2).all(), "t4d banks=2 dense forward is finite")

    # t5: pipeline source anchors ------------------------------------------
    src = open(os.path.join(EP13, "hf_pipeline.py"), encoding="utf-8").read()
    ok("svd_ver_want = svd_init_ver(\"diag\", args.banks)" in src
       and 'elif args.bank_init == "dense":' in src
       and 'svd_ver_want = svd_init_ver("dense", args.banks)' in src,
       "t5a svd_ver_want derived through svd_init_ver (+dense elif)")
    ok('{"t2": "t2", "dense": "dense"}.get(args.bank_init, "")' in src,
       "t5b fit dir suffix covers dense (fit_r<rank>dense)")
    ok('core=("dense" if args.bank_init == "dense"' in src
       and 'else "diag"))' in src,
       "t5c _save_svd_init passes core= to expert_basis_init")
    n_gates = src.count('args.bank_init in ("t2", "dense")')
    ok(n_gates >= 5,
       f"t5d apply_core/artifact/accounting gates on (t2, dense): {n_gates} >= 5")
    ok("bank_init=args.bank_init" in src,
       "t5e field_meta profile stamps bank_init")
    stale = [ln for ln in src.splitlines()
             if 'args.bank_init == "t2"' in ln]
    ok(len(stale) == 3,
       f"t5f only the 3 legitimate '== t2' gates remain: polish profile, "
       f"t2 svd_ver, _save_svd_init t2-branch (got {len(stale)})")

    # t6: --list-flags (torch-free subprocess) ------------------------------
    r = subprocess.run([sys.executable, os.path.join(EP13, "hf_pipeline.py"),
                        "--list-flags"], capture_output=True, text=True)
    ok(r.returncode == 0 and "--bank-init" in r.stdout
       and "dense" in r.stdout,
       "t6 --list-flags shows the dense init choice")

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)}")
        for f in FAILED:
            print("  - " + f)
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
