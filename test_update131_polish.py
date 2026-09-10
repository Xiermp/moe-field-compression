#!/usr/bin/env python3
"""update-13.1 tests: the best-state OFF-BY-ONE fix + the train="core" polish.

Reproduces the 2026-09-08 explosion in miniature: the module starts from a
"bank init" that is already optimal at step 0 (the pool target IS the
module's own output + tiny noise). OLD behavior: muon@2e-3 kicks the
converged point, the divergence bail restored "best" == init + one kick
(the snapshot was taken AFTER opt.step() while its score was the PRE-step
loss), and the block shipped an mse ~1000x the step-0 value -> artifact
KL 2.288. NEW behavior: the decision block runs BEFORE the step, the bail
restores the PRISTINE init, and train="core" freezes the basis entirely.
"""
import os
import sys
import traceback

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
BASE = "/home/z/my-project/ep13"
sys.path.insert(0, BASE)

from hf_field_transform import FieldSparseMoe, fit_field_module  # noqa: E402
from bench_toy_speed import make_block, make_pool                # noqa: E402

ok, fail = [], []


def check(name, fn):
    try:
        fn()
        ok.append(name)
        print(f"PASS  {name}", flush=True)
    except Exception:
        fail.append(name)
        print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)


blk = make_block(d=256, dff=192, n_exp=16, top_k=4, seed=100, r_true=12,
                 shared_frac=0.7)
X, Y = make_pool(blk, 8192, seed=7, d=256)
Xf = X[:6144]

DEV = "cpu"


def new_mod(rank=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = FieldSparseMoe(blk["geom"], rank, gate_w=blk["gw"], act_fn=F.silu,
                       dtype=torch.float32,
                       init={"Ugu": torch.randn(2 * blk["geom"]["d_ff"], rank,
                                                generator=g) * 0.02,
                             "Vgu": torch.randn(blk["geom"]["d_model"], rank,
                                                generator=g) * 0.02,
                             "Udn": torch.randn(blk["geom"]["d_model"], rank,
                                                generator=g) * 0.02,
                             "Vdn": torch.randn(blk["geom"]["d_ff"], rank,
                                                generator=g) * 0.02,
                             "Cgu": torch.randn(blk["geom"]["n_exp"], rank,
                                                generator=g) * 0.02,
                             "Cdn": torch.randn(blk["geom"]["n_exp"], rank,
                                                generator=g) * 0.02})
    with torch.no_grad():
        m.wgud.copy_(blk["m_gu"])
        m.wdnd.copy_(blk["m_dn"])
    return m


def pool_output(m, X):
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 1024):
            xb = X[i:i + 1024].view(1, -1, X.shape[-1])
            outs.append(m.forward(xb).reshape(-1, xb.shape[-1]))
    return torch.cat(outs)


def make_pristine_optimal(rank=32, seed=0, noise=0.01):
    """A module whose CURRENT params are the global optimum of the pool
    mse (the 'bank init' scenario), plus a small noise floor so the best
    score is > 0 (the 2x divergence bail needs that)."""
    m = new_mod(rank, seed)
    Y0 = pool_output(m, Xf)
    g = torch.Generator().manual_seed(31)
    Yt = Y0 + noise * torch.randn(Y0.shape, generator=g)
    return m, Yt


def snap(m):
    return {n: getattr(m, n).detach().clone() for n in m.field_names}


def same(a, b):
    return all(torch.equal(a[n], b[n]) for n in a)


def t_divergence_bail_ships_pristine():
    """THE regression: a converged init + muon@2e-3 must ship the PRISTINE
    init (old code shipped init + one NS kick: re-eval ~1000x step 0)."""
    m, Yt = make_pristine_optimal(32, 0)
    p0 = snap(m)
    first_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    last = fit_field_module(m, Xf, Yt, steps=120, bs=512, lr=2e-3, device=DEV,
                            log_prefix="t1", method="muon", seed=5,
                            autocast="off", guard=True)
    shipped_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    assert same(snap(m), p0), "shipped params are NOT the pristine init"
    assert shipped_mse <= max(4.0 * first_mse, 1e-6), \
        (first_mse, shipped_mse, last)
    # the returned mse describes the shipped weights (re-eval of the restore)
    assert abs(last - shipped_mse) <= 0.5 * max(shipped_mse, 1e-9) + 1e-6, \
        (last, shipped_mse)


def t_core_freezes_basis():
    """train='core': the U*/V* basis stays bit-identical, the fit cannot
    explode through it, and the pool mse does not get worse."""
    m, Yt = make_pristine_optimal(32, 1)
    p0 = snap(m)
    base_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    last = fit_field_module(m, Xf, Yt, steps=60, bs=512, lr=5e-5, device=DEV,
                            log_prefix="t2", method="adamw", seed=6,
                            autocast="off", guard=True, train="core")
    p1 = snap(m)
    for n in ("Ugu", "Vgu", "Udn", "Vdn"):
        assert torch.equal(p0[n], p1[n]), f"basis factor {n} moved"
    end_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    assert end_mse <= max(1.5 * base_mse, 1e-6), (base_mse, end_mse, last)


def t_core_still_learns_cores():
    """train='core' is not a no-op: the cores/centroids receive updates when
    the optimum is NOT exactly at the init (offset cores, tiny noise floor
    so the core error dominates it)."""
    m, Yt = make_pristine_optimal(32, 2, noise=0.001)
    with torch.no_grad():     # de-optimize the cores: the polish must move
        m.Cgu.add_(0.5 * torch.randn_like(m.Cgu))
        m.Cdn.add_(0.5 * torch.randn_like(m.Cdn))
    p0 = snap(m)
    base_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    fit_field_module(m, Xf, Yt, steps=150, bs=512, lr=2e-4, device=DEV,
                     log_prefix="t3", method="adamw", seed=7,
                     autocast="off", guard=True, train="core")
    p1 = snap(m)
    moved = [n for n in ("Cgu", "Cdn") if not torch.equal(p0[n], p1[n])]
    assert len(moved) == 2, f"cores did not train: {moved}"
    for n in ("Ugu", "Vgu", "Udn", "Vdn"):
        assert torch.equal(p0[n], p1[n]), f"basis factor {n} moved"
    end_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    assert end_mse < 0.8 * base_mse, (base_mse, end_mse)


def t_bad_train_mode_rejected():
    m, Yt = make_pristine_optimal(16, 3)
    try:
        fit_field_module(m, Xf, Yt, steps=2, bs=64, lr=1e-4, device=DEV,
                         method="adam", seed=8, autocast="off", train="wat")
    except ValueError:
        return
    raise AssertionError("unknown train mode was accepted")


for name, fn in [("divergence bail ships the PRISTINE init (KL-2.288 fix)",
                  t_divergence_bail_ships_pristine),
                 ("train=core freezes the U*/V* basis",
                  t_core_freezes_basis),
                 ("train=core still trains the cores",
                  t_core_still_learns_cores),
                 ("unknown train mode rejected", t_bad_train_mode_rejected)]:
    check(name, fn)

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
