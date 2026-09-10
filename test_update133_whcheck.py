#!/usr/bin/env python3
"""update-13.3 tests: honest (raw Frobenius) capture + train="cores".

The external review made four claims about the t2 pipeline; this update
turns them into measurements:
  - "the whitened capture is rose-tinted glasses" -> whbank_build now computes
    the HONEST plain-Frobenius capture next to the whitened one (raw_capture
    norm-trick). This test pins the norm-trick against brute-force
    reconstruction on random data (it must be EXACT).
  - "W0 (the centroids) must stay frozen" -> train="cores": U*/V* AND
    wgud/wdnd frozen, only Cgu/Cdn train. Bit-identity checked here.
  - "add damping to the whitening" -> the damping was ALREADY there
    (inv_sqrt eps = 1e-3*lambda_max); 13.3 only makes it configurable
    (--wh-damp-frac/--wh-damp-base). inv_sqrt(eps_abs=...) pinned here.
"""
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
# the test ships INSIDE the package: prefer the code dir next to this file
# (honest extract-test from a clean unpack), fall back to the local ep13
# checkout when run from scripts/
BASE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(BASE, "hf_field_transform.py")):
    BASE = "/home/z/my-project/ep13"
sys.path.insert(0, BASE)

from hf_field_transform import FieldSparseMoe, fit_field_module  # noqa: E402
from bench_toy_speed import make_block, make_pool                # noqa: E402
from whbank_build import raw_capture, inv_sqrt                   # noqa: E402

ok, fail = [], []


def check(name, fn):
    try:
        fn()
        ok.append(name)
        print(f"PASS  {name}", flush=True)
    except Exception:
        fail.append(name)
        print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)


def _rand_case(m, n, k, seed):
    rng = np.random.default_rng(seed)
    dM = rng.standard_normal((m, n)).astype(np.float32)
    A = rng.standard_normal((m, m)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    U, _ = np.linalg.qr(A)               # orthonormal columns (m, k)
    V, _ = np.linalg.qr(B)
    U, V = U[:, :k].copy(), V[:, :k].copy()
    C = (rng.standard_normal((k, k)) * 0.3).astype(np.float32)
    return dM, U, V, C


def t_raw_capture_matches_brute_force():
    """The norm-trick must equal the explicit 1 - ||dM - UCV^T||^2/||dM||^2."""
    for (m, n, k, seed) in [(48, 32, 7, 0), (64, 96, 16, 1),
                            (128, 64, 64, 2), (32, 32, 32, 3)]:
        dM, U, V, C = _rand_case(m, n, k, seed)
        trick = raw_capture(dM, U, V, C)
        R = U @ C @ V.T
        brute = 1.0 - float(np.sum((dM - R) ** 2)) / float(np.sum(dM ** 2))
        assert abs(trick - brute) <= 1e-5 * max(1.0, abs(brute)), \
            (m, n, k, trick, brute)


def t_raw_capture_bounds():
    """C=0 -> 0; an exact-span delta reconstructs to ~1; never exceeds 1."""
    dM, U, V, C = _rand_case(40, 24, 8, 4)
    assert abs(raw_capture(dM, U, V, np.zeros_like(C))) < 1e-12
    # delta fully inside span(U) x span(V): exact reconstruction
    M0 = np.random.default_rng(5).standard_normal((8, 24)).astype(np.float32)
    dM_in = (U @ M0).astype(np.float32)          # rows in span(U)
    dM_in = (dM_in @ V @ V.T).astype(np.float32)  # also in span(V) on the right
    cap = raw_capture(dM_in, U, V, U.T @ dM_in @ V)
    assert abs(cap - 1.0) < 1e-4, cap
    dM2, U2, V2, C2 = _rand_case(48, 32, 7, 6)
    assert raw_capture(dM2, U2, V2, C2) <= 1.0 + 1e-5


def t_inv_sqrt_eps_abs():
    """eps_abs override == the manual damped formula; the default path is
    bit-identical to eps_abs = 1e-3 * lambda_max (cache compatibility)."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((16, 16)).astype(np.float64)
    C = A @ A.T + np.eye(16) * 0.1
    lam_max = float(np.linalg.eigvalsh(C)[-1])
    eps = 3e-4 * lam_max
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    manual = (V * (1.0 / np.sqrt(np.maximum(w, 0.0) + eps))) @ V.T
    got = inv_sqrt(C, eps_abs=eps)
    assert np.allclose(got, manual, rtol=1e-9, atol=1e-12)
    default = inv_sqrt(C)
    eps_default = 1e-3 * max(lam_max, 1e-12)
    w2, V2 = np.linalg.eigh((C + C.T) / 2.0)
    manual_def = (V2 * (1.0 / np.sqrt(np.maximum(w2, 0.0) + eps_default))) @ V2.T
    assert np.allclose(default, manual_def, rtol=1e-9, atol=1e-12)


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


def snap(m):
    return {n: getattr(m, n).detach().clone() for n in m.field_names}


def t_cores_freezes_centroids():
    """train='cores' (13.3): U*/V* AND the centroids wgud/wdnd stay
    bit-identical; the cores still train and the mse drops."""
    m = new_mod(32, 1)
    Y0 = pool_output(m, Xf)
    g = torch.Generator().manual_seed(31)
    Yt = Y0 + 0.001 * torch.randn(Y0.shape, generator=g)
    with torch.no_grad():     # de-optimize the cores: the polish must move
        m.Cgu.add_(0.5 * torch.randn_like(m.Cgu))
        m.Cdn.add_(0.5 * torch.randn_like(m.Cdn))
    p0 = snap(m)
    base_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    fit_field_module(m, Xf, Yt, steps=250, bs=512, lr=2e-4, device=DEV,
                     log_prefix="t4", method="adamw", seed=9,
                     autocast="off", guard=True, train="cores")
    p1 = snap(m)
    for n in ("Ugu", "Vgu", "Udn", "Vdn", "wgud", "wdnd"):
        assert torch.equal(p0[n], p1[n]), f"frozen tensor {n} moved"
    moved = [n for n in ("Cgu", "Cdn") if not torch.equal(p0[n], p1[n])]
    assert len(moved) == 2, f"cores did not train: {moved}"
    end_mse = float(F.mse_loss(pool_output(m, Xf), Yt).item())
    # note: SLOWER than train="core" (centroids frozen = less capacity) -
    # that is exactly the regime being tested; 5% gain is enough here
    assert end_mse < 0.95 * base_mse, (base_mse, end_mse)


def t_bad_train_mode_rejected_133():
    m = new_mod(16, 3)
    Y0 = pool_output(m, Xf)
    try:
        fit_field_module(m, Xf, Y0, steps=2, bs=64, lr=1e-4, device=DEV,
                         method="adam", seed=8, autocast="off", train="wat")
    except ValueError:
        return
    raise AssertionError("unknown train mode was accepted")


def t_build_block_t2_rawcap_smoke():
    """End-to-end smoke of the 13.3 build path with a patched delta stream:
    with IDENTITY whitening factors the whitened and the raw captures must
    COINCIDE (both degenerate to plain Frobenius), caps_raw lands in the
    npz, and the reuse path prints it."""
    import tempfile
    import whbank_build as wb
    from whbank_build import Whit
    rng = np.random.default_rng(11)
    geom = {"d_model": 32, "d_ff": 16, "n_exp": 4, "hidden_act": "silu"}
    d_gu = [(rng.standard_normal((32, 32)) * 0.05).astype(np.float32)
            for _ in range(4)]
    d_dn = [(rng.standard_normal((32, 16)) * 0.05).astype(np.float32)
            for _ in range(4)]
    ident = Whit(np.eye(32, dtype=np.float32), np.eye(32, dtype=np.float32),
                 np.eye(32, dtype=np.float32), np.eye(32, dtype=np.float32))
    ident16 = Whit(np.eye(16, dtype=np.float32), np.eye(16, dtype=np.float32),
                   np.eye(32, dtype=np.float32), np.eye(32, dtype=np.float32))
    covs = {"gu": ident, "dn": ident16}
    orig_iter = wb.iter_deltas
    wb.iter_deltas = lambda block, mgu, mdn: iter(list(zip(d_gu, d_dn)))
    try:
        with tempfile.TemporaryDirectory() as td:
            rows = wb.build_block_t2(0, block=None, mgu=None, mdn=None,
                                     geom=geom, covs=covs, ks=[8],
                                     dia_d=None, whdir=td, force=True)
            for kind in ("gu", "dn"):
                z = np.load(os.path.join(td, "t2_k8", "blk00_%s.npz" % kind))
                assert "caps_raw" in z.files, f"caps_raw missing ({kind})"
                caps = z["caps"].astype(np.float64)
                raw = z["caps_raw"].astype(np.float64)
                # identity whitening: the two lenses must agree
                assert np.allclose(caps, raw, rtol=2e-3, atol=2e-4), \
                    (kind, caps, raw)
                assert np.all(raw <= 1.0 + 1e-5) and np.all(raw >= -1e-3)
            # verify path on the same files (expert 0)
            good = wb.verify_block(0, block=None, mgu=None, mdn=None,
                                   geom=geom, covs=covs, whdir=td,
                                   t2_ks=[8], dia_d=None, r=None)
            assert good, "verify_block failed on freshly built banks"
    finally:
        wb.iter_deltas = orig_iter


for name, fn in [("raw_capture == brute force (norm-trick exact)",
                  t_raw_capture_matches_brute_force),
                 ("raw_capture bounds (C=0, exact span, <=1)",
                  t_raw_capture_bounds),
                 ("inv_sqrt eps_abs override + default cache compat",
                  t_inv_sqrt_eps_abs),
                 ("train=cores freezes U*/V* AND centroids, cores train",
                  t_cores_freezes_centroids),
                 ("unknown train mode rejected (all|core|cores)",
                  t_bad_train_mode_rejected_133),
                 ("build_block_t2 smoke: raw==whitened @ identity whitening",
                  t_build_block_t2_rawcap_smoke)]:
    check(name, fn)

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
