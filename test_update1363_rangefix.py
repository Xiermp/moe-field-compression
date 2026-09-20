#!/usr/bin/env python3
"""13.6.3 regression: the pass-1 range finder must be the SECOND moment.

The joint-v1 bug: Y = sum_e dW_e @ Om == (sum_e dW_e) @ Om == 0 at the
exact-mean center (the pipeline ALWAYS centers at the arithmetic mean), so
Q = orth(fp32 noise) and the whole SVD init lived on a garbage subspace -
diag capture ~0.1 instead of ~0.5 on synthetic blocks, step-0 mse at the
centroid line. joint-v2 accumulates Y = sum_e dW_e dW_e^T @ Om.

Self-contained (synthetic experts, no HF model). Run: python3 test_update1363_rangefix.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_field_transform import (  # noqa: E402
    FieldSparseMoe, expert_basis_init, SVD_INIT_VER)

torch.set_num_threads(2)
FAILED = []


def ok(cond, name):
    print(("PASS  " if cond else "FAIL  ") + name, flush=True)
    if not cond:
        FAILED.append(name)


class _Exp:
    def __init__(self, Wgu, Wdn):
        self.gate_up_proj = Wgu
        self.down_proj = Wdn


class FakeBlock:
    def __init__(self, Wgu, Wdn):
        self.experts = _Exp(Wgu, Wdn)


def make_block(d=96, dff=48, n_exp=16, seed=100, r_true=10, outlier=None):
    g = torch.Generator().manual_seed(seed)

    def ln(a, b):
        return torch.randn(a, b, generator=g) * (1.0 / math.sqrt(b))

    m_gu, m_dn = ln(2 * dff, d) * 0.9, ln(d, dff) * 0.9
    Us_gu, Vs_gu = ln(2 * dff, r_true), ln(d, r_true)
    Us_dn, Vs_dn = ln(d, r_true), ln(dff, r_true)
    spec = torch.exp(-torch.arange(r_true, dtype=torch.float32) / 3.0)
    Wgu = torch.zeros(n_exp, 2 * dff, d)
    Wdn = torch.zeros(n_exp, d, dff)
    for e in range(n_exp):
        cg, cd = torch.randn(r_true, generator=g) * 0.5, \
            torch.randn(r_true, generator=g) * 0.5
        Ue = torch.randn(2 * dff, r_true, generator=g) * 0.05
        Ve = torch.randn(d, r_true, generator=g) * 0.05
        Ue2 = torch.randn(d, r_true, generator=g) * 0.05
        Ve2 = torch.randn(dff, r_true, generator=g) * 0.05
        Wgu[e] = m_gu + (Us_gu * cg.unsqueeze(0)) @ Vs_gu.t() \
            + (Ue * spec.unsqueeze(0)) @ Ve.t()
        Wdn[e] = m_dn + (Us_dn * cd.unsqueeze(0)) @ Vs_dn.t() \
            + (Ue2 * spec.unsqueeze(0)) @ Ve2.t()
    for W, m in ((Wgu, m_gu), (Wdn, m_dn)):
        cur = (W - m.unsqueeze(0)).norm() / m.norm().clamp_min(1e-12)
        W.copy_(m.unsqueeze(0) + (W - m.unsqueeze(0)) * (0.35 / cur))
    if outlier is not None:
        e, s = outlier
        Wgu[e] = m_gu + (Wgu[e] - m_gu) * s
        Wdn[e] = m_dn + (Wdn[e] - m_dn) * s
    return Wgu, Wdn


def subspace_capture(block, mgu, mdn, rank, side):
    """Energy of the deltas inside the pass-1 Q subspace (the joint-v1 bug
    metric: noise Q gave ~q/out_dim ~ 0.1-0.2, the fixed Q gives ~1)."""
    g = torch.Generator().manual_seed(917)
    m0 = mgu if side == "gu" else mdn
    outs, ins = m0.shape
    q = min(outs, rank + 16)
    Om = torch.randn(outs, q, generator=g) / math.sqrt(q)   # joint-v2 layout
    Ws = [(wgu if side == "gu" else wdn)
          for wgu, wdn in expert_basis_init_iter(block)]
    Y = torch.zeros(outs, q)
    den = 0.0
    for w in Ws:
        d = w - m0
        Y += d @ (d.t() @ Om)
        den += float(d.norm() ** 2)
    Q, _ = torch.linalg.qr(Y)
    num = sum(float(((Q.t() @ (w - m0)).norm()) ** 2) for w in Ws)
    return num / max(den, 1e-12)


def expert_basis_init_iter(block):
    exp = block.experts
    for e in range(exp.gate_up_proj.shape[0]):
        yield exp.gate_up_proj[e], exp.down_proj[e]


def main():
    rank = 8
    ok(SVD_INIT_VER == "joint-v2", f"t1 SVD_INIT_VER is joint-v2 "
       f"(got {SVD_INIT_VER})")
    for tag, kw in (("balanced", {}), ("outlier", dict(outlier=(3, 4.0)))):
        Wgu, Wdn = make_block(**kw)
        blk = FakeBlock(Wgu, Wdn)
        mgu, mdn = Wgu.mean(0), Wdn.mean(0)   # EXACT sample mean (the bug
        # needs sum_e dW_e == 0; a slightly-off center hides it)
        ini = expert_basis_init(blk, mgu.clone(), mdn.clone(), rank,
                                log_prefix=tag)
        sc_gu = subspace_capture(blk, mgu, mdn, rank, "gu")
        sc_dn = subspace_capture(blk, mgu, mdn, rank, "dn")
        ok(sc_gu > 0.95 and sc_dn > 0.95,
           f"t2[{tag}] pass-1 subspace capture ~1 (gu {sc_gu:.3f} "
           f"dn {sc_dn:.3f}; joint-v1 gave ~0.1-0.2)")
        ok(ini["capture"]["gu"] > 0.3 and ini["capture"]["dn"] > 0.3,
           f"t3[{tag}] diag capture healthy (gu {ini['capture']['gu']:.3f} "
           f"dn {ini['capture']['dn']:.3f}; joint-v1 gave ~0.1)")
        ok(str(ini["svd_ver"]).startswith("joint-v2"),
           f"t4[{tag}] init stamped {ini['svd_ver']}")
        geom = dict(n_exp=Wgu.shape[0], d_model=Wdn.shape[1],
                    d_ff=Wdn.shape[1] // 2, top_k=4, norm_topk=True,
                    field_mode="postact", core="diag", u_mode="none",
                    router_kind="softmax")
        mod = FieldSparseMoe(geom, rank, gate_w=torch.randn(
            Wgu.shape[0], Wdn.shape[1]), init=dict(ini),
            dtype=torch.float32)
        ok(all(float(getattr(mod, k).detach().norm()) > 0
               for k in ("Ugu", "Vgu", "Cgu", "Udn", "Vdn", "Cdn")),
           f"t5[{tag}] init transfers into FieldSparseMoe (nonzero U/V/C)")
    # bank-2 residual stage still runs on the fixed pass-1
    Wgu, Wdn = make_block()
    ini2 = expert_basis_init(FakeBlock(Wgu, Wdn), Wgu.mean(0).clone(),
                             Wdn.mean(0).clone(), rank, banks=2,
                             log_prefix="b2")
    ok("capture2" in ini2 and ini2["capture2"]["dn"] > 0.05,
       f"t6 banks=2 residual stage works (capture2 dn "
       f"{ini2['capture2']['dn']:.3f})")
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED")
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
