#!/usr/bin/env python3
"""probe_capacity.py - decisive ONE-BLOCK probe (task 29.8).

Question: the shipped fit's residual mse (e.g. blk20 0.694 -> 0.168) - is it
the OPTIMIZER's plateau or the PARAMETRIZATION's capacity ceiling?

Three arms, all on one block, all warm-started from the SHIPPED fit:
  eval   - re-measure the shipped field on the pool (baseline, no training)
  cont   - continue fitting the SAME r=128 single-bank field
  bank2  - add a SECOND coordinate bank (U2/V2 fresh, C2=0 -> +64 reachable
           mixture dims, breaks the min(n_exp, rank)=64 ceiling measured in
           the synthetic test) and fit
  postact- same parameters, but per-expert field branches with POST-activation
           mixing (out = sum_e z_e * FFN_e(x; W0 + U diag(C_e) V^T), i.e. the
           external review's "Variant A"). Decisive test of H1 (SwiGLU
           cross-terms of the pre-activation composition) on REAL weights:
           at top_k=1 the two compositions coincide exactly, at top_k=2 they
           differ by the cross-terms z1z2(g1*u2 + g2*u1).
  du     - ADD a per-expert DOWN-output delta bank (13.5, external review
           fig12/fig14): du_e = duA_e @ duB_e^T (fresh duA randn*0.02, duB=0
           LoRA start), rank a = --du-rank, mixed over the top-k AFTER the
           nonlinearity. Tests whether the residual is the OUTPUT-direction
           ceiling: the coordinate banks (cont/bank2) only address the
           span(Udn) output subspace (toy: novelty exactly 0 for V-side
           connectors), du adds NEW output directions (+E*(r+d)*a
           params/block). du << bank2 => the bottleneck is the output
           dictionary, not the mixture space.

If bank2 drops the mse materially below cont - capacity binds and the
two-bank mode becomes build 10.9. If cont == bank2 - the residual is the
low-rank/expressiveness ceiling itself and the design needs per-expert
bases instead. If postact drops materially below cont - the pre-activation
composition is the bottleneck and the runtime switches to Variant A.

Usage (from the folder with hf_pipeline.py):
  python probe_capacity.py --root <cache dir with pairs_blk*.pt> --block 20
  python probe_capacity.py --root . --block 20 --arm bank2 --steps 800
The cache dir is the one holding pairs_blk0.pt (see the stage-4 log line
"block 0: N pairs -> .../pairs_blk0.pt"); init_blk{B}.pt sits next to it,
fit_blk{B}.pt in its fit_r<rank> subdir (auto-found).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_field_transform import FieldSparseMoe, fit_field_module  # noqa: E402


class TwoBankField(FieldSparseMoe):
    """bank1 = the shipped fit (warm), bank2 = fresh U2/V2 with C2=0.
    Reachable mixture space: rowspan(C1) + rowspan(C2) -> up to 2*n_exp."""

    def __init__(self, geom, rank, gate_w, gate_bias, shared, init):
        super().__init__(geom, rank, gate_w=gate_w, act_fn=F.silu,
                         gate_bias=gate_bias, shared=shared, init=init)
        d, dff, r = geom["d_model"], geom["d_ff"], rank
        grng = torch.Generator().manual_seed(7654321)
        for nm, out, inp in (("gu", 2 * dff, d), ("dn", d, dff)):
            self.register_parameter(
                f"U2{nm}", nn.Parameter(torch.randn(out, r, generator=grng)
                                        * 0.02))
            self.register_parameter(
                f"V2{nm}", nn.Parameter(torch.randn(inp, r, generator=grng)
                                        * 0.02))
            self.field_names += [f"U2{nm}", f"V2{nm}"]
        for nm in ("gu", "dn"):
            self.register_parameter(
                f"C2{nm}", nn.Parameter(torch.zeros(geom["n_exp"], r)))
            self.field_names.append(f"C2{nm}")

    def forward_from_z(self, x, z):
        c1gu, c1dn = z @ self.Cgu, z @ self.Cdn
        c2gu, c2dn = z @ self.C2gu, z @ self.C2dn
        gu = x @ self.wgud.t() + (x @ self.Vgu * c1gu) @ self.Ugu.t() \
            + (x @ self.V2gu * c2gu) @ self.U2gu.t()
        g, u = gu.chunk(2, dim=-1)
        h = self.act_fn(g) * u
        return h @ self.wdnd.t() + (h @ self.Vdn * c1dn) @ self.Udn.t() \
            + (h @ self.V2dn * c2dn) @ self.U2dn.t()


class PostactField(FieldSparseMoe):
    """Identical parameters to the shipped field; the composition is the
    external review's "Variant A": per-expert branches, post-activation
    mixing. gu base (x @ wgud^T) is shared across the selected experts -
    only the low-rank deltas are per-expert, so the extra FLOPs vs the
    preact field are ~1 extra down-GEMM per token."""

    def forward_from_z(self, x, z):
        zt, ei = z.topk(min(self.k, z.shape[-1]), dim=-1)
        gu0 = x @ self.wgud.t()                  # shared pre-base
        y = None
        for j in range(zt.shape[-1]):
            cgu, cdn = self.Cgu[ei[:, j]], self.Cdn[ei[:, j]]
            gu = gu0 + (x @ self.Vgu * cgu) @ self.Ugu.t()
            g, u = gu.chunk(2, dim=-1)
            h = self.act_fn(g) * u               # NONLINEARITY PER EXPERT
            yj = zt[:, j:j + 1] * (
                h @ self.wdnd.t() + (h @ self.Vdn * cdn) @ self.Udn.t())
            y = yj if y is None else y + yj
        return y


def find_files(root, block, rank):
    pairs = init = fit = None
    for dirpath, _dirs, files in os.walk(root):
        p = f"pairs_blk{block}.pt"
        if p in files and pairs is None:
            pairs = os.path.join(dirpath, p)
            i = f"init_blk{block}.pt"
            if i in files:
                init = os.path.join(dirpath, i)
        if fit is None and f"fit_blk{block}.pt" in files \
                and f"fit_r{rank}" in dirpath:
            fit = os.path.join(dirpath, f"fit_blk{block}.pt")
    if init is None and pairs is not None:
        cand = os.path.join(os.path.dirname(pairs), f"init_blk{block}.pt")
        init = cand if os.path.isfile(cand) else None
    return pairs, init, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--arm", choices=["eval", "cont", "bank2", "postact", "du"],
                    default="eval")
    ap.add_argument("--du-rank", type=int, default=4,
                    help="rank a of the du delta bank for --arm du (13.5)")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--workers-threads", type=int, default=0,
                    help="0 = all cores")
    a = ap.parse_args()
    if a.workers_threads:
        torch.set_num_threads(a.workers_threads)

    pairs, init, fit = find_files(a.root, a.block, a.rank)
    for nm, p in (("pairs", pairs), ("init", init), ("fit", fit)):
        if p is None:
            sys.exit(f"{nm}_blk{a.block}.pt not found under {a.root}")
    print(f"pairs: {pairs}\ninit:  {init}\nfit:   {fit}", flush=True)

    d = torch.load(pairs, map_location="cpu")
    X, Y = d["X"], d["Y"]
    ini = torch.load(init, map_location="cpu")
    prev = torch.load(fit, map_location="cpu")
    geom = ini["geom"]

    if a.arm == "bank2":
        mod = TwoBankField(geom, a.rank, ini["gw"], ini.get("eb"),
                           ini.get("shared"), prev)
    elif a.arm == "postact":
        mod = PostactField(geom, a.rank, gate_w=ini["gw"],
                           gate_bias=ini.get("eb"), shared=ini.get("shared"),
                           init=prev)
    elif a.arm == "du":                              # 13.5: du-банк на Udn
        g = dict(geom) | {"u_mode": "rank4", "u_rank": a.du_rank}
        mod = FieldSparseMoe(g, a.rank, gate_w=ini["gw"],
                             gate_bias=ini.get("eb"),
                             shared=ini.get("shared"), init=prev)
    else:
        mod = FieldSparseMoe(geom, a.rank, gate_w=ini["gw"],
                             gate_bias=ini.get("eb"),
                             shared=ini.get("shared"), init=prev)
        if a.arm == "cont" and "gw_tuned" in prev:
            with torch.no_grad():
                mod.gw.copy_(prev["gw_tuned"])

    with torch.no_grad():
        z_all = mod._z(X.float())
        errs = []
        for s in range(0, X.shape[0], 4096):
            pred = mod.forward_from_z(X[s:s + 4096].float(),
                                      z_all[s:s + 4096])
            errs.append(((pred - Y[s:s + 4096].float()) ** 2).mean(1))
        base = float(torch.cat(errs).mean())
    print(f"[{a.arm}] shipped-field mse on the full pool: {base:.5f} "
          f"(pipeline guard said ~0.16780 for blk20 - sanity anchor)",
          flush=True)

    if a.arm == "eval":
        return

    t0 = time.time()
    mse = fit_field_module(mod, X, Y, steps=a.steps, bs=a.bs, lr=a.lr,
                           device="cpu", log_prefix=f"blk{a.block}:{a.arm}",
                           log_every=50, guard=False, method="muon-cosine",
                           seed=909 + a.block, jitter=a.jitter, early_stop=0,
                           autocast="off", muon_max_dim=512, muon_ns_steps=5,
                           lr_warmup=0)
    print(f"[{a.arm}] done in {time.time() - t0:.0f}s: {base:.5f} -> "
          f"{mse:.5f} ({100 * (base - mse) / max(base, 1e-12):.1f}% below "
          f"the shipped state)", flush=True)
    if a.arm == "postact":
        print(f"[{a.arm}] H1 reading: compare this final mse with 'cont' on "
              f"the same block - postact << cont means the pre-activation "
              f"composition (SwiGLU cross-terms) is the real bottleneck; "
              f"postact ~= cont means it is not (capacity story -> bank2)",
              flush=True)
    if a.arm == "du":
        print(f"[{a.arm}] du reading (13.5): compare with 'cont' AND 'bank2' "
              f"on the same block - du << cont means new OUTPUT directions "
              f"pay (output-dictionary ceiling, -> --u-mode rank{a.du_rank}/"
              f"full in the pipeline); du ~= cont means the output side is "
              f"not the binding constraint (mixture-space story -> bank2)",
              flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"probe_blk{a.block}_{a.arm}.pt")
    torch.save({n: getattr(mod, n).detach().clone()
                for n in mod.field_names}, out)
    print(f"weights saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
