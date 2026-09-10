#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""update-13.2 tests: спайк-проба (whbank_build --variants spk) и
deploy_check (фит vs записанный артефакт).

1. spike math: гауссовский остаток -> awq@1% заметно ниже порога топлива;
   остаток с инжектированными пиками (0.3% элементов x40) -> awq@1% сильно
   выше гауссовского ориентира (топливо ЕСТЬ).
2. deploy_check: мини-артефакт (safetensors, bf16, правильные имена) с
   параметрами фит-модуля -> все тензоры OK, mse ratio ~1, exit 0;
   искалеченный Cgu (x1.5) -> FAIL по таблице, exit 1.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
# 13.5.1: resolve from THIS file's location - the folder is portable now,
# the old hardcode /home/z/my-project/ep13 broke after every re-unpack
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import whbank_build as WB                       # noqa: E402
from hf_field_transform import FieldSparseMoe, apply_core  # noqa: E402

ok, fail = [], []


def check(name, fn):
    try:
        fn()
        ok.append(name)
        print(f"PASS  {name}", flush=True)
    except Exception:
        fail.append(name)
        print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)


class _DiagWh:
    """Диагональный отбеливатель: Mw = Sy12 M Sx12, обратные ходы."""

    def __init__(self, m, n, seed=0, scale=0.3):
        g = np.random.default_rng(seed)
        self.Sy12 = np.diag(1.0 + scale * g.random(m)).astype(np.float32)
        self.Sx12 = np.diag(1.0 + scale * g.random(n)).astype(np.float32)
        self.Sy_i = np.diag(1.0 / np.diag(self.Sy12)).astype(np.float32)
        self.Sx_i = np.diag(1.0 / np.diag(self.Sx12)).astype(np.float32)

    def fw(self, M):
        return self.Sy12 @ M @ self.Sx12


def _write_bank(whdir, rank, kind, U, V, cores):
    d = os.path.join(whdir, "t2_k%d" % rank)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "blk00_%s.npz" % kind),
             U=U, V=V, cores=cores, caps=np.zeros(cores.shape[0], np.float32),
             meta=np.array([rank, 0, 0], np.int64))


def _run_spike(dims, n_exp, rank, spike_frac, spike_amp, seed=11):
    """Строит дельты (T2-проекция в фиксированном базисе [+ пики]), пишет
    банк файлов и вызывает spike_block с подменённым iter_deltas.
    Возвращает (rows, captured_stdout)."""
    rng = np.random.default_rng(seed)
    tmp = tempfile.mkdtemp(prefix="spk_")
    covs = {k: _DiagWh(*dims[k], seed=100 + j) for j, k in enumerate(dims)}
    bases, crafted = {}, {k: [] for k in dims}
    for kind, (m, n) in dims.items():
        U, _ = np.linalg.qr(rng.standard_normal((m, rank)))
        V, _ = np.linalg.qr(rng.standard_normal((n, rank)))
        bases[kind] = (U.astype(np.float32), V.astype(np.float32))
    for e in range(n_exp):
        for kind, (m, n) in dims.items():
            U, V = bases[kind]
            Uw = covs[kind].Sy12 @ U
            Vw = covs[kind].Sx12 @ V          # raw V -> whitened Vw
            Mw = rng.standard_normal((m, n)).astype(np.float32)
            if spike_frac and e % 2 == 0:      # каждый 2-й эксперт - с пиками
                R = rng.standard_normal((m, n)).astype(np.float32)
                k_sp = max(1, int(spike_frac * m * n))
                R.ravel()[rng.choice(m * n, k_sp, replace=False)] *= spike_amp
                Mw = Mw + R
            crafted[kind].append(Mw)           # дельта = Mw (identity-базис)
    orig_iter = WB.iter_deltas
    WB.iter_deltas = lambda *a, **k: iter(
        [(crafted["gu"][e], crafted["dn"][e]) for e in range(n_exp)])
    try:
        for kind, (m, n) in dims.items():
            U, V = bases[kind]
            Uw = covs[kind].Sy12 @ U
            Vw = covs[kind].Sx12 @ V          # raw V -> whitened Vw
            cores = np.stack([(Uw.T @ M @ Vw).astype(np.float32)
                              for M in crafted[kind]])
            _write_bank(tmp, rank, kind, U, V, cores)
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rows = WB.spike_block(
                0, block=None, mgu=None, mdn=None,
                geom={"d_model": dims["gu"][1], "d_ff": dims["dn"][1],
                      "n_exp": n_exp},
                covs=covs, whdir=tmp, rank=rank,
                fracs=(0.001, 0.005, 0.01, 0.02))
        return rows, buf.getvalue(), tmp
    finally:
        WB.iter_deltas = orig_iter
        shutil.rmtree(tmp, ignore_errors=True)


def t_spike_gauss_vs_spikes():
    ref1 = WB.gauss_topk_share(0.01)
    assert abs(ref1 - 0.144) < 0.01, ref1
    dims = {"gu": (96, 64), "dn": (64, 48)}
    # чистый гаусс (spike_frac=0): awq@1% ниже порога топлива 25%
    rows0, out0, _ = _run_spike(dims, 4, 8, 0.0, 0.0, seed=11)
    assert "НЕ подтверждён" in out0, out0
    # пики: 0.3% элементов x40 у половины экспертов -> топливо ЕСТЬ
    rows1, out1, _ = _run_spike(dims, 6, 8, 0.003, 40.0, seed=12)
    assert "ЕСТЬ" in out1, out1
    # и capture с пиками должен быть заметно выше, чем на чистом гауссе
    awq0 = [r[3] for r in rows0]
    awq1 = [r[3] for r in rows1]
    assert all(b > a for a, b in zip(awq0, awq1)), (awq0, awq1)


def t_deploy_check_ok():
    root = tempfile.mkdtemp(prefix="dep_ok_")
    pool, fit_dir, art = _mk_fixture(root, mangle=False)
    r = subprocess.run(
        [sys.executable, os.path.join(BASE, "deploy_check.py"),
         "--pool", pool, "--fit-dir", fit_dir, "--artifact", art,
         "--blocks", "0"], capture_output=True, text=True, timeout=300)
    sys.stdout.write(r.stdout[-1500:])
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-800:])
    assert "DEPLOY CHECK: OK" in r.stdout
    assert "FAIL" not in r.stdout
    shutil.rmtree(root, ignore_errors=True)


def t_deploy_check_catches_mangling():
    root = tempfile.mkdtemp(prefix="dep_bad_")
    pool, fit_dir, art = _mk_fixture(root, mangle=True)
    r = subprocess.run(
        [sys.executable, os.path.join(BASE, "deploy_check.py"),
         "--pool", pool, "--fit-dir", fit_dir, "--artifact", art,
         "--blocks", "0"], capture_output=True, text=True, timeout=300)
    sys.stdout.write(r.stdout[-1500:])
    assert r.returncode == 1, (r.returncode, r.stdout[-2000:], r.stderr[-800:])
    assert "Cgu" in r.stdout and "FAIL" in r.stdout
    shutil.rmtree(root, ignore_errors=True)


def _mk_fixture(root, mangle=False):
    """Мини field-артефакт + пул + фит-файлы (dense core, preact)."""
    g = torch.Generator().manual_seed(9)
    geom = {"n_exp": 8, "d_model": 64, "d_ff": 48, "top_k": 2,
            "norm_topk": False, "banks": 1}
    rank = 12
    gw = torch.randn(geom["n_exp"], geom["d_model"], generator=g) * 0.1
    mod = FieldSparseMoe(apply_core(geom, True) | {"field_mode": "preact"},
                         rank, gate_w=gw, act_fn=F.silu, dtype=torch.float32,
                         init={})
    with torch.no_grad():
        for n in ("Ugu", "Vgu", "Udn", "Vdn"):
            getattr(mod, n).normal_(0, 0.05, generator=g)
        mod.Cgu.normal_(0, 0.05, generator=g)
        mod.Cdn.normal_(0, 0.05, generator=g)
    params = {n: getattr(mod, n).detach().clone() for n in mod.field_names}
    X = torch.randn(4096, geom["d_model"], generator=g)
    with torch.no_grad():
        Y = mod.forward(X.view(1, -1, geom["d_model"])) \
            .reshape(-1, geom["d_model"])
    os.makedirs(root, exist_ok=True)
    torch.save({"X": X.to(torch.bfloat16), "Y": Y.to(torch.bfloat16)},
               os.path.join(root, "pairs_blk0.pt"))
    torch.save({"geom": geom, "gw": gw,
                "mgu": params["wgud"].clone(), "mdn": params["wdnd"].clone()},
               os.path.join(root, "init_blk0.pt"))
    fit_dir = os.path.join(root, "fit_r12t2")
    os.makedirs(fit_dir, exist_ok=True)
    torch.save(params, os.path.join(fit_dir, "fit_blk0.pt"))
    art = os.path.join(root, "artifact")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "config.json"), "w") as f:
        json.dump({"field": {"rank": rank, "core": "dense",
                             "field_mode": "preact"}}, f)
    from safetensors.torch import save_file
    sd = {}
    for n, t in params.items():
        tt = t.to(torch.bfloat16)
        if mangle and n == "Cgu":
            tt = (t * 1.5).to(torch.bfloat16)
        sd["model.layers.3.mlp." + n] = tt.contiguous()
    sd["model.layers.3.mlp.gate.weight"] = gw.to(torch.bfloat16).contiguous()
    save_file(sd, os.path.join(art, "model.safetensors"))
    return root, fit_dir, art


for name, fn in [
        ("spike probe: gauss below threshold, spikes above",
         t_spike_gauss_vs_spikes),
        ("deploy_check: faithful artifact passes", t_deploy_check_ok),
        ("deploy_check: mangled Cgu is caught",
         t_deploy_check_catches_mangling)]:
    check(name, fn)

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
