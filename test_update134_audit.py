#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_update134_audit.py - тесты init_audit.py (UPDATE-13.4).

Синтетический сценарий: "истина" = FieldSparseMoe в postact с ТОЧНЫМИ
t2-параметрами (дельты представимы без остатка). Тогда на её же пуле:

  t1: HANDOFF npz->init_svd: OK (бит-в-бит)            - раунд-трип t2_init_dict
  t2: CENTROID > INIT и POSTACT(init) << INIT          - армы меряют то, что заявлено;
                                                         вердикт о кросс-членах срабатывает
  t3: FIT-арм читает fit_blk и отличается от INIT      - загрузка параметров фита
  t4: подменённый init_svd -> HANDOFF MISMATCH         - курок ловится
  t5: LONGFIT-арм отрабатывает (--longfit-steps 120)

Запуск: /path/to/python scripts/test_update134_audit.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

# 13.5.1: resolve from THIS file's location (old hardcode /home/z/my-project/ep13
# broke after the folder was re-unpacked elsewhere)
EP13 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EP13)

D, DFF, R, E, T = 32, 64, 16, 8, 4096


def build_synthetic(root):
    torch.manual_seed(11)
    np.random.seed(11)
    pool = os.path.join(root, "pool")
    fitd = os.path.join(root, "fit")
    whd = os.path.join(fitd, "whbank")
    os.makedirs(pool, exist_ok=True)
    os.makedirs(whd, exist_ok=True)

    Ugu = torch.randn(2 * DFF, R) * 0.05
    Vgu = torch.randn(D, R) * 0.05
    Cgu = torch.randn(E, R, R) * 0.4
    Udn = torch.randn(D, R) * 0.05
    Vdn = torch.randn(DFF, R) * 0.05
    Cdn = torch.randn(E, R, R) * 0.4
    mgu = torch.randn(2 * DFF, D) * 0.1
    mdn = torch.randn(D, DFF) * 0.1
    gw = torch.randn(E, D) * 0.3

    geom = dict(d_model=D, d_ff=DFF, n_exp=E, top_k=2, norm_topk=True,
                router_kind="softmax", router_scale=1.0, dff_shexp=0)
    ini = dict(geom=geom, gw=gw, mgu=mgu, mdn=mdn)

    from hf_field_transform import FieldSparseMoe, apply_core

    def make(mode, init_extra):
        g = apply_core(geom, True) | {"field_mode": mode}
        m = FieldSparseMoe(g, R, gate_w=gw, init=dict(ini, **init_extra))
        with torch.no_grad():
            m.wgud.copy_(mgu)
            m.wdnd.copy_(mdn)
        return m

    exact = dict(Ugu=Ugu, Vgu=Vgu, Cgu=Cgu, Udn=Udn, Vdn=Vdn, Cdn=Cdn)
    truth = make("postact", exact)                     # истина: точный postact
    X = torch.randn(1, T, D)
    with torch.no_grad():
        Y = truth.forward(X)
    X, Y = X.reshape(T, D), Y.reshape(T, D)

    torch.save({"X": X.to(torch.bfloat16), "Y": Y.to(torch.bfloat16)},
               os.path.join(pool, "pairs_blk0.pt"))
    torch.save(ini, os.path.join(pool, "init_blk0.pt"))

    # npz банка (как их пишет whbank_build.build_block_t2) + init_svd через
    # РЕАЛЬНУЮ t2_init_dict (раунд-трип = тест хендоффа)
    os.makedirs(os.path.join(whd, "t2_k%d" % R), exist_ok=True)
    for side, U, V, C in (("gu", Ugu, Vgu, Cgu), ("dn", Udn, Vdn, Cdn)):
        np.savez(os.path.join(whd, "t2_k%d" % R, "blk00_%s.npz" % side),
                 U=U.numpy().astype(np.float32),
                 V=V.numpy().astype(np.float32),
                 cores=C.numpy().astype(np.float32),
                 meta=np.array([R, 0, 0], dtype=np.int64),
                 caps=np.ones(E, dtype=np.float32) * 0.99)
    import whbank_build as wb
    svd = wb.t2_init_dict(0, geom, R, whd)
    torch.save(svd, os.path.join(fitd, "init_svd_blk0.pt"))

    # fit_blk: слегка испорченные ядра (фит-арм должен от них отличаться)
    fit_out = dict(Ugu=Ugu, Vgu=Vgu, Cgu=Cgu + 0.05 * torch.randn(E, R, R),
                   Udn=Udn, Vdn=Vdn, Cdn=Cdn + 0.05 * torch.randn(E, R, R),
                   wgud=mgu.clone(), wdnd=mdn.clone())
    torch.save(fit_out, os.path.join(fitd, "fit_blk0.pt"))
    with open(os.path.join(fitd, "fit_meta.json"), "w") as f:
        json.dump({"field_mode": "preact"}, f)
    return pool, fitd, svd


def run_audit(pool, fitd, extra=()):
    cmd = [sys.executable, os.path.join(EP13, "init_audit.py"),
           "--pool", pool, "--fit-dir", fitd,
           "--blocks", "0", "--rank", str(R)] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise AssertionError("init_audit упал:\n" + p.stdout[-3000:]
                             + "\n" + p.stderr[-2000:])
    return p.stdout


def grab(out, key):
    m = re.search(r"block 0:.*" + key + r"\s+([0-9.eE+-]+)", out)
    return float(m.group(1)) if m else None


def main():
    root = tempfile.mkdtemp(prefix="t134_")
    try:
        pool, fitd, svd = build_synthetic(root)

        # t1-t3: полный прогон
        out = run_audit(pool, fitd)
        assert "HANDOFF npz->init_svd: OK" in out, "t1: handoff OK не найден"
        init = grab(out, "INIT")
        cen = grab(out, "CENTROID")
        fit = grab(out, r"\| FIT")
        post = grab(out, r"POSTACT\(init\)")
        postf = grab(out, "POSTACT-FIT")
        assert init is not None and cen is not None and post is not None, \
            "t2: не разобрать числа:\n" + out
        assert cen > 2.0 * init, "t2: CENTROID (%g) должен быть >> INIT (%g)" \
            % (cen, init)
        assert post < 0.5 * init, \
            "t2: POSTACT (%g) должен быть << INIT (%g) - истина же postact" \
            % (post, init)
        assert "* block 0: POSTACT" in out, "t2: вердикт о кросс-членах не сработал"
        assert fit is not None and abs(fit - init) > 0, \
            "t3: FIT-арм не читается/не отличается"
        assert postf is not None, "t3: POSTACT-FIT отсутствует"
        print("t1 PASS  handoff бит-в-бит")
        print("t2 PASS  INIT %g < POSTACT %g < CENTROID %g, вердикт сработал"
              % (init, post, cen))
        print("t3 PASS  FIT %g / POSTACT-FIT %g прочитаны" % (fit, postf))

        # t4: подмена init_svd -> MISMATCH
        root2 = tempfile.mkdtemp(prefix="t134b_")
        try:
            pool2 = os.path.join(root2, "pool")
            fitd2 = os.path.join(root2, "fit")
            shutil.copytree(pool, pool2)
            shutil.copytree(fitd, fitd2)
            svd2 = dict(svd)
            svd2["Cgu"] = svd["Cgu"] + 0.01
            torch.save(svd2, os.path.join(fitd2, "init_svd_blk0.pt"))
            out2 = run_audit(pool2, fitd2)
            assert "MISMATCH" in out2, "t4: подмена не поймана"
            print("t4 PASS  подмена init_svd ловится (MISMATCH)")
        finally:
            shutil.rmtree(root2, ignore_errors=True)

        # t5: LONGFIT-арм
        out3 = run_audit(pool, fitd, ["--longfit-steps", "120"])
        lf = grab(out3, "LONGFIT")
        assert lf is not None, "t5: LONGFIT-арм не отработал"
        print("t5 PASS  LONGFIT %g посчитан" % lf)

        print("\nALL PASSED (5)")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
