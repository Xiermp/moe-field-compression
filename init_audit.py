#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""init_audit.py - UPDATE-13.4: куда девается ценность t2-инита?

Парадокс прогона 13.3 (guard зелёный на всех 23 блоках, KL 1.535): на КАЖДОМ
блоке init baseline (eval8) совпадает с pure-centroid линией в пределах +-2%,
хотя whitened capture - gu 99.1% / dn 72.3%. По формуле banner'а step-0 mse
должен сидеть на ~(1-capture) x pure-centroid (~0.3x), а не ~1.0x.

Инструмент раскладывает разрыв на взаимоисключающие причины, по блокам, на
ТЕХ ЖЕ парах пула, на которых шёл stage-5 (модель не нужна для базовых армов):

  INIT          модуль ровно как его строит stage-5 (init_svd_blk) -> ~ step-0 mse из лога
  CENTROID      тот же модуль, ядра Cgu/Cdn = 0                   -> ~ pure-centroid из лога
  FIT           параметры из fit_blk (если есть)                  -> ~ финальный mse из лога
  POSTACT       INIT-параметры, field_mode=postact                -> цена композиции
                (кросс-члены SwiGLU: смесь пре-активаций против
                смеси пост-активаций; postact существует с 10.9)
  POSTACT-FIT   FIT-параметры в postact
  LONGFIT       (опц. --longfit-steps N) продолжение полировки из INIT
                (adamw, train=cores, lr --longfit-lr)             -> тест недосходимости фита

Плюс три проверки:
  HANDOFF       ядра/U/V из npz t2_k{rank} vs init_svd_blk{i}.pt - бит-в-бит?
                (тихое расхождение = курок: модуль получает не то, что
                построил банк; проверяется ВСЕГДА, модель не нужна)
  LOAD x CAP    загрузка экспертов (замороженный роутер на пуле) vs
                per-expert whitened caps из npz -> проверка «mean-of-ratios»:
                средняя капа может прятать горячих экспертов с низкой капой
  RAW CAP       (опц. --gguf/--src/--local-path) честный Frobenius capture
                по КАЖДОМУ эксперту из дельт модели (raw_capture из 13.3;
                банк этого прогона старый - caps_raw в нём нет)

Вердикты печатаются в конце, по приоритету. Пример:

  python init_audit.py --pool results/cache_..._r64 --fit-dir results/field_..._r64 \
      --blocks 0,10,20 --rank 64
  # с честной капой по экспертам (один стриминг-проход на блок):
  python init_audit.py ... --gguf <файл.gguf> --src <HF-каталог>
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from hf_field_transform import (FieldSparseMoe, apply_core,  # noqa: E402
                                fit_field_module, load_pairs_block)

FP = np.float32


# ----------------------------------------------------------------- утилиты --
def resolve_act(name):
    name = (name or "silu").strip().lower()
    try:
        from transformers.activations import ACT2FN
        if name in ACT2FN:
            return ACT2FN[name]
    except Exception:                                          # noqa: BLE001
        pass
    m = {"silu": F.silu, "swiglu": F.silu, "gelu": F.gelu, "relu": F.relu}
    if name in m:
        return m[name]
    raise SystemExit("init_audit: неизвестный --act %r" % name)


def parse_blocks(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def find_pool_blocks(pool_dir):
    fs = sorted(f for f in os.listdir(pool_dir)
                if f.startswith("pairs_blk") and f.endswith(".pt"))
    return [int(f[len("pairs_blk"):-len(".pt")]) for f in fs]


@torch.no_grad()
def full_mse(mod, Xf, tgt, z_all, bs, device):
    """Полно-пульный mse с чистым роутингом (та же метрика, что step-0/финал
    в логе stage-5, но детерминированная по всему пулу, а не 8 батчам)."""
    tot, n = 0.0, 0
    for s in range(0, Xf.shape[0], bs):
        xb = Xf[s:s + bs].to(device)
        zb = z_all[s:s + bs].to(device)
        yb = tgt[s:s + bs].to(device)
        d = mod.forward_from_z(xb, zb).float() - yb
        tot += float((d * d).sum().item())
        n += d.numel()
    return tot / max(n, 1)


def build_module(geom, rank, ini, fit_init, field_mode, core, act_fn, banks,
                 device, u_mode="none", u_rank=4):
    g = apply_core(geom, core == "dense") | {"field_mode": field_mode,
                                             "u_mode": u_mode,
                                             "u_rank": u_rank}
    mod = FieldSparseMoe(g, rank, gate_w=ini["gw"], act_fn=act_fn,
                         gate_bias=ini.get("eb"), shared=ini.get("shared"),
                         banks=banks, init=fit_init).to(device)
    with torch.no_grad():
        mod.wgud.copy_(ini["mgu"])
        mod.wdnd.copy_(ini["mdn"])
    return mod


def pool_targets(mod, Xi, Yi):
    """Точная копия подготовки fit_field_module: Xf, замороженный z, tgt
    (shared-ветка hy_v3 складывается в цель)."""
    with torch.no_grad():
        Xf = Xi.float()
        z_all = mod._z(Xf)
        if hasattr(mod, "sh_gu"):
            sg, su = (Xf @ mod.sh_gu.t()).chunk(2, dim=-1)
            tgt = Yi.float() - (mod.act_fn(sg) * su) @ mod.sh_dn.t()
        else:
            tgt = Yi.float()
    return Xf, z_all, tgt


# ------------------------------------------------------------- handoff/caps --
def handoff_check(npz_dir, rank, i, svd):
    """npz t2_k{rank} vs init_svd_blk - бит-в-бит? (t2_init_dict делает
    ascontiguousarray+from_numpy, т.е. расхождение = курок)."""
    res = {}
    for side in ("gu", "dn"):
        p = os.path.join(npz_dir, "t2_k%d" % rank, "blk%02d_%s.npz" % (i, side))
        if not os.path.isfile(p):
            res[side] = dict(status="npz-missing")
            continue
        z = np.load(p)
        k = int(z["meta"][0])
        rr = min(rank, k)
        out = dict(status="ok", k=k)
        for npz_key, pt_key in (("U", "U" + side), ("V", "V" + side),
                                ("cores", "C" + side)):
            a = z[npz_key]
            if npz_key == "cores":
                a = a[:, :rr, :rr]
            else:
                a = a[:, :rr]
            b = svd[pt_key].numpy()
            d = float(np.abs(a.astype(FP) - b.astype(FP)).max()) \
                if a.shape == b.shape else float("inf")
            out[pt_key] = d
        out["caps"] = z["caps"].astype(np.float64)
        out["caps_raw"] = (z["caps_raw"].astype(np.float64)
                           if "caps_raw" in z.files else None)
        out["ok"] = all(out[x] == 0.0 for x in ("Ugu", "Vgu", "Cgu", "Udn",
                                                "Vdn", "Cdn") if x in out)
        res[side] = out
    return res


def loadx_caps(z_all, caps_gu, caps_dn, n_exp):
    """Загрузка экспертов (масса весов смеси + токены) vs per-expert caps."""
    mass = z_all.sum(0).double().cpu().numpy()
    toks = (z_all > 0).sum(0).double().cpu().numpy()
    tot = float(mass.sum()) or 1.0
    order = np.argsort(-mass)
    hot = order[:max(1, n_exp // 4)]
    rows = []
    for e in order[:8]:
        rows.append((int(e), 100.0 * mass[e] / tot, int(toks[e]),
                     caps_gu[e] if caps_gu is not None else float("nan"),
                     caps_dn[e] if caps_dn is not None else float("nan")))
    summary = {}
    for side, caps in (("gu", caps_gu), ("dn", caps_dn)):
        if caps is None:
            continue
        c = np.asarray(caps, dtype=np.float64)
        summary[side] = dict(
            mean=float(np.mean(c)) * 100.0,
            hot_mean=float(np.mean(c[hot])) * 100.0,
            min=float(np.min(c)) * 100.0,
            max=float(np.max(c)) * 100.0)
    return rows, summary


def raw_caps_stream(args, i, ini, npz_dir, rank):
    """Честный raw capture по КАЖДОМУ эксперту из дельт модели (один
    стриминг-проход на блок). Требует источник весов."""
    import whbank_build as wb
    model, stream, blocks = wb.load_model(args.src, args.gguf, args.local_path)
    if i >= len(blocks):
        raise SystemExit("init_audit: блока %d нет в источнике (%d всего)"
                         % (i, len(blocks)))
    block = blocks[i][1]
    ctx = stream.with_block(i) if stream is not None \
        else contextlib.nullcontext()
    caps = {"gu": [], "dn": []}
    with ctx:
        z = {}
        for side in ("gu", "dn"):
            z[side] = np.load(os.path.join(npz_dir, "t2_k%d" % rank,
                                           "blk%02d_%s.npz" % (i, side)))
        Ugu, Vgu = z["gu"]["U"][:, :rank], z["gu"]["V"][:, :rank]
        Udn, Vdn = z["dn"]["U"][:, :rank], z["dn"]["V"][:, :rank]
        Cgu = z["gu"]["cores"][:, :rank, :rank]
        Cdn = z["dn"]["cores"][:, :rank, :rank]
        mgu, mdn = ini["mgu"], ini["mdn"]
        for e, (dgu, ddn) in enumerate(wb.iter_deltas(block, mgu, mdn)):
            caps["gu"].append(wb.raw_capture(dgu, Ugu, Vgu, Cgu[e]))
            caps["dn"].append(wb.raw_capture(ddn, Udn, Vdn, Cdn[e]))
    del model
    return {s: np.asarray(v, dtype=np.float64) for s, v in caps.items()}


# --------------------------------------------------------------------- CLI --
def main():
    ap = argparse.ArgumentParser(
        description="UPDATE-13.4: аудит t2-инита (куда девается capture)")
    ap.add_argument("--pool", required=True,
                    help="каталог кэша пула (pairs_blk*.pt, init_blk*.pt)")
    ap.add_argument("--fit-dir", required=True,
                    help="каталог фита (init_svd_blk*.pt, fit_blk*.pt, "
                         "whbank/t2_k*/)")
    ap.add_argument("--blocks", default="0,10,20")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--banks", type=int, default=1)
    ap.add_argument("--core", default="dense", choices=["dense", "diag"])
    ap.add_argument("--field-mode", default=None,
                    choices=["preact", "postact"],
                    help="по умолчанию берётся из fit_meta.json (иначе preact)")
    ap.add_argument("--u-mode", default=None,
                    choices=["none", "rank1", "rank4", "full"],
                    help="13.5: по умолчанию берётся из fit_meta.json "
                         "(иначе none) - аудит обязан видеть тот же модуль, "
                         "что и артефакт (du-параметры не должны молча "
                         "отваливаться)")
    ap.add_argument("--u-rank", type=int, default=None,
                    help="ранг a du-дельты (по умолчанию из fit_meta.json)")
    ap.add_argument("--act", default="silu")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bs", type=int, default=8192,
                    help="кусок пула для полно-пульного mse")
    ap.add_argument("--longfit-steps", type=int, default=0,
                    help=">0: продолжить полировку из INIT (тест "
                         "недосходимости фита); профиль adamw/train=cores")
    ap.add_argument("--longfit-lr", type=float, default=1e-4)
    ap.add_argument("--longfit-bs", type=int, default=4096)
    ap.add_argument("--longfit-method", default="adamw")
    ap.add_argument("--longfit-early-stop", type=int, default=0)
    ap.add_argument("--gguf", default=None,
                    help="GGUF-файл весов (для RAW CAP; вместе с --src)")
    ap.add_argument("--src", default=None,
                    help="HF-каталог/light-catalog (конфиг+скелет; как в "
                         "hf_pipeline)")
    ap.add_argument("--local-path", default=None,
                    help="полный HF-лоад (маленькие модели; альтернатива "
                         "--src/--gguf)")
    args = ap.parse_args()
    if args.gguf and not (args.src or args.local_path):
        raise SystemExit("init_audit: для GGUF нужен ещё и --src (HF-каталог "
                         "с конфигом, тот же, что в hf_pipeline) - или "
                         "используй --local-path для маленьких моделей")

    torch.set_grad_enabled(False)
    device = args.device
    act_fn = resolve_act(args.act)
    blocks = sorted(set(parse_blocks(args.blocks)) & set(find_pool_blocks(args.pool)))
    if not blocks:
        raise SystemExit("init_audit: в %s нет pairs_blk*.pt" % args.pool)
    npz_dir = os.path.join(args.fit_dir, "whbank")
    fit_meta = {}
    mp = os.path.join(args.fit_dir, "fit_meta.json")
    if os.path.isfile(mp):
        try:
            with open(mp, encoding="utf-8") as f:
                fit_meta = json.load(f)
        except Exception:                                      # noqa: BLE001
            pass
    field_mode = args.field_mode or fit_meta.get("field_mode", "preact")
    u_mode = args.u_mode or fit_meta.get("u_mode", "none")      # 13.5
    u_rank = args.u_rank or int(fit_meta.get("u_rank", 4))
    have_source = bool(args.gguf or args.src or args.local_path)

    print("init_audit (13.5): pool=%s fit=%s blocks=%s rank=%d mode=%s "
          "core=%s u_mode=%s(a=%s)" % (args.pool, args.fit_dir, blocks,
                                       args.rank, field_mode, args.core,
                                       u_mode, u_rank), flush=True)
    if not have_source:
        print("  RAW CAP выключен: нет --gguf/--src/--local-path "
              "(per-expert честная капа не считается)", flush=True)

    results = {}
    for i in blocks:
        print("\n=== block %d ===" % i, flush=True)
        Xi, Yi = load_pairs_block(os.path.join(args.pool,
                                               "pairs_blk%d.pt" % i))
        ini = torch.load(os.path.join(args.pool, "init_blk%d.pt" % i),
                         map_location="cpu")
        geom = ini["geom"]
        svd_p = os.path.join(args.fit_dir, "init_svd_blk%d.pt" % i)
        svd = torch.load(svd_p, map_location="cpu") \
            if os.path.isfile(svd_p) else None
        fit_init = dict(ini)
        if svd:
            fit_init.update(svd)

        # ---- HANDOFF (всегда): npz vs init_svd бит-в-бит
        ho = handoff_check(npz_dir, args.rank, i, svd) if svd else {}
        if ho:
            ho_ok = all(v.get("ok") for v in ho.values())
            mds = [max(v.get(x, 0.0) for x in ("U" + s, "V" + s, "C" + s))
                   for s, v in ho.items() if v.get("status") == "ok"]
            print("  HANDOFF npz->init_svd: %s (max|d| %.3g)"
                  % ("OK (бит-в-бит)" if ho_ok else "MISMATCH - курок!",
                     max(mds) if mds else float("nan")), flush=True)

        # ---- INIT / CENTROID / FIT / POSTACT
        mod = build_module(geom, args.rank, ini, fit_init, field_mode,
                           args.core, act_fn, args.banks, device,
                           u_mode, u_rank)
        Xf, z_all, tgt = pool_targets(mod, Xi, Yi)
        r = {}
        r["INIT"] = full_mse(mod, Xf, tgt, z_all, args.bs, device)
        cgu0, cdn0 = mod.Cgu.detach().clone(), mod.Cdn.detach().clone()
        with torch.no_grad():
            mod.Cgu.zero_()
            mod.Cdn.zero_()
        r["CENTROID"] = full_mse(mod, Xf, tgt, z_all, args.bs, device)
        with torch.no_grad():
            mod.Cgu.copy_(cgu0)
            mod.Cdn.copy_(cdn0)

        fit_p = os.path.join(args.fit_dir, "fit_blk%d.pt" % i)
        if os.path.isfile(fit_p):
            out = torch.load(fit_p, map_location="cpu")
            mod2 = build_module(geom, args.rank, ini, {}, field_mode,
                                args.core, act_fn, args.banks, device,
                                u_mode, u_rank)
            missing, unexpected = mod2.load_state_dict(
                {k: v for k, v in out.items()
                 if k in dict(mod2.named_parameters())}, strict=False)
            r["FIT"] = full_mse(mod2, Xf, tgt, z_all, args.bs, device)
            del mod2
        else:
            r["FIT"] = None

        # postact: те же параметры, другая композиция
        modp = build_module(geom, args.rank, ini, fit_init, "postact",
                            args.core, act_fn, args.banks, device,
                            u_mode, u_rank)
        r["POSTACT"] = full_mse(modp, Xf, tgt, z_all, args.bs, device)
        if r["FIT"] is not None:
            modf = build_module(geom, args.rank, ini, {}, "postact",
                                args.core, act_fn, args.banks, device,
                                u_mode, u_rank)
            modf.load_state_dict(
                {k: v for k, v in out.items()
                 if k in dict(modf.named_parameters())}, strict=False)
            r["POSTACT-FIT"] = full_mse(modf, Xf, tgt, z_all, args.bs, device)
            del modf
        del modp

        # ---- LONGFIT (опция): продолжение полировки из INIT
        if args.longfit_steps > 0:
            modl = build_module(geom, args.rank, ini, fit_init, field_mode,
                                args.core, act_fn, args.banks, device,
                                u_mode, u_rank)
            torch.set_grad_enabled(True)
            fit_field_module(modl, Xi, Yi, args.longfit_steps,
                             args.longfit_bs, args.longfit_lr, device,
                             log_prefix="audit longfit blk%d" % i,
                             method=args.longfit_method, seed=5 + i,
                             lr_warmup=20, train="cores",
                             guard=False, strict_guard=False,
                             early_stop=args.longfit_early_stop)
            torch.set_grad_enabled(False)
            r["LONGFIT"] = full_mse(modl, Xf, tgt, z_all, args.bs, device)
            del modl

        # ---- LOAD x CAP (mean-of-ratios check)
        caps_gu = caps_dn = None
        if ho.get("gu", {}).get("caps") is not None:
            caps_gu = ho["gu"]["caps"]
            caps_dn = ho["dn"].get("caps")
            r["WH"] = {s: 100.0 * float(np.mean(c)) for s, c in
                       (("gu", caps_gu), ("dn", caps_dn)) if c is not None}
        n_exp = int(geom["n_exp"])
        rows, capsum = loadx_caps(z_all, caps_gu, caps_dn, n_exp)
        print("  hot experts (масса весов смеси | капа whitened gu/dn):",
              flush=True)
        for e, mpct, tk, cg, cd in rows:
            print("    exp %2d: %5.1f%% mass, %6d tok | %.1f%% / %.1f%%"
                  % (e, mpct, tk, cg, cd), flush=True)

        # ---- RAW CAP (опция): per-expert честная капа из дельт модели
        if have_source:
            try:
                rc = raw_caps_stream(args, i, ini, npz_dir, args.rank)
                r["RAW"] = {s: float(np.mean(rc[s])) * 100.0
                            for s in ("gu", "dn")}
                print("  RAW CAP (честный, среднее по экспертам): gu %.1f%% "
                      "dn %.1f%%" % (r["RAW"]["gu"], r["RAW"]["dn"]),
                      flush=True)
                if "WH" in r:
                    print("  whitened CAP: gu %.1f%% dn %.1f%%"
                          % (r["WH"]["gu"], r["WH"]["dn"]), flush=True)
            except Exception as e:                             # noqa: BLE001
                print("  RAW CAP не удался: %r" % e, flush=True)

        del Xf, tgt, z_all, Xi, Yi, mod
        results[i] = r
        fmt = lambda v: ("%.5g" % v) if v else "-"          # noqa: E731
        line = ("  block %d: INIT %s | CENTROID %s | FIT %s | POSTACT(init) %s"
                % (i, fmt(r["INIT"]), fmt(r["CENTROID"]), fmt(r["FIT"]),
                   fmt(r["POSTACT"])))
        if "POSTACT-FIT" in r:
            line += " | POSTACT-FIT %s" % fmt(r["POSTACT-FIT"])
        if "LONGFIT" in r:
            line += " | LONGFIT %s" % fmt(r["LONGFIT"])
        print(line + "\n  (INIT/CENTROID сравни с step-0/pure-centroid из "
              "лога stage-5; FIT - с финальным mse блока)", flush=True)

    # ------------------------------------------------------------- вердикты
    print("\n=== ВЕРДИКТЫ (по приоритету) ===", flush=True)
    v = []
    for i, r in results.items():
        if r["POSTACT"] < 0.75 * r["INIT"]:
            v.append("block %d: POSTACT на %.0f%% ниже INIT - кросс-члены "
                     "preact съедают четверть+ ошибки: перезапуск с "
                     "--field-mode postact (фит/refine/экспорт его знают)"
                     % (i, 100.0 * (1.0 - r["POSTACT"] / r["INIT"])))
        if r["FIT"] is not None and r.get("LONGFIT") is not None \
                and r["LONGFIT"] < 0.9 * r["FIT"]:
            v.append("block %d: LONGFIT %.5g << FIT %.5g - фит "
                     "недосходится: поднять --fit-steps (полировка ещё не "
                     "в полу)" % (i, r["LONGFIT"], r["FIT"]))
    for i, r in results.items():
        if r.get("RAW") and r.get("WH") and "dn" in r["WH"] \
                and r["RAW"]["dn"] < 0.7 * r["WH"]["dn"]:
            v.append("block %d: raw dn %.1f%% << whitened dn %.1f%% - "
                     "захват живёт в выбросах ковариации: пересобрать "
                     "банк 13.3 (--force) с --wh-damp-frac 0.01 "
                     "--wh-damp-base mean-diag"
                     % (i, r["RAW"]["dn"], r["WH"]["dn"]))
    for i, r in results.items():
        if r["CENTROID"] > 0 and abs(r["INIT"] / r["CENTROID"] - 1.0) < 0.05 \
                and r["FIT"] is not None and r["FIT"] < 0.9 * r["CENTROID"]:
            v.append("block %d: INIT ~ CENTROID, но FIT заметно ниже - инит "
                     "на шаге 0 не даёт функционального выигрыша, а фит его "
                     "находит: см. LOAD x CAP и RAW CAP выше (hot-эксперты / "
                     "mean-of-ratios / raw<<whitened)" % i)
    if not v:
        print("  явных патологий не найдено - смотри таблицу выше и "
              "UPDATE-13.4.md (раздел «что дальше»)", flush=True)
    for s in v:
        print("  * " + s, flush=True)


if __name__ == "__main__":
    main()
