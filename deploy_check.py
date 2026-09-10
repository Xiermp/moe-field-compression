#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy_check.py — на каком слое живёт ошибка: в банке/фите или в пути
«fit-модуль -> артефакт»?

Сравнивает на ОДНИХ И ТЕХ ЖЕ парах кэша пула (pairs_blk{i}.pt, base-модель не
нужна, полный артефакт не грузится — только шафлы с тензорами поля):

  mse_fit  — fit-модуль (hf_field_transform.FieldSparseMoe) с параметрами из
             fit_blk{i}.pt: то, что решил фит (в 13.1 это может быть и
             pristine-инит — см. "best state restored");
  mse_art  — те же пары через тензоры, РЕАЛЬНО записанные в артефакт
             (safetensors, bf16, как деплоится);
  таблица верности тензоров: shape / max|Δ| / rel err против fit_blk
             (bf16-округление даёт rel ~0.4% на элемент; сильно больше —
             манглинг на записи: масштаб, транспон, чужой тензор).
             13.3: рядом печатается ВЕЛИЧИНА тензора max|t| — НЕ путать:
             max|Δ| это разница фит-vs-артефакт (bf16-шум), а не масштаб;
             rel 0.39% = теоретический потолок bf16 (2^-8), а max|t| —
             реальный масштаб (проверка обвинений вида «Cdn равен нулю»).

  13.3: mse печатается с rel-rms (sqrt(mse/E[y²])) — абсолютные mse
             НЕ сравнимы между блоками (масштаб сигнала растёт с глубиной);
             0.297 при rel-rms 2% — это «нормально», 0.297 при 30% — беда.

Интерпретация:
  тензоры OK и mse_art <= 1.15 * mse_fit -> запись верна; качество артефакта =
      качество инита/фита самого по себе (вопрос вместимости линзы: ранг/
      спайки, см. whbank_build --variants spk);
  mse_art >> mse_fit (и тензоры OK)      -> расхождение математики рантайма
      (шаблон modeling_field) с fit-математикой — следующий подозреваемый;
  тензор FAIL                            -> найден конкретный баг записи.

Сравнение с_WHITENED-capture банка связывает три мира: capture (файлы банка),
mse_fit (функционально, фит-модуль), mse_art (функционально, деплой-веса).

Запуск:
  python deploy_check.py --pool results/cache_<...> \
      --fit-dir results/cache_<...>/fit_r64t2 \
      --artifact field_<tag>_r64 --blocks 0-3
Выход: 0 = деплой верен, 1 = найдено расхождение (или --strict на warning).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def parse_blocks(s):
    s = (s or "").strip().lower()
    if s in ("", "all"):
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def artifact_keys(art):
    """{key: shard_path} по всем safetensors артефакта (index или скан)."""
    from safetensors import safe_open
    keys = {}
    idx = os.path.join(art, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        for k, shard in wm.items():
            keys[k] = os.path.join(art, shard)
    else:
        for shard in sorted(glob.glob(os.path.join(art, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    keys[k] = shard
    if not keys:
        raise SystemExit("нет safetensors-шафлов в %s" % art)
    return keys


def load_tensor(keys, name):
    from safetensors import safe_open
    p = keys.get(name)
    if p is None:
        return None
    with safe_open(p, framework="pt", device="cpu") as f:
        if name not in f.keys():
            return None
        return f.get_tensor(name).float()


def pool_mse(mod, X, Y, chunk=2048):
    """Полный пул блока через модуль (fp32), чанками. fit-модуль сам строит
    роутинг (замороженный gw) — как в фите."""
    import torch
    import torch.nn.functional as F
    tot, n = 0.0, 0
    with torch.no_grad():
        for a in range(0, max(X.shape[0], 1), chunk):
            xb = X[a:a + chunk].float().view(1, -1, X.shape[-1])
            yb = Y[a:a + chunk].float().view(1, -1, Y.shape[-1])
            out = mod.forward(xb).float()
            tot += F.mse_loss(out, yb, reduction="sum").item()
            n += out.numel()
    return tot / max(n, 1)


def tensor_row(name, ref, art, rtol, atol_frac):
    """Строка сравнения одного тензора (ref из fit_blk, art из артефакта)."""
    import torch
    if art is None:
        return name, False, "ОТСУТСТВУЕТ в артефакте"
    if tuple(art.shape) != tuple(ref.shape):
        return name, False, "shape %s != %s" % (tuple(art.shape),
                                                tuple(ref.shape))
    d = (art - ref).abs()
    scale = float(ref.abs().max()) if ref.numel() else 0.0
    atol = atol_frac * max(scale, 1e-12)
    viol = float((d - rtol * ref.abs()).max()) if ref.numel() else 0.0
    ok = viol <= atol
    rel = float((d / ref.abs().clamp_min(1e-6)).max()) if ref.numel() else 0.0
    return name, ok, ("max|d| %.3e | max|t| %.3e | rel %.2f%% -> %s"
                      % (float(d.max()), scale, 100 * rel,
                         "OK" if ok else "FAIL (rtol %.0f%% + atol %.1e)"
                         % (100 * rtol, atol)))


def main():
    ap = argparse.ArgumentParser(
        description="deploy_check: фит vs записанный артефакт на пуле")
    ap.add_argument("--pool", required=True,
                    help="кэш пайплайна (pairs_blk*.pt + init_blk*.pt)")
    ap.add_argument("--fit-dir", required=True,
                    help="каталог фит-файлов (например fit_r64t2: "
                         "fit_blk*.pt, mse.json)")
    ap.add_argument("--artifact", required=True, help="каталог field-артефакта")
    ap.add_argument("--blocks", default="0-2")
    ap.add_argument("--rank", type=int, default=None,
                    help="ранг поля (по умолчанию из config.json артефакта)")
    ap.add_argument("--rtol", type=float, default=0.02,
                    help="допуск на элемент (bf16 ~0.4%%, дефолт 2%%)")
    ap.add_argument("--atol-frac", type=float, default=1e-4,
                    help="абсолютный допуск как доля max|tensor|")
    ap.add_argument("--strict", action="store_true",
                    help="warning-расхождения тоже считать провалом")
    ap.add_argument("--no-mse", action="store_true",
                    help="только таблица тензоров (быстро)")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hf_field_transform import FieldSparseMoe, apply_core, load_pairs_block

    with open(os.path.join(args.artifact, "config.json"),
              encoding="utf-8") as f:
        fld = (json.load(f).get("field") or {})
    rank = args.rank or int(fld.get("rank", 0))
    if not rank:
        raise SystemExit("не знаю rank (ни --rank, ни config.json:field)")
    mode = fld.get("field_mode", "preact")
    core = fld.get("core", "diag")

    keys = artifact_keys(args.artifact)
    layer_ids = sorted({int(k.split(".layers.")[1].split(".")[0])
                        for k in keys if k.endswith(".mlp.Ugu")})
    if not layer_ids:
        raise SystemExit("в артефакте нет *.mlp.Ugu — это field-артефакт?")
    print("артефакт: rank %d, core %s, mode %s; field-слои: %s"
          % (rank, core, mode, layer_ids), flush=True)

    try:
        with open(os.path.join(args.fit_dir, "mse.json"), encoding="utf-8") as f:
            fit_reported = json.load(f)
    except Exception:
        fit_reported = []

    blocks = parse_blocks(args.blocks)
    if blocks is None:
        blocks = list(range(len(layer_ids)))
    bad = 0
    warn = 0
    for i in blocks:
        if i >= len(layer_ids):
            print("block %d: нет слоя в артефакте - пропуск" % i)
            continue
        L = layer_ids[i]
        ini = torch.load(os.path.join(args.pool, "init_blk%d.pt" % i),
                         map_location="cpu")
        fit = torch.load(os.path.join(args.fit_dir, "fit_blk%d.pt" % i),
                         map_location="cpu")
        X, Y = load_pairs_block(os.path.join(args.pool, "pairs_blk%d.pt" % i))
        geom = apply_core(dict(ini["geom"]), core == "dense") \
            | {"field_mode": mode}
        act = torch.nn.functional.silu

        def build(params, gate_w):
            return FieldSparseMoe(geom, rank, gate_w=gate_w, act_fn=act,
                                  gate_bias=ini.get("eb"),
                                  shared=ini.get("shared"),
                                  banks=int(geom.get("banks", 1)),
                                  init=params)

        # --- что решил фит (fit_blk; gw = gw_tuned | исходный) -------------
        ref = {n: fit[n].float() for n in fit if n != "gw_tuned"}
        ref_gw = (fit["gw_tuned"].float() if "gw_tuned" in fit
                  else ini["gw"].float())
        mse_fit = (None if args.no_mse
                   else pool_mse(build(ref, ref_gw), X, Y))

        # --- что реально записано в артефакт --------------------------------
        prefix = next(k for k in keys if k.endswith(".mlp.Ugu")
                      and f".layers.{L}." in k)
        prefix = prefix[: -len("Ugu")]
        art, rows = {}, []
        names = sorted(set(list(ref) + ["gw"]))
        if ini.get("eb") is not None:
            names.append("e_score_correction_bias")
        if ini.get("shared"):
            names += ["sh_gu", "sh_dn"]
        for n in names:
            aname = prefix + ("gate.weight" if n == "gw" else n)
            t = load_tensor(keys, aname)
            art[n] = t
            r = ref.get(n)
            if r is None:                      # буфер (bias/shared) из ini
                r = {"gw": ref_gw}.get(n)
                if r is None and n == "e_score_correction_bias":
                    r = ini["eb"].float()
                elif r is None and n in ("sh_gu", "sh_dn") \
                        and isinstance(ini.get("shared"), (list, tuple)):
                    r = ini["shared"][0 if n == "sh_gu" else 1].float()
            if r is not None:
                nm, ok, msg = tensor_row(n, r, t, args.rtol, args.atol_frac)
                rows.append("  %-26s %s" % (nm, msg))
                if not ok:
                    bad += 1
        art_gw = art.get("gw") if art.get("gw") is not None else ref_gw
        mse_art = (None if args.no_mse
                   else pool_mse(build({**ref, **{k: v for k, v in art.items()
                                                  if v is not None}},
                                     art_gw), X, Y))
        # пол осмысленности: отношение двух почти-нулевых mse - шум; тревога
        # только когда ошибка артефакта измерима на масштабе сигнала
        y_energy = float((Y.float() ** 2).mean()) if Y.numel() else 0.0
        mse_floor = max(1e-8, 1e-4 * y_energy)

        rep = (fit_reported[i] if isinstance(fit_reported, list)
               and i < len(fit_reported) else None)
        print("block %d (layer %d):" % (i, L), flush=True)
        for ln in rows:
            print(ln, flush=True)
        if mse_fit is not None:
            ratio = mse_art / max(mse_fit, 1e-30)
            ok_m = ratio <= 1.15 or mse_art <= mse_floor
            if not ok_m:
                bad += 1
            # 13.3: rel-rms = sqrt(mse/E[y^2]) — честный масштаб ошибки
            # относительно сигнала; абсолютные mse между блоками несравнимы
            def _rms(m):
                return 100.0 * (m / max(y_energy, 1e-30)) ** 0.5
            print("  mse: фит %s (%.2f%% rms) | репортед %s | артефакт %s "
                  "(%.2f%% rms) -> ratio %.3f %s   [E[y^2] = %.4g]"
                  % (("%.6g" % mse_fit), _rms(mse_fit),
                     ("%.6g" % rep) if rep is not None else "-",
                     "%.6g" % mse_art, _rms(mse_art), ratio,
                     "OK" if ok_m else "FAIL (>1.15x - расхождение "
                     "запись/рантайм)", y_energy), flush=True)
        print(flush=True)

    if bad or warn:
        print("DEPLOY CHECK: FAIL (%d расхождений) - см. строки выше"
              % bad, flush=True)
        sys.exit(1)
    print("DEPLOY CHECK: OK - запись и функциональность артефакта совпадают "
          "с фит-модулем; качество = инит/фит сам по себе", flush=True)


if __name__ == "__main__":
    main()
