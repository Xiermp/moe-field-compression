# -*- coding: utf-8 -*-
# version: 2026-09-08.1 - UPDATE-13 (whbanks): замена банка дельт в ГОТОВОМ
# артефакте без рефита и без requant бэкбона.
"""
bank_apply.py — подмена дельта-банка поля в собранном артефакте.

Зачем: быстрый путь "проверить и заменить". Артефакт после фита несёт
диагональные координаты C (n_exp, r) - его step-0 capture был слепым
(joint-v1 ~0.3%). bank_apply подменяет U/V/C на банки из whbank_build
(Tucker-2: общий отбелённый базис + ПЛОТНОЕ ядро k×k), обновляет
config.field (rank/core) и field_meta.json. Бэкбон, роутер, центроиды,
токенизатор НЕ трогаются (копируются как есть); перезаписываются только
те шафлы, где лежат тензоры поля.

Что делает замена с качеством: step-0 банк Tucker-2 k×k несёт capture
общего подпространства (смотри REPORT whbank_build: гейт 45%); фит поверх
замены (--bank-init t2 в пайплайне) поднимает качество дальше - это
основной путь, bank_apply - мгновенная проверка без фита.

Форматы:
  артефакт: HF-папка от write_field_artifact (config.json с cfg.field,
            model*.safetensors + index, field_meta.json). Поддерживаются
            артефакты banks=1 (diag ИЛИ уже dense).
  банк:     <whbank>/t2_k{K}/blk{i}_{gu|dn}.npz от whbank_build
            (U (m,K), V (n,K), cores (E,K,K), caps, meta).

ВАЖНО: whsvd_r* (per-expert базис) в общий базис контейнера НЕ выражается -
bank_apply принимает только t2_k* варианты; whsvd остаётся потолком качества
и диагностикой (см. UPDATE-13.md).

CLI:
  python bank_apply.py --artifact <dir> --whbank <fit_dir>/whbank \
      --variant t2_k32            # сухой прогон: только план
  python bank_apply.py --artifact <dir> --whbank <fit_dir>/whbank \
      --variant t2_k32 --apply
"""

import argparse
import json
import os
import shutil

import numpy as np


def load_plan(artifact):
    """Читает config.json + field_meta.json + art_meta-путь из пула."""
    with open(os.path.join(artifact, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    fi = cfg.get("field")
    if not fi:
        raise SystemExit("в config.json нет cfg.field - это не field-артефакт")
    if int(fi.get("banks", 1)) >= 2:
        raise SystemExit("bank_apply поддерживает только banks=1 артефакты "
                         "(bank-2payload формы не совпадают) - используй "
                         "пайплайн --bank-init t2 для полного рефита")
    meta_p = os.path.join(artifact, "field_meta.json")
    meta = {}
    if os.path.isfile(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
    return cfg, fi, meta


def shard_map(artifact):
    """{tensor_key: shard_filename} из index или заголовков шафлов."""
    from safetensors import safe_open
    idx_p = os.path.join(artifact, "model.safetensors.index.json")
    if os.path.isfile(idx_p):
        with open(idx_p, encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        return dict(wm)
    out = {}
    for fn in sorted(os.listdir(artifact)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(artifact, fn), framework="pt") as f:
                for k in f.keys():
                    out[k] = fn
    return out


def apply_variant(artifact, whbank, variant, apply, force=False):
    cfg, fi, meta = load_plan(artifact)
    r_old = int(fi["rank"])
    vdir = os.path.join(whbank, variant)
    if not os.path.isdir(vdir):
        raise SystemExit("нет %s - сначала whbank_build (--variants t2)" % vdir)
    blocks = sorted({int(fn.split("_")[0][3:]) for fn in os.listdir(vdir)
                     if fn.startswith("blk") and fn.endswith("_gu.npz")})
    if not blocks:
        raise SystemExit("в %s нет blk*_gu.npz" % vdir)
    print("артефакт: %s (rank %d, core %s, %d блоков из field_meta)"
          % (artifact, r_old, fi.get("core", "diag"), int(fi["n_layers"])))
    print("банк:     %s (%d блоков)" % (vdir, len(blocks)))
    n_layers = int(fi["n_layers"])
    if len(blocks) != n_layers:
        print("  WARNING: блоков в банке %d, в артефакте %d - заменяю "
              "первые %d" % (len(blocks), n_layers, min(len(blocks), n_layers)))
    # новый ранг = k варианта (из meta первого файла)
    z0 = np.load(os.path.join(vdir, sorted(
        fn for fn in os.listdir(vdir) if fn.endswith("_gu.npz"))[0]))
    k_new = int(z0["meta"][0])
    n_exp = int(z0["cores"].shape[0])
    if n_exp != int(fi["n_exp"]):
        raise SystemExit("n_exp банка %d != артефакта %d"
                         % (n_exp, int(fi["n_exp"])))

    wmap = shard_map(artifact)
    layers = (meta.get("block_names")
              or [f"model.layers.{j}.mlp" for j in range(n_layers)])
    FIELD = ("Ugu", "Vgu", "Cgu", "Udn", "Vdn", "Cdn")
    touched, plan = set(), []
    for i, layer in enumerate(layers):
        if i not in blocks:
            continue
        zg = np.load(os.path.join(vdir, "blk%02d_gu.npz" % i))
        zd = np.load(os.path.join(vdir, "blk%02d_dn.npz" % i))
        newt = {"Ugu": zg["U"][:, :k_new], "Vgu": zg["V"][:, :k_new],
                "Cgu": zg["cores"][:, :k_new, :k_new],
                "Udn": zd["U"][:, :k_new], "Vdn": zd["V"][:, :k_new],
                "Cdn": zd["cores"][:, :k_new, :k_new]}
        caps = (float(np.mean(zg["caps"].astype(np.float64))),
                float(np.mean(zd["caps"].astype(np.float64))))
        plan.append((i, layer, {n: t.shape for n, t in newt.items()}, caps))
        for nm in FIELD:
            key = f"{layer}.{nm}"
            if key in wmap:
                touched.add(wmap[key])
    cap_mean = float(np.mean([c for p in plan for c in p[3]]))
    print("замена:  U,V -> (%d колонок), C -> плотное ядро (%d, %d, %d); "
          "capture банка (whitened, среднее по блокам): gu/dn ~%.1f%%"
          % (k_new, n_exp, k_new, k_new, 100 * cap_mean))
    for i, layer, shapes, caps in plan[:3]:
        print("  blk%02d %s: %s  cap gu=%.1f%% dn=%.1f%%"
              % (i, layer, shapes, 100 * caps[0], 100 * caps[1]))
    print("  ... (%d блоков всего); перезаписываемые шафлы: %s"
          % (len(plan), ", ".join(sorted(touched)) or "-"))
    if not apply:
        print("\nDRY-RUN: ничего не записано. Добавь --apply для записи.")
        return
    if r_old != k_new and not force and meta:
        print("  WARNING: ранг меняется %d -> %d (старые iron/eval-кэши "
              "валидны - они не зависят от ранга поля)" % (r_old, k_new))

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    tdir = os.path.join(artifact, "_bank_apply_tmp")
    os.makedirs(tdir, exist_ok=True)
    for fn in sorted(touched):
        src_p = os.path.join(artifact, fn)
        dst_p = os.path.join(tdir, fn)
        shard = {}
        with safe_open(src_p, framework="pt") as f:
            for key in f.keys():
                shard[key] = f.get_tensor(key)
        for i, layer, _, _ in plan:
            zg = np.load(os.path.join(vdir, "blk%02d_gu.npz" % i))
            zd = np.load(os.path.join(vdir, "blk%02d_dn.npz" % i))
            newt = {"Ugu": zg["U"][:, :k_new], "Vgu": zg["V"][:, :k_new],
                    "Cgu": zg["cores"][:, :k_new, :k_new],
                    "Udn": zd["U"][:, :k_new], "Vdn": zd["V"][:, :k_new],
                    "Cdn": zd["cores"][:, :k_new, :k_new]}
            for nm, t in newt.items():
                key = f"{layer}.{nm}"
                if wmap.get(key) == fn:
                    shard[key] = torch.from_numpy(
                        np.ascontiguousarray(t, dtype=np.float32))
        tmp = dst_p + ".tmp"
        save_file(shard, tmp, metadata={"format": "pt"})
        os.replace(tmp, dst_p)
        print("  переписан %s" % fn)
    for fn in sorted(touched):
        os.replace(os.path.join(tdir, fn), os.path.join(artifact, fn))
    shutil.rmtree(tdir, ignore_errors=True)

    # config.field + field_meta: новый ранг/ядро
    fi["rank"] = int(k_new)
    fi["core"] = "dense"
    with open(os.path.join(artifact, "config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)
    meta.update(rank=int(k_new), core="dense")
    meta["bank_apply"] = dict(variant=variant, whbank=os.path.abspath(whbank),
                              blocks=len(plan),
                              capture_whitened_mean=cap_mean)
    with open(os.path.join(artifact, "field_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nГОТОВО: артефакт переключён на %s (rank %d, core=dense). "
          "Проверка: та же iron/eval-команда, что и раньше; откат - "
          "пересобери артефакт пайплайном или держи копию папки." % (variant, k_new))


def main():
    p = argparse.ArgumentParser(
        description="bank_apply: замена банка дельт в артефакте (UPDATE-13)")
    p.add_argument("--artifact", required=True, help="папка field-артефакта")
    p.add_argument("--whbank", required=True,
                   help="каталог банков (fit_r<rank>t2/whbank или <pool>/whbank)")
    p.add_argument("--variant", required=True,
                   help="t2_k{K} - только Tucker-2 (whsvd в контейнер не "
                        "помещается)")
    p.add_argument("--apply", action="store_true",
                   help="записать изменения (без флага - сухой прогон)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not args.variant.startswith("t2_k"):
        raise SystemExit("bank_apply принимает только t2_k* варианты; "
                         "whsvd_r* - per-expert базис, в общий контейнер "
                         "не выражается")
    apply_variant(args.artifact, args.whbank, args.variant,
                  apply=args.apply, force=args.force)


if __name__ == "__main__":
    main()
