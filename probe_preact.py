#!/usr/bin/env python3
# version: 2026-09-07 (update 11.2) - PROBE C: preact cross-noise, OFFLINE.
"""Измеряет кросс-шум preact-композиции артефакта БЕЗ единого прогона модели.

Зачем: IRON TEST показал, что остаточная KL живёт в самих дельтах (H2), но не
умеет отличить "ранга не хватает" от "композиция портит". В режиме preact
смесь координат c(z)=z@C подставляется ДО нелинейности, и SwiGLU рождает
кросс-члены z1*z2*(g1*u2 + g2*u1), которые фит компенсирует лишь частично -
это и есть "оглушённость кросс-шумом SwiGLU". В режиме postact нелинейность
применяется ПОЭКСПЕРТНО и кросс-члены равны нулю точно.

Метод: для одних и тех же входов x из СОХРАНЁННОГО пула пар (pairs_blk{i}.pt)
артефакт считается в обеих композициях (переключением field_mode - параметры
идентичны), и сравнивается:
    rel_pre   = RMS(y_preact - y_base) / RMS(y_base)
    rel_post  = RMS(y_postact - y_base) / RMS(y_base)
    comp_gap  = RMS(y_preact - y_postact) / RMS(y_base)   # чистый кросс-шум
где y_base - выход базового блока из пула (цель фита). Если comp_gap >= ~50%
rel_pre - композиция доминирует (починка: --field-mode postact или рефит),
если comp_gap мал при больших rel_* - это ёмкость ранга (H2, банк/ранг).

Офлайн: ни base, ни GGUF не нужны - только папка артефакта и папка пула
(cache_{tag} с pairs_blk{i}.pt). Работает и на CPU.

Пример:
    python3 probe_preact.py --src field_MyModel_r128 --pool cache_MyModel
    python3 probe_preact.py --src ... --pool ... --topk 8  # кросс-члены растут с k
"""
import argparse
import os
import sys
import time

import hf_env  # noqa: F401 - HF cache inside the project; BEFORE transformers
import torch


def rms_rel(a, b):
    """RMS(a - b) / RMS(b) - агрегированная относительная ошибка."""
    num = (a.float() - b.float()).pow(2).mean()
    den = b.float().pow(2).mean().clamp_min(1e-12)
    return float((num / den).sqrt())


def main():
    ap = argparse.ArgumentParser(
        description="PROBE C: preact vs postact composition gap of a field "
                    "artifact on the saved pair pool (offline, no model runs)")
    ap.add_argument("--src", required=True,
                    help="artifact dir (the field_*_rN folder written by the "
                         "pipeline)")
    ap.add_argument("--pool", required=True,
                    help="calibration pool dir (cache_{tag} with "
                         "pairs_blk{i}.pt)")
    ap.add_argument("--blocks", default="", metavar="0,1,2",
                    help="block indices to probe (default: every block that "
                         "has a pool file)")
    ap.add_argument("--n", type=int, default=4096,
                    help="rows sampled from each block's pool (default 4096)")
    ap.add_argument("--batch", type=int, default=2048,
                    help="rows per forward chunk (RAM bound)")
    ap.add_argument("--topk", type=int, default=0, metavar="K",
                    help="override the field routing k (0 = keep the "
                         "artifact's own; cross-terms grow with k)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"no artifact dir: {args.src}")
    if not os.path.isdir(args.pool):
        sys.exit(f"no pool dir: {args.pool}")

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"loading the artifact: {args.src} ({args.dtype})...", flush=True)
    t0 = time.time()
    art = AutoModelForCausalLM.from_pretrained(
        args.src, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True).eval()
    dev = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        art = art.to("cuda")
    print(f"  loaded on {dev} in {time.time() - t0:.0f} s", flush=True)

    # field modules in MODEL order (== the pool's collect_pairs order)
    mods = [m for _, m in art.named_modules()
            if hasattr(m, "field_mode") and hasattr(m, "wgud")]
    if not mods:
        sys.exit("no field modules found (hasattr field_mode+wgud) - is this "
                 "a field artifact?")
    print(f"field modules: {len(mods)}", flush=True)
    if args.topk:
        for m in mods:
            m.top_k = int(args.topk)
        print(f"routing k overridden on all field modules: "
              f"top_k={args.topk}", flush=True)

    n_blocks = len(mods)
    if args.blocks:
        want = [int(x) for x in str(args.blocks).split(",") if x.strip()]
    else:
        want = [i for i in range(n_blocks)
                if os.path.isfile(os.path.join(args.pool,
                                               f"pairs_blk{i}.pt"))]
    if not want:
        sys.exit(f"no pairs_blk{{i}}.pt in {args.pool} - nothing to probe")
    missing = [i for i in want
               if i >= n_blocks
               or not os.path.isfile(os.path.join(args.pool,
                                                  f"pairs_blk{i}.pt"))]
    if missing:
        print(f"WARNING: no pool files / no module for blocks {missing[:6]} "
              f"- skipped", flush=True)
        want = [i for i in want if i not in missing]
    if not want:
        sys.exit("nothing to probe after the pool/module match")

    g = torch.Generator().manual_seed(args.seed)
    print(flush=True)
    print(f"{'blk':>4} | {'rel_pre':>8} | {'rel_post':>8} | {'comp_gap':>8} "
          f"| share of the preact error", flush=True)
    agg = []
    for i in want:
        d = torch.load(os.path.join(args.pool, f"pairs_blk{i}.pt"),
                       map_location="cpu")
        X, Y = d["X"], d["Y"]
        n = min(int(args.n), int(X.shape[0]))
        idx = torch.randperm(int(X.shape[0]), generator=g)[:n]
        m = mods[i]
        mode0 = str(m.field_mode)
        sum_pre = sum_post = sum_comp = sum_den = 0.0
        own_drift = 0.0     # runtime output in its OWN mode vs the matching
        with torch.no_grad():   # recomputed composition: must be ~0
            for lo in range(0, n, args.batch):
                sl = idx[lo:lo + args.batch]
                xb = X[sl].unsqueeze(0).to(dev)      # [1, B, d]
                yb = Y[sl].unsqueeze(0).to(dev)      # [1, B, d] base target
                m.field_mode = "preact"
                y_pre = m.forward(xb)
                m.field_mode = "postact"
                y_post = m.forward(xb)
                m.field_mode = mode0
                y_own = m.forward(xb)
                ref = y_pre if mode0 == "preact" else y_post
                own_drift = max(own_drift, rms_rel(y_own, ref))
                sum_pre += float((y_pre.float() - yb.float()).pow(2).sum())
                sum_post += float((y_post.float() - yb.float()).pow(2).sum())
                sum_comp += float((y_pre.float() - y_post.float())
                                  .pow(2).sum())
                sum_den += float(yb.float().pow(2).sum())
                del y_pre, y_post, y_own, xb, yb
        den = max(sum_den, 1e-12)
        rel_pre = (sum_pre / den) ** 0.5
        rel_post = (sum_post / den) ** 0.5
        comp = (sum_comp / den) ** 0.5
        share = 100 * comp / max(rel_pre, 1e-9)
        if own_drift > 1e-3:
            print(f"    (WARNING blk{i}: runtime own-mode output deviates "
                  f"from the recomputed composition: {own_drift:.4f} - the "
                  f"probe numbers for this block are unreliable)", flush=True)
        verdict = ("COMPOSITION dominates (postact refit is the fix)"
                   if share >= 50.0 else
                   "capacity dominates (H2: more rank / bank)")
        print(f"{i:>4} | {rel_pre:8.4f} | {rel_post:8.4f} | {comp:8.4f} "
              f"| {share:5.0f}%  -> {verdict}", flush=True)
        agg.append((i, rel_pre, rel_post, comp, share))
        del d, X, Y

    if agg:
        mean_pre = sum(a[1] for a in agg) / len(agg)
        mean_post = sum(a[2] for a in agg) / len(agg)
        mean_gap = sum(a[3] for a in agg) / len(agg)
        mean_share = 100 * mean_gap / max(mean_pre, 1e-9)
        print(flush=True)
        print(f"MEAN over {len(agg)} blocks: rel_pre {mean_pre:.4f} | "
              f"rel_post {mean_post:.4f} | comp_gap {mean_gap:.4f} "
              f"({mean_share:.0f}% of the preact error)", flush=True)
        if mean_share >= 50.0:
            print("verdict: preact cross-noise is a major term -> rebuild/"
                  "refit with --field-mode postact (params identical, zero "
                  "cross-term error, ~1 extra down-GEMM per token)", flush=True)
        else:
            print("verdict: composition is NOT the bottleneck here -> the "
                  "iron-test residual is capacity (H2): more rank / bank2",
                  flush=True)


if __name__ == "__main__":
    main()
