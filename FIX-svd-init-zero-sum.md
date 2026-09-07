# FIX: zero-sum SVD init (expert_basis_init pass 1) — build 10.9 hotfix

## TL;DR

`expert_basis_init` in `hf_field_transform.py` sketches the expert deltas with
**one shared random Omega** and **sums** the per-expert sketches:

```
Y = sum_e (dW_e - mean) @ Omega  =  (sum_e dW_e) @ Omega  =  0 @ Omega
```

Because `mean` is the exact mean over the same experts, `sum_e dW_e == 0`
identically. `Y` is fp32 rounding noise, `qr(Y)` returns an arbitrary garbage
basis, and the whole SVD init is blind. This is the direct cause of the
"delta energy captured at step 0 - gu 0.3%, dn 0.3%" line in your logs.

The function's own docstring even says: *"sum_e dW_e = 0 identically; the
stacked-delta shared basis is the meaningful variant"* — but the code sums
the sketches instead of stacking them.

**Consequences:** every run so far started the fit from a noise basis. The
"flat spectrum / H2 capacity wall" conclusion is WITHDRAWN — it was measured
through a garbage basis. Re-fit after the fix before believing any capacity
verdict.

## Proof (synthetic experts sharing a known rank-8 basis, 10% noise)

| init | capture (diag) | capture_proj (subspace) |
|---|---|---|
| OLD (shared Omega) | 11.1% (blind; on the real model it landed at 0.3%) | — |
| NEW (per-expert Omega) | **83.4% / 68.6%** (gu/dn) | **86.3%** |
| Oracle (exact generating basis) | 88.1% | — |
| NEW on unstructured deltas | 1.1% (honest ceiling) | 6.2% |

Test: `scripts/test_svd_init_fix.py` (runs the REAL patched function).

## The patch (surgical, ~10 lines)

In `hf_field_transform.py`, inside `expert_basis_init`, find the pass-1 block
(search for `Ygu += dgu @ Om_gu`).

**BEFORE:**

```python
    Om_gu = torch.randn(in_dim_gu, q_gu, generator=g) / math.sqrt(q_gu)
    Om_dn = torch.randn(in_dim_dn, q_dn, generator=g) / math.sqrt(q_dn)
    Ygu = torch.zeros(out_dim_gu, q_gu)
    Ydn = torch.zeros(out_dim_dn, q_dn)
    den = {"gu": 0.0, "dn": 0.0}       # sum_e ||dW_e||^2 (true capture denum)
    n_exp = 0
    for wgu, wdn in _iter_expert_w(block):
        dgu, ddn = wgu - mgu, wdn - mdn
        Ygu += dgu @ Om_gu
        Ydn += ddn @ Om_dn
        den["gu"] += float(dgu.norm() ** 2)
        den["dn"] += float(ddn.norm() ** 2)
        n_exp += 1
```

**AFTER:**

```python
    # Per-expert independent Omega: this sketches the HORIZONTAL stack
    # [dW_1 ... dW_N]. A single shared Omega zeroes out here:
    # sum_e (dW_e - mean) == 0 identically, so Y would be pure fp32
    # rounding noise and QR would return an arbitrary garbage basis.
    # With independent Omega_e: E[Y Y^T] = sum_e dW_e dW_e^T - exactly the
    # summed output-side Gram the pass-2 eigh wants. Same shapes, same cost.
    Ygu = torch.zeros(out_dim_gu, q_gu)
    Ydn = torch.zeros(out_dim_dn, q_dn)
    den = {"gu": 0.0, "dn": 0.0}       # sum_e ||dW_e||^2 (true capture denum)
    n_exp = 0
    for ei, (wgu, wdn) in enumerate(_iter_expert_w(block)):
        dgu, ddn = wgu - mgu, wdn - mdn
        ge = torch.Generator().manual_seed(seed + 1000 * ei)
        Ygu += dgu @ (torch.randn(in_dim_gu, q_gu, generator=ge) / math.sqrt(q_gu))
        Ydn += ddn @ (torch.randn(in_dim_dn, q_dn, generator=ge) / math.sqrt(q_dn))
        den["gu"] += float(dgu.norm() ** 2)
        den["dn"] += float(ddn.norm() ** 2)
        n_exp += 1
```

The two `Om_gu/Om_dn` lines BEFORE the loop are deleted. Nothing else in the
function changes: `qr`, pass 2, the eigh calls and the `C_e = diag(...)`
coordinates keep working as before — now on a real basis. Math:
`E[Y Y^T] = sum_e dW_e dW_e^T`, the exact object the downstream eigh needs.

## Cache invalidation (required — old inits AND fits are poisoned)

1. Bump the init version stamp so the self-heal rebuilds everything:
   in `hf_pipeline.py` find the SVD version stamp (search `svd_ver` /
   `"v0"` near `_svd_file_ver`) and change the string, e.g. `"v0"` ->
   `"v1-zerosum-fix"`.
2. Belt and suspenders — delete the poisoned fit folder (fits were trained
   from the noise basis; they must be redone anyway):

```bat
rmdir /s /q "D:\all\codes\expert shinkenator\results\cache_NanoColibri-Instruct-GGUF\fit_r128"
```

The 25 GB pool (pairs / log-probs / centroids) is NOT touched — it is
rank- and init-independent.

## Re-run (rank 128 first — one variable at a time)

Same command as your last run, keep `--verify-topk 1`:

```bash
python hf_pipeline.py --model mradermacher/NanoColibri-Instruct-GGUF --gguf-quant Q8_0 \
  --rank 128 --per-layer-cap 65536 --calib-windows 32 --fit-steps 600 \
  --fit-method muon-cosine --fit-workers 2 --fit-jitter 0.0 --fit-early-stop 50 \
  --io-cache disk --profile high --muon-ns-steps 5 --fit-autocast auto \
  --refine-rounds 2 --verify-topk 1
```

## How to read the new init line (the decisive diagnostic)

Stage 4/5 will now print e.g. `svd init: delta energy captured at step 0 - gu X%, dn Y%`,
plus the init dict carries `capture_proj_*` (subspace capture without the
diagonal loss):

| outcome | meaning | next move |
|---|---|---|
| capture jumps to 30-70%+ | the fit was crippled by the blind init; expect much lower per-block mse and KL | re-run the IRON TEST on the new artifact |
| capture stays low, but capture_proj is high | basis is fine; the **diagonal** per-expert coordinates are the bottleneck | Tucker-2 core `G_e` (r x r) instead of `diag(C_e)` — the next structural fix |
| both stay low | deltas genuinely do not share a low-rank subspace | per-group bases (cluster experts, own U/V per group) |

Run the `probe_capacity.py` arms AFTER the refit, so they load the fresh fit
files (they anchor on the shipped fit, which was poisoned until now).
