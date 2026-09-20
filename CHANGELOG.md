# CHANGELOG - expert-press / field engine

Moved out of the hf_pipeline.py header at 13.6 (2026-09-10.2, CLI-CLARITY):
the notes below are verbatim, newest first. The newest version notes stay
in the hf_pipeline.py header. Full per-update write-ups: UPDATE-*.md.

## version: 2026-09-13.3 - 13.7.1: (a) --bank-init default flips svd ->
    dense (the user-facing goal is maximal zero-shot quality with zero
    tuning; dense won EVERY zero-shot A/B to date: toy TinyMoE, the
    micro-real MoE google/switch-base-8, and the third geometry
    HuggingFaceTB/nanowhale-100m-base - lower weight rel-MSE on all 8
    layers x ranks 8/16/32 in the Task-47 zero-shot bench; --bank-init svd
    stays as the bit-identical 13.6.x compat mode). (b) ROBUST-HF-LOAD:
    load_source_model now loads SOURCE models through load_hf_model_robust
    (hf_field_transform) instead of the from_pretrained fast path - the
    fast path materializes NON-PERSISTENT buffers (persistent=False, e.g.
    the freqs_cis rope table of deepseek-v4-style custom-code models) from
    UNINITIALIZED memory: sporadic per-process NaN logits and
    nondeterministic forwards that look like a broken checkpoint (the
    nanowhale card even misdiagnosed this as "bf16 NaN"; neither
    low_cpu_mem_usage=False nor _fast_init=False cures it - measured).
    The robust path instantiates the class directly via from_config (the
    __init__-computed buffer values survive), loads the snapshot shards
    strict (tied keys allowed), ties weights and NaN-gates every
    non-persistent buffer. Verified on nanowhale-100m-base: from_pretrained
    = garbage buffers in ~half the processes; robust load = finite buffers
    + stable ppl 19.58 (3/3 processes). New test test_update1371_loadfix
    (robust round-trip, dtype, the gate, the CLI default).

## version: 2026-09-13.2 - EXPERT-MEANS-FIX: expert_means() on the ModuleList
    expert layout (per-expert w1/w3/w2 modules, e.g.
    HuggingFaceTB/nanowhale-100m-base) accumulated .sum(0): the "centroids"
    came out as vectors (d,)/(dff,) instead of matrices (2dff,d)/(d,dff),
    so the svd init crashed or mis-shaped on every ModuleList-layout model
    (the batched gate_up_proj/down_proj path was unaffected). The full
    matrices are accumulated now; OLMoE-style runs are bit-identical and
    caches stay valid. Found by the zero-shot init bench on nanowhale; the
    same bench also surfaced a transformers-5.x fast-init hazard for
    custom-code models: NON-PERSISTENT buffers (freqs_cis) materialize from
    uninitialized memory (504 NaN entries -> NaN logits) - recompute such
    buffers after from_pretrained or load non-fast.

## version: 2026-09-13.1 - DENSE-INIT (13.7): --bank-init dense - the FULL
    core C_e = U^T dW_e V (E, r, r) on the SAME joint-v2 basis as the
    legacy diag init (expert_basis_init core="dense"; data-free, straight
    from the projected-delta stash, zero extra streaming cost). The
    zero-shot audits (toy TinyMoE + the micro-REAL MoE google/switch-base-8
    with real routed activations) showed the DIAGONAL was the zero-shot
    bottleneck: the dense core doubles the captured energy on the same
    subspace, cuts the routed-activation error by -6..-11% (up side) /
    -38..-54% (down side) at step 0, and wins AFTER the fit too (equal
    budget: final KL 0.010 vs 0.020 at r=32). Plumbing: the new
    svd_init_ver() helper is the single stamp authority ("joint-v2-dense"
    [+b2]; the default diag namespace "joint-v2" stays bit-identical to
    13.6.x so old caches remain valid), the geom carries core="dense" from
    stage 4 on, the fit lives in its own fit_r<rank>dense, the artifact
    uses the Tucker-2 dense core the container has had since UPDATE-13
    (bank 2 stays diagonal), field_meta profile stamps bank_init. The t2
    polish profile is deliberately NOT applied to dense (the init is not
    converged - the default full-fit profile is the right one). Tests:
    test_update137_dense.py 17/17; 13.6.4 fit-fresh 12/12, CLI 21/21,
    13.5 du, 13.4, 13.3, 13.2, 13.1, 13.6.3 rangefix - green.

## version: 2026-09-11.1b - FIT-FRESH (13.6.4): --fit-fresh - the one-flag
    "refit from scratch": stage 5 (fit) ignores fit_meta.json /
    fit_partial.json / fit_blk*.pt and refits ALL blocks with the current
    settings. The per-block resume stays the DEFAULT (a killed fit of
    hundreds of steps/block never restarts from block 0); the flag is for
    the deliberate clean refit under the same settings. The calibration
    pool and init_svd_blk* caches are untouched. Both fit-cache notices now
    advertise the flag ("pass --fit-fresh ..."), so no more manual fit_r*
    deletion. hf_cli gains the flag (group "optional - fit: guard & resume",
    full description in --list-flags); canonical_command emits it when set.
    Tests: test_update1364_fitfresh.py 12/12 (parser, canonical round-trip,
    source anchors incl. the unconditional partial_p definition, torch-free
    --list-flags, functional decision-chain replica); CLI 21/21, 13.5 du,
    13.1, 13.2, 13.6.3 rangefix - green.
## version: 2026-09-11.1 - SVD-INIT RANGE FIX (13.6.3): expert_basis_init's
    pass-1 range finder accumulated Y = sum_e dW_e @ Omega = (sum_e dW_e) @
    Omega - IDENTICALLY ZERO at the mean-centered deltas (expert_means is
    the exact arithmetic mean, so sum_e dW_e == 0 by construction): Q was
    the QR of fp32 noise, and the whole init (U/V/C, bank-2 residual, the
    capture banner) lived on a garbage subspace. The toy: diag capture
    0.08-0.11 (noise projection) instead of the achievable 0.45-0.7; step-0
    mse 1.7-3x the fixed init. FIX: second-moment range finder
    Y = sum_e dW_e dW_e^T @ Omega (never degenerate). SVD_INIT_VER
    joint-v1 -> joint-v2: stale cached init files rebuild automatically.
    Toy verified: subspace capture 1.00 (was ~0.10), diag capture
    gu 0.082->0.447 / dn 0.109->0.482, step-0 mse -41% (balanced) / -66%
    (outlier deltas x4); centroid re-test under the fixed init: the
    arithmetic mean stays the best anchor.
## version: 2026-09-08.4 - WH-CHECK (13.3): --fit-train gains "cores" (the
    stricter polish: U*/V* AND centroids w* frozen, only the cores train -
    the external "freeze W0" recipe; A/B vs "core" decides the default);
    the SVD-init banner also prints the HONEST (plain-Frobenius) capture
    when the bank file carries caps_raw (whbank_build 13.3).
## version: 2026-09-08.3 - T2-POLISH (13.1): two fixes for the bank-init fit
    explosion (2026-09-08 run: every block diverged at step ~100, best-state
    re-eval ~1000x the step-0 mse, artifact KL 2.288 / +322%). (1) FIT-LOOP
    OFF-BY-ONE: the best-state snapshot was taken AFTER opt.step() while its
    score was the PRE-step loss - with an init that is already optimal at
    step 0 the "best state" was always "init + one optimizer kick" and the
    divergence bail shipped that kicked state; the decision block now runs
    BEFORE the step (same fix in polish_router_module). (2) --fit-train
    {all,core}: core freezes the shared U*/V* basis (the Newton-Schulz
    candidates), only cores/centroids/router train - a near-convex polish
    for an init that already captures most of the delta energy. With
    --bank-init t2 the defaults become a polish profile (adamw @ 1e-4,
    warmup 20, train=core) unless the user pins --fit-method/--fit-lr;
    the init baseline (eval8) is now always printed (also with
    --skip-fit-guard). fit_sig carries fit_train (t2 fits refit once).
## version: 2026-09-08.2 - WHBANKS (13): --bank-init t2 replaces the blind
    diagonal init with a Tucker-2 DENSE core built in the WHITENED metric
    (whbank_build.py + whrank.py): the shared basis comes from the whitened
    scatters of the REAL deltas (covs from the pair-pool cache, factors
    cached in fit_r<rank>t2/whbank), the per-expert payload becomes a dense
    core C (n_exp, r, r) instead of a diagonal (r,) - the step-0 capture
    jumps from the joint-v1 diag ~0.3% to the shared-subspace energy (gate
    45%). Both bank families stay buildable/checkable side by side
    (whbank_build.py: t2_k* + dia_d* + whsvd_r* report). t2 keys on a
    SEPARATE fit dir (fit_r<rank>t2); default --bank-init svd is
    bit-identical to UPDATE-12. Runtime template reads cfg.field.core.
## version: 2026-09-08.1 - BANK2 (12): the H2 fix from the IRON diagnostics
    (the k-sweep on the old model showed the gap PERSISTS at top-1 -> the
    low-rank delta capacity of the single bank dominates). --banks 2 adds a
    SECOND coordinate bank (U2,V2,C2 per side): the reachable delta space
    doubles, the mixture ceiling min(n_exp, rank) lifts to 2*n_exp. The bank-2
    SVD init is a GREEDY RESIDUAL stage computed from the projection stash
    (no extra streaming pass): R_e = B_e - U_B diag(C_e) V^T, the same
    topk_eigh/joint-align machinery on the residual. banks=2 keys on a
    SEPARATE fit dir (fit_r<rank>b2) + fit_sig/banks, so existing r=banks=1
    caches/artifacts are untouched (nothing is re-checked). Runtime template
    reads cfg.field.banks; old artifacts load unchanged.
## version: 2026-09-07.1 - IRON PROBES (11.2): decomposition of the iron-test
    residual KL into the four known failure modes, almost for free:
    (1) probe A "blind at start": every eval window is a cold start, so KL is
    now ALSO accumulated per position bucket (0-7 / 8-31 / 32-127 / 128+) in
    eval_vs_cache_disk AND in the live iron eval - zero extra model passes.
    (2) probe B "lost without a prompt": --iron-prompt TEXT prepends TEXT to
    every eval window (same scored targets) and runs prompted vs unprompted
    live passes at the reference k (+2 passes only when the flag is set).
    (3) probe C "SwiGLU cross-noise": probe_preact.py (offline, no model
    passes) measures the preact-vs-postact composition gap of the artifact
    itself on the saved pair pool.
## version: 2026-09-06.7 - 10.9-PRE DIAGNOSTICS + FIELD MODE (29.8/29.9):
    (1) --verify-topk K: after the normal verify, base vs artifact are
    re-evaluated with MoE top_k=K on BOTH models (routers patched at runtime,
    KL computed LIVE - the lp cache was built under the original top_k). This
    is the "iron test": at K=1 the field's preact/postact compositions
    coincide, so a collapsing gap indicts the pre-activation mixing (H1) and
    a persisting gap indicts the low-rank delta capacity (H2).
    (2) --field-mode {preact,postact}: postact composition end-to-end (fit,
    refine, artifact export config.field.field_mode, runtime template);
    parameters are identical, fit_sig includes field_mode. Default preact =
    previous behavior bit-for-bit.
    (3) probe_capacity.py gained the 'postact' arm (same test at block level,
    no model needed).
## version: 2026-09-05.6 - POOL/SVD-VERSION-AWARE FIT SIG + JOINT INIT UPGRADE:
    (1) fit_sig now fingerprints the POOL (per-block pair counts) and the SVD
    init version - before, a --pool-recalibrate with a bigger cap silently
    KEPT the old fit (the sig did not see the pool), and better init files
    did not re-trigger the fit either. (2) init_svd_blk*.pt carry svd_ver;
    files built by an OLDER init algorithm are rebuilt automatically by the
    same self-heal that covers missing files (2026-09-05.5), so upgrading
    needs no flags and no manual deletions. (3) the stage-5 banner now shows
    the EFFECTIVE init (it was printed before the effective value existed,
    always reading like "svd init"), and the fit log prints the init's
    step-0 delta-energy capture so an uninformative init is visible.
## version: 2026-09-05.5 - SVD-INIT SELF-HEAL: missing fit_dir/init_svd_blk*.pt
    (pool cache from a pre-SVD build, or an earlier run that fell back to the
    random init) are detected BEFORE the fit and rebuilt by a streaming pass
    from the real expert deltas - the optimizer never starts blind again. The
    stage-5 fallback is now a loud WARNING + report metadata, and refine
    rounds key their signature on the fit state (a re-fit invalidates stale
    refine caches: the captured pairs are the field's own forward outputs).
## version: 2026-09-05.4 - REFINE RESUME + PARALLEL REFIT: a completed refine
    round is skipped on re-runs (done_r*.json markers in the run cache), a
    half-done round reuses the captured pairs (pairs_sig.json) instead of
    re-streaming the whole model for hours, and the refit loop now honors
    --fit-workers (before: always one worker). --refresh-refine forces a
    full redo.
## version: 2026-09-05.2 - SWAP-STORM GUARD (refine freeze fix): the capture
    pass of the refine round kept per-block pair chunks of up to 8192 pairs in
    RAM for ALL blocks at once (~1.1 GB) while the io-cache ram copy was also
    growing - on a 3.2 GB-free box Windows slid into a swap-storm at ~block 13
    (no MemoryError, the run just froze). Now: the flush threshold adapts to
    free RAM (1024 pairs below 8 GB -> resident ~= one batch, ~0.4 GB), the
    capture pass prints a per-window progress line ("frozen" vs "working" is
    visible), and BlockStreamRunner re-checks io-cache ram at every stage
    (see hf_stream).
## version: 2026-09-05.1 - LOW-RAM FIX: (1) a cached pair pool SMALLER than the
    requested --per-layer-cap is now KEPT by default (>= usable floor of
    max(4096, 16*rank, cap/2) pairs/block) instead of forcing the most
    expensive re-collection, which OOM-crashed low-RAM boxes and looked like
    "nothing was saved" - --pool-recalibrate forces the old behavior;
    (2) the "cached pool holds ..." notice prints once per run, not 3-6x.
## version: 2026-09-04.3 - RESUME FIX: the run cache no longer resets to zero
    after an interruption ("nothing was saved"): (1) art_meta.json is written
    at the START of stage 4 (it was written last, so a kill during the silent
    centroids+SVD loop invalidated the whole pair pool on restart); (2) the
    stage-4 init loop is resumable per block (existing init_blk*.pt are
    reused, only the missing ones are rebuilt) and prints per-block progress
    + timing; (3) the stage-5 fit is resumable per block (fit_blk*.pt +
    fit_partial.json with the exact fit_sig; only unfitted blocks re-run;
    finished blocks are reused verbatim); (4) all cache jsons and pair/fit
    saves are atomic (tmp + os.replace) - a kill mid-save cannot leave a torn
    file that poisons the next run. Semantics of the stage-4 skip: the pair
    pool is reused whenever it is on disk (the expensive, model-forward part);
    only missing centroids/init files are rebuilt from the model.
## version: 2026-09-04.2 - UPDATE-10 speed rebuild + the user's three holes:
    --fit-init {svd,random}: SVD init of U,V,C from the STACKED expert deltas
    (shared-basis randomized SVD over the streamed experts, saved per-rank as
    fit_dir/init_svd_blk*.pt; --refresh-init rebuilds those for an existing
    pool without recalibrating); --fit-autocast {auto,on,off} with the honest
    real-step probe (1 warmup + 3 timed steps per dtype arm, >=1.2x rule);
    --fit-method muon|muon-cosine (BY-NAME split: U*/V* only, C*/gw never),
    --muon-max-dim/--muon-ns-steps; jitter routing follows the clean anchor
    row inside fit_field_module. Toy bench: svd-init same quality in ~3x
    fewer steps, autocast 1.7-1.8x/step.
    --fit-guard-warmup (-1=auto) and --strict-fit-guard (old hard error),
    --fit-lr-warmup (linear Adam adaptation ramp)
