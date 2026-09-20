# version: 2026-09-13.3 - DENSE-DEFAULT (13.7.1): --bank-init default flips
#   svd -> dense. Evidence: the dense full core beat the legacy diag init on
#   EVERY zero-shot A/B to date - the toy TinyMoE, the micro-real MoE
#   google/switch-base-8 (routed-activation error -6..-11% up / -38..-54%
#   down, final KL 0.010 vs 0.020 at r=32 after the same fit budget), and
#   the third geometry HuggingFaceTB/nanowhale-100m-base (E=4, d=320,
#   dff=640, top-2: lower weight rel-MSE on all 8 layers x all ranks 8/16/32
#   in the Task-47 zero-shot bench). The user-facing goal is maximal
#   zero-shot quality with zero tuning - the best-known init is now the
#   default. --bank-init svd keeps the bit-identical 13.6.x behavior; t2
#   untouched. No file-format change: dense artifacts already carried
#   core="dense" since 13.7.
# version: 2026-09-13.1 - added --bank-init dense (13.7): the FULL core
#   C_e = U^T dW_e V on the same joint-v2 basis as svd - the zero-shot-first
#   init mode (best step-0 quality on the toy AND the micro-real MoE, and it
#   wins after the fit too). Lives in fit_r<rank>dense; the default svd and
#   the t2 whbank are untouched.
# version: 2026-09-11.1 - added --fit-fresh (13.6.4): the one-flag "refit
#   from scratch" - stage 5 ignores fit_meta.json / fit_partial.json /
#   fit_blk*.pt and refits ALL blocks with the current settings; the pool
#   and the SVD-init caches are kept. No more manual fit_r* deletion.
# version: 2026-09-10.2 - CLI-CLARITY (13.6): the "collector" - single source
#   of truth for the command line. What used to be a 76-flag flat argparse
#   block plus a 141-line changelog header is now data-driven: (1) COMPONENTS
#   - every runtime file with its version stamp, VER constants, role, flow
#   and a sha256 fingerprint printed on EVERY run (the log always shows WHICH
#   files actually executed, and a shadow copy on sys.path is flagged);
#   (2) STAGES - the 9-stage table with does/reads/writes/uses (--list-stages);
#   (3) GROUPS - every flag with a required/optional tag, default and full
#   description (--help shows short lines, --list-flags the full table);
#   (4) canonical_command - the effective re-run command printed each run.
#   Stdlib only: importing this module must never pull torch/transformers.
"""hf_cli.py - the CLI collector for hf_pipeline.py.

Everything the command line is made of lives here, in one place:

  COMPONENTS  the runtime files: what each one is, its version stamp, its
              VER constants, what it feeds - printed as a manifest with
              sha256 fingerprints on EVERY run (a log always shows WHICH
              files executed; a shadow copy on sys.path is flagged)
  STAGES      the 9 pipeline stages: what each does, reads, writes, which
              components it uses (--list-stages)
  GROUPS      every CLI flag: its group, required/optional tag, default,
              short help (--help) and full description (--list-flags)

No third-party imports: safe to use before the bootstrap stage, and the
info commands (--version / --list-flags / --list-stages / --help) never
need torch.
"""
import argparse
import hashlib
import os
import re
import shlex
import sys

CLI_REV = "2026-09-10.2"

# ---------------------------------------------------------------------------
# COMPONENTS - the files this pipeline is made of.
#   core  - loaded by every pipeline run (the fingerprint banner shows each)
#   role  - what the file IS (one line)
#   feeds - who consumes its outputs ("where it flows")
# The version stamp is the file's first "# version:" line; VER constants are
# scraped from the source text (no import, so a broken file still reports).
# ---------------------------------------------------------------------------
COMPONENTS = [
    dict(file="hf_pipeline.py", core=True,
         role="orchestrator: stage plan, directories, fit presets, reports",
         feeds="runs every stage; writes results/ + the artifact"),
    dict(file="hf_cli.py", core=True,
         role="CLI collector: parser, flag/stage tables, component manifest",
         feeds="hf_pipeline.main() (this banner, --help, --list-*)"),
    dict(file="hf_env.py", core=True,
         role="pins HF caches INSIDE the project folder (HF_HOME etc.)",
         feeds="transformers / huggingface_hub downloads"),
    dict(file="hf_gguf_to_hf.py", core=True,
         role="GGUF -> HF: resolve/download, light catalog, on-the-fly dequant",
         feeds="stage 1 (download); every streaming tensor read"),
    dict(file="hf_stream.py", core=True,
         role="BlockStreamRunner: backbone in RAM, per-block expert reads, "
              "prefetch, io-cache",
         feeds="stages 3-4 (base/calibrate), refine capture, save"),
    dict(file="hf_field_transform.py", core=True,
         role="field engine core: FieldSparseMoe (+du-bank 13.5), pair pool, "
              "fit, SVD/t2 init, artifact writer",
         feeds="stages 4-7; the artifact itself"),
    dict(file="whbank_build.py", core=True,
         role="whitened Tucker-2 delta banks: build/check t2_k*/dia_d*/whsvd",
         feeds="stage 5 (fit) under --bank-init t2"),
    dict(file="whrank.py", core=True,
         role="whitening/rank math behind the banks",
         feeds="whbank_build"),
    dict(file="modeling_field_template.py", core=True,
         role="runtime template rendered INTO the artifact "
              "(reads cfg.field incl. u_mode)",
         feeds="stage 6 (save) -> the artifact's modeling file"),
    dict(file="hf_chat.py", core=True,
         role="chat with the finished artifact",
         feeds="you, after the run"),
    dict(file="probe_capacity.py", core=False,
         role="block-level capacity probe (preact/postact/du arms)",
         feeds="diagnostics"),
    dict(file="probe_preact.py", core=False,
         role="offline preact-vs-postact composition gap on the pair pool",
         feeds="diagnostics"),
    dict(file="init_audit.py", core=False,
         role="init/handoff audit: npz->init_svd round-trip, verdicts",
         feeds="diagnostics"),
    dict(file="router_audit.py", core=False,
         role="router k-sweep / balance audit on an artifact",
         feeds="diagnostics"),
    dict(file="router_ft.py", core=False,
         role="router fine-tune on a finished artifact",
         feeds="repair"),
    dict(file="bank_apply.py", core=False,
         role="swap a delta bank into an EXISTING artifact",
         feeds="repair"),
    dict(file="deploy_check.py", core=False,
         role="artifact self-check for a target machine",
         feeds="deploy"),
    dict(file="field_dims.py", core=False,
         role="per-dim contribution probe of the field",
         feeds="diagnostics"),
    dict(file="temp_calibrate.py", core=False,
         role="temperature/min-p calibration for hf_chat",
         feeds="hf_chat"),
]

VER_RE = re.compile(r'^([A-Z][A-Z0-9_]*_VER)\s*=\s*["\']([^"\']+)["\']', re.M)


def file_facts(base, fname):
    """sha256 + version stamp + VER constants of one file, WITHOUT importing
    it (a stale/broken file must still report its identity)."""
    path = os.path.join(base, fname)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        blob = f.read()
    text = blob.decode("utf-8", "replace")
    m = re.search(r"^# version:\s*(.+)$", text, re.M)
    return dict(
        sha=hashlib.sha256(blob).hexdigest()[:12],
        stamp=(m.group(1).strip() if m else ""),
        vers=["%s=%s" % kv for kv in VER_RE.findall(text)],
        size=len(blob),
    )


def shadow_copies(base):
    """Other hf_pipeline.py copies visible on sys.path / cwd - the classic
    'why does my edit do nothing' trap. Returns [(dir, sha12), ...]."""
    out, seen = [], set()
    for d in [os.getcwd()] + list(sys.path):
        d = os.path.abspath(d or ".")
        if not os.path.isdir(d) or d in seen or d == base:
            continue
        seen.add(d)
        f = file_facts(d, "hf_pipeline.py")
        if f:
            out.append((d, f["sha"]))
    return out


def _stamp_head(stamp, cap=34):
    """'2026-09-10.1 - DU-BANK (13.5): long text...' -> '2026-09-10.1 - DU-BANK (13.5)'"""
    if not stamp:
        return "(no stamp)"
    return stamp.split(":")[0].strip()[:cap]


def print_manifest(base, verbose=False):
    """The runtime fingerprint banner: WHAT files/versions THIS run executes.
    Compact on every run; --version adds role/flows/VER constants."""
    core = [c for c in COMPONENTS if c["core"]]
    tools = [c for c in COMPONENTS if not c["core"]]
    print("\n" + "=" * 74)
    print("== RUNTIME COMPONENTS of THIS run  (base: %s)" % base)
    print("=" * 74)
    missing = []
    for c in core:
        f = file_facts(base, c["file"])
        if f is None:
            missing.append(c["file"])
            print("%-28s MISSING  <- the run will fail fast" % c["file"])
            continue
        print("%-28s %s  sha:%s" % (c["file"],
                                    _stamp_head(f["stamp"]), f["sha"]))
        if verbose:
            print("    role : %s" % c["role"])
            print("    flows: %s" % c["feeds"])
            if f["vers"]:
                print("    vers : %s" % ", ".join(f["vers"]))
    for c in tools:
        f = file_facts(base, c["file"])
        if f is None:
            continue  # optional tools may be absent
        print("%-28s %s  sha:%s  (%s)" % (
            c["file"], _stamp_head(f["stamp"], 26), f["sha"],
            c["feeds"]))
    if missing:
        print("MISSING CORE FILES: %s" % ", ".join(missing))
    for d, sha in shadow_copies(base):
        print("WARNING: another hf_pipeline.py is importable from %s "
              "(sha:%s) - a shadow copy on sys.path/cwd can hijack imports; "
              "clean PYTHONPATH or run from the folder you think you run"
              % (d, sha))
    print("-" * 74, flush=True)


def print_version(base):
    """--version: identity card of the whole file set."""
    pipe = file_facts(base, "hf_pipeline.py")
    print("expert-press field engine | hf_pipeline %s | cli %s"
          % (_stamp_head((pipe or {}).get("stamp", "?")), CLI_REV))
    print_manifest(base, verbose=True)


# ---------------------------------------------------------------------------
# STAGES - the plan the pipeline executes (used by --list-stages and by the
# per-stage PLAN printout before every run).
# ---------------------------------------------------------------------------
STAGES = [
    dict(id="download", no="1",
         does="resolve + download the source (GGUF quant or bf16), light "
              "catalog (config+tokenizer)",
         reads="HF hub, or --gguf / --local-path",
         writes="hf_cache/hub, gguf_out catalog",
         uses="hf_gguf_to_hf"),
    dict(id="texts", no="2",
         does="calibration/eval text split (no overlap - leak fix) + "
              "tokenization",
         reads="corpus URLs or --calib-file/--eval-file/--calib-dataset",
         writes="cache_<tag>/texts_*.pt, eval_tokens.pt",
         uses="hf_pipeline"),
    dict(id="base", no="3",
         does="base ppl/log-prob cache + demo generation (STREAMING; the full "
              "model never loads)",
         reads="GGUF + texts",
         writes="cache_<tag>/lp_base/",
         uses="hf_stream"),
    dict(id="calibrate", no="4",
         does="pair pool (MoE in->out pairs via hooks) + block "
              "centroids/geometry (STREAMING)",
         reads="GGUF + texts",
         writes="cache_<tag>/pairs_blk*.pt, init_blk*.pt, art_meta.json",
         uses="hf_stream + hf_field_transform"),
    dict(id="fit", no="5",
         does="per-block field fit from the pool (model NOT in RAM; "
              "--fit-workers parallelizes)",
         reads="pairs_blk*.pt + init_svd_blk*.pt (or whbank t2)",
         writes="fit_r<rank>[duX|b2|t2]/fit_blk*.pt, fit_meta.json",
         uses="hf_field_transform (+whbank_build on t2)"),
    dict(id="refine", no="5b",
         does="self-distillation: the field feeds itself forward, warm refit "
              "rounds (opt-in, --refine-rounds)",
         reads="fit_blk*.pt + the lp cache",
         writes="refine_r<rank>/ rounds, refit fits",
         uses="hf_field_transform + hf_stream"),
    dict(id="save", no="6",
         does="assemble the artifact STREAMINGLY (backbone copied, experts "
              "dropped, field injected)",
         reads="GGUF backbone + fit_blk*.pt",
         writes="field_<tag>_r<rank>[duX]/ (config, safetensors, field_meta, "
                "README)",
         uses="write_field_artifact + modeling_field_template"),
    dict(id="verify", no="7",
         does="reload the artifact as a NORMAL model: KL/ppl vs the base + "
              "demo generation (+ iron probes)",
         reads="the artifact + the lp cache",
         writes="verify metrics into the report",
         uses="hf_field_transform (eval_vs_cache_disk)"),
    dict(id="report", no="8",
         does="artifact README + results/ report; self-cleanup of the dequant "
              "checkpoint after success",
         reads="run metrics",
         writes="artifact README.md, results/report*",
         uses="hf_pipeline"),
]

STAGE_ORDER = [s["id"] for s in STAGES]
STAGE_DESCR = {s["id"]: "%-9s %-2s  %s" % (s["id"], s["no"], s["does"])
               for s in STAGES}


def print_stage_table():
    """--list-stages: the full stage table + the plan toggles."""
    print("\nPipeline stages (run order):")
    w = max(len(s["id"]) for s in STAGES)
    for s in STAGES:
        print("\n  %s %-2s  %s" % (s["id"].ljust(w), s["no"], s["does"]))
        print("  %s     reads : %s" % (" " * w, s["reads"]))
        print("  %s     writes: %s" % (" " * w, s["writes"]))
        print("  %s     uses  : %s" % (" " * w, s["uses"]))
    print("\ncomponents behind the stages: hf_cli.print_manifest() / --version")
    print("\ntoggles:")
    print("  --stages fit,save,verify   run ONLY these stages")
    print("  --skip base,report         run all EXCEPT these")
    print("  --skip download            reuse-only: no network/downloads")
    print("  --test-only                texts+base+verify only (existing "
          "artifact, no fit/refine/save)")
    print("  --force-topk N             override the routing k for the whole "
          "run (base + artifact)")
    print("  --verify-topk 1,4,8        iron-test sweep: both models forced "
          "to each K (live eval)")
    print("  --iron-prompt TEXT         probe B: prompted vs unprompted "
          "KL at the reference k (+2 live passes)")
    print("  (--skip-reload-check == --skip verify; a re-run also auto-skips")
    print("   whatever is already cached: pool, log-probs, fits)")

# ---------------------------------------------------------------------------
# GROUPS - every CLI flag, with:
#   req   - pick | core | opt | tune | diag | info  (the "required vs
#           optional" column; legend printed by --list-flags)
#   short - the one-liner shown by --help (always mentions its default)
#   full  - the complete description shown by --list-flags (defaults to short)
#   kw    - everything else goes to argparse unchanged (default/type/choices/)
#           (action/metavar) - dests are argparse-standard (--fit-steps ->
#           fit_steps), so behavior is identical to the old flat parser.
# ---------------------------------------------------------------------------
def _F(flag, req, short, full="", **kw):
    return dict(args=(flag,), req=req, short=short,
                full=full or short, kw=kw,
                dest=flag.lstrip("-").replace("-", "_"))


GROUPS = [
 ("required - model source (pick ONE; defaults exist, so a bare run works)", [
    _F("--model", "pick",
       short="HF repo id (default: mradermacher OLMoE Q4_K_M GGUF); a bf16 "
             "repo needs CUDA",
       full="HF repo id. A GGUF repo downloads the --gguf-quant file; a bf16 "
            "repo (allenai/OLMoE-1B-7B-0924) loads safetensors and needs "
            "CUDA (~15 GB).",
       default="mradermacher/OLMoE-1B-7B-0924-GGUF"),
    _F("--gguf-quant", "pick",
       short="quant to take from a GGUF repo: Q4_K_M | Q4_K_S | Q3_K_M | "
             "Q8_0 | auto",
       full="which quant to download from a GGUF repo: Q4_K_M | Q4_K_S | "
            "Q3_K_M | Q8_0 | ... | auto (picks the best available)",
       default="Q4_K_M"),
    _F("--gguf-file", "pick", short="exact .gguf filename inside the repo "
                                    "(rarely needed)", default=None),
    _F("--gguf", "pick", short="local .gguf file - skips the download",
       default=None),
    _F("--gguf-out", "opt", short="folder for the light catalog / dequant "
                                  "checkpoint", default=None),
    _F("--gguf-base-repo", "opt",
       short="repo for the exact config/tokenizer (default: auto from GGUF "
             "metadata)", default=None),
    _F("--local-path", "pick",
       short="path to an already downloaded HF model dir", default=None),
    _F("--auto", "opt", short="zero-config: gguf-quant auto + balanced fit "
                              "preset (explicit flags win)",
       action="store_true"),
 ]),
 ("core - field shape (what the artifact IS)", [
    _F("--rank", "core", short="field rank r, the main size/quality lever "
                               "(default 32)", type=int, default=32),
    _F("--banks", "core",
       short="coordinate banks: 2 adds a second (U2,V2,C2) bank - capacity "
             "doubles (UPDATE-12)",
       full="field coordinate banks (UPDATE-12): 2 = a second (U2,V2,C2) "
            "bank per side - the delta capacity doubles (H2 fix); fits into "
            "fit_r<rank>b2, old caches untouched (default 1)",
       type=int, choices=(1, 2), default=1),
    _F("--bank-init", "core",
       short="init: dense (default) = full core on the joint SVD basis, "
             "best zero-shot (13.7); t2 = whitened Tucker-2 (whbank); "
             "svd = legacy joint diag",
       full="delta-bank init (13.7): dense (DEFAULT) = DENSE core "
            "C_e = U^T dW_e V on the SAME joint-v2 basis as svd, data-free "
            "(no calibration pool needed) - the zero-shot step-0 quality "
            "jumps (micro-real A/B: routed-activation error -6..-11% up "
            "side, -38..-54% down side; the gain survives the fit; "
            "confirmed on a third geometry, nanowhale-100m: lower weight "
            "rel-MSE on every layer and rank). Costs +r^2 per expert "
            "(~131K floats/block at r=32 - the x60 compression survives); "
            "fits into fit_r<rank>dense, old caches untouched. t2 = "
            "Tucker-2 DENSE core from the whitened whbank (banks cached in "
            "fit_r<rank>t2/whbank) - a converged polish start, not a "
            "zero-shot pick. svd = legacy joint-aligned diag, bit-identical "
            "to UPDATE-12 (the 13.6.x default; kept as the compat mode)",
       choices=["svd", "t2", "dense"], default="dense"),
    _F("--field-mode", "core",
       short="preact = fused mix before the nonlinearity; postact = "
             "per-expert branches (review Variant A)",
       full="composition of the field (10.9): preact = one fused pass, "
            "coordinates mixed BEFORE the nonlinearity (cheapest); postact = "
            "per-expert branches, nonlinearity PER EXPERT (zero SwiGLU "
            "cross-term error, ~1 extra down-GEMM per token). Same "
            "parameters either way",
       choices=["preact", "postact"], default="preact"),
    _F("--u-mode", "core",
       short="13.5 du-bank: per-expert deltas on the DOWN output factor "
             "(rank4 is the default stand)",
       full="per-expert deltas on the DOWN output factor (13.5, external "
            "review fig12/fig14): du_e = duA_e @ duB_e^T (rank modes) or "
            "duEdn_e (r,d) for 'full', mixed over the top-k AFTER the "
            "nonlinearity - new OUTPUT directions. LoRA start (B=0) is "
            "bit-identical to 'none'. Budget/block: rank-a +E*(r+d)*a, full "
            "+E*r*d. Fits into fit_r<rank>du*",
       choices=["none", "rank1", "rank4", "full"], default="none"),
    _F("--u-rank", "core", short="rank a of the du delta for --u-mode rank4 "
                                 "(default 4)", type=int, default=4),
    _F("--out", "opt", short="artifact folder (default: results/"
                             "field_<tag>_r<rank>[duX])", default=None),
 ]),
 ("optional - fit: optimizer (stage 5)", [
    _F("--fit-preset", "opt", short="bundle: fast | balanced | quality "
                                    "(explicit --fit-* flags always win)",
       default=None),
    _F("--fit-steps", "tune", short="steps per block (default 300, or the "
                                    "preset value)", type=int, default=None),
    _F("--fit-bs", "tune", short="fit batch size (default 4096, or the "
                                 "preset value)", type=int, default=None),
    _F("--fit-lr", "tune", short="learning rate (default 2e-3, or the preset "
                                 "value)", type=float, default=None),
    _F("--fit-method", "tune",
       short="adam | adamw | adam-cosine | rmsprop | muon | muon-cosine",
       full="optimizer: adam | adamw | adam-cosine | rmsprop | muon | "
            "muon-cosine (muon: NS-orthogonalized updates for the U*/V* "
            "factors, BY-NAME split; C*/router always stay on Adam)",
       default=None),
    _F("--fit-workers", "tune", short="parallel workers for independent "
                                      "blocks (2-4 on a multi-core CPU)",
       type=int, default=1),
    _F("--fit-autocast", "tune",
       short="bf16-autocast fit; auto = the honest real-step probe decides",
       full="bf16-autocast fit (params fp32, matmuls bf16 via oneDNN - "
            "1.7-1.8x/step on the toy): auto = the honest real-step probe "
            "decides per geometry (keeps fp32 when bf16 is not faster)",
       choices=["auto", "on", "off"], default="auto"),
    _F("--fit-jitter", "tune",
       short="gaussian noise on fit inputs (0.2-0.3 helps small pools; "
             "0 = off)", type=float, default=0.0),
    _F("--fit-early-stop", "tune", short="stop after 2 flat mse checkpoints "
                                         "every N steps (0 = off)",
       type=int, default=0),
 ]),
 ("optional - fit: init & what trains", [
    _F("--fit-init", "opt",
       short="svd = shared basis of the REAL expert deltas (~50-70 pct "
             "energy at step 0); random = legacy",
       full="U,V,C initialization: svd = shared basis of the REAL expert "
            "deltas (captures most of the delta energy at step 0; computed "
            "once per rank from the streamed experts); random = the old "
            "randn*0.02/zeros",
       choices=["svd", "random"], default="svd"),
    _F("--fit-train", "opt",
       short="what the fit may touch: all | core | cores (basis frozen = "
             "polish)",
       full="what the fit may touch (13.1/13.3): all = the legacy full fit; "
            "core = the shared U*/V* basis stays FROZEN, only cores C*/"
            "centroids/router train; cores = the stricter polish: the "
            "centroids w* are FROZEN too (external 'freeze W0' recipe), "
            "only the cores train (default: core with --bank-init t2, all "
            "otherwise)",
       choices=["all", "core", "cores"], default=None),
    _F("--muon-max-dim", "tune",
       short="muon split gate: NS-orthogonalize only factors with "
             "min(shape) <= this", type=int, default=512),
    _F("--muon-ns-steps", "tune", short="Newton-Schulz iterations per muon "
                                        "update (default 5)", type=int,
       default=5),
    _F("--fit-lr-warmup", "tune", short="linear lr ramp over the first N fit "
                                        "steps (0 = off)", type=int,
       default=0),
 ]),
 ("optional - fit: guard & resume", [
    _F("--fit-guard-warmup", "opt",
       short="arm the 2x divergence bail only after N steps (-1 = auto, "
             "0 = always armed)", type=int, default=-1),
    _F("--strict-fit-guard", "opt",
       short="abort the run when a block's mse is not 2 pct under the "
             "baseline (default: warn + ship best)",
       action="store_true"),
    _F("--skip-fit-guard", "opt", short="disable the fit guards entirely",
       action="store_true"),
    _F("--fit-fresh", "opt",
       short="refit EVERY block from scratch: ignore fit_meta.json / "
             "fit_partial.json / fit_blk*.pt (pool + SVD-init kept)",
       full="--fit-fresh: stage 5 (fit) behaves as if NO fit cache existed "
            "on disk - every block is refitted from scratch with the "
            "current settings. The calibration pool and the init_svd_blk* "
            "caches are NOT touched, so this is cheap to add. The default "
            "(off) resumes an interrupted fit per block, which is what you "
            "want after a kill - pass this flag only when you deliberately "
            "want a clean refit under the same settings.",
       action="store_true"),
    _F("--refresh-init", "opt",
       short="rebuild ONLY init_svd_blk*.pt for the existing pool and exit",
       action="store_true"),
    _F("--refresh-refine", "opt",
       short="ignore refine caches (done markers + pairs) and redo the "
             "rounds", action="store_true"),
 ]),
 ("optional - router & refine", [
    _F("--fit-router", "opt",
       short="let the ORIGINAL router join the rebuild: after | joint | off",
       full="the original router joins the rebuild (in place, pairs from "
            "disk): after = short anchored polish once the field fit is "
            "done; joint = the router trains alongside the field from step "
            "0. Toy-bench caveat: usually NOT the bottleneck - treat as a "
            "cheap diagnostic; --refine-rounds is the stronger lever",
       choices=["off", "after", "joint"], default="off"),
    _F("--router-steps", "tune", short="anchored router-polish steps for "
                                       "--fit-router after (default 80)",
       type=int, default=80),
    _F("--router-lr", "tune", short="router polish lr (default: the fit lr)",
       type=float, default=None),
    _F("--router-anchor", "tune", short="L2 anchor pulling the router to the "
                                        "original (0 = free; higher = safer)",
       type=float, default=0.03),
    _F("--refine-rounds", "opt",
       short="self-distillation rounds after the first fit (try 1-2; "
             "0 = off)", type=int, default=0),
 ]),
 ("optional - data & calibration (stages 2-4)", [
    _F("--calib-file", "opt", short="calibration text file (default: the "
                                    "bundled corpus URLs)", default=None),
    _F("--eval-file", "opt", short="eval text file (disjoint from "
                                   "calibration)", default=None),
    _F("--calib-dataset", "opt", short="HF dataset id, e.g. wikitext-2-raw-v1 "
                                       "(requires datasets)", default=None),
    _F("--text-cap", "tune", short="text characters cap (default 3000000)",
       type=int, default=3_000_000),
    _F("--calib-windows", "tune", short="calibration windows (default 3)",
       type=int, default=3),
    _F("--calib-bsz", "tune", short="calibration batch size (default 8, or "
                                    "16 on a GPU under the high profile)",
       type=int, default=None),
    _F("--calib-ctx", "tune", short="calibration context length (default "
                                    "512)", type=int, default=512),
    _F("--per-layer-cap", "tune", short="pairs per block cap for the pool "
                                        "(default 8192)", type=int,
       default=8192),
    _F("--pool-recalibrate", "opt",
       short="force pool re-collection when the cache is smaller than the "
             "cap (default: keep a usable pool)",
       full="force re-collection when the cached pair pool holds fewer "
            "pairs/block than --per-layer-cap (default: keep the cached "
            "pool when still usable - re-collection is the most expensive "
            "stage and OOM-crashes low-RAM boxes)",
       action="store_true"),
 ]),
 ("optional - eval, verify & demo (stage 7)", [
    _F("--eval-chunks", "tune", short="eval chunks for ppl/KL (default 50)",
       type=int, default=50),
    _F("--kl-chunks", "tune", short="KL chunks (default 16)", type=int,
       default=16),
    _F("--eval-ctx", "tune", short="eval context length (default 512)",
       type=int, default=512),
    _F("--gen-tokens", "tune", short="demo generation tokens (default 48; "
                                     "0 = off)", type=int, default=48),
    _F("--gen-rep-pen", "tune", short="repetition penalty for demo "
                                      "generations (default 1.15; 1.0 = off)",
       type=float, default=1.15),
    _F("--verify-topk", "diag",
       short="iron test sweep: live KL with MoE top_k forced to each K, "
             "e.g. 1,4,8",
       full="iron test (29.8, update 11: now a SWEEP): after the normal "
            "verify, re-run base vs artifact LIVE with MoE top_k set to "
            "each K on BOTH models. '1' removes the router mixture entirely "
            "(H1 vs H2 verdict); several values show how KL behaves as more "
            "experts mix",
       default="", metavar="K1,K2,..."),
    _F("--iron-prompt", "diag",
       short="probe B: prepend TEXT to every eval window (prompted vs "
             "unprompted KL)",
       default="", metavar="TEXT"),
    _F("--force-topk", "diag",
       short="override MoE top_k for the WHOLE run (base + artifact; the lp "
             "cache goes to a separate folder)",
       full="override the MoE routing top_k with N for BOTH the base and "
            "the artifact for the WHOLE run (0 = the value from the GGUF "
            "metadata / artifact). The base log-prob cache goes to a "
            "SEPARATE lp_base_topkN folder, so the native cache and the "
            "standard verify numbers stay intact. NOTE (13.6.2): the pair "
            "pool is ALWAYS collected at the NATIVE k - the field is fitted "
            "on native-k pairs, so a forced-k run verifies the artifact on "
            "a routing regime it was not fitted for (the iron test); for "
            "the sweep on a finished artifact prefer "
            "--test-only --verify-topk",
       type=int, default=0, metavar="N"),
 ]),
 ("optional - io, memory & speed", [
    _F("--io-cache", "tune",
       short="ram = packed GGUF tensors kept in RAM (later passes read no "
             "disk); default: auto by profile/RAM",
       choices=["disk", "ram"], default=None),
    _F("--io-threads", "tune", short="threads for GGUF dequant of expert "
                                     "tensors (default 1, or 4 under high)",
       type=int, default=None),
    _F("--prefetch", "tune", short="background prefetch of the next expert "
                                   "block (default 1; 0 saves ~1 block of "
                                   "RAM)", type=int, default=1),
    _F("--threads", "tune", short="limit torch CPU threads (default: all "
                                  "cores)", type=int, default=None),
    _F("--device", "opt", choices=["auto", "cuda", "cpu"], default="auto",
       short="compute device (default auto)"),
    _F("--dtype", "opt", choices=["auto", "bfloat16", "float16", "float32"],
       default="auto", short="compute dtype (default auto = bf16)"),
    _F("--profile", "opt",
       short="hardware profile auto|low|high; explicit io-*/calib-bsz/dtype "
             "always win",
       full="hardware profile: auto detects it (CUDA, or 32 GB+ RAM and 8+ "
            "cores -> high, else low); high lifts the conservative defaults "
            "(io-cache ram, io-threads 4, calib-bsz 16 on GPU, fp16 on "
            "pre-Ampere GPUs); low = the classic cautious defaults",
       choices=["auto", "low", "high"], default="auto"),
    _F("--low-mem", "opt", short="memory-frugal caps (lower RAM, nearly the "
                                 "same metrics)", action="store_true"),
    _F("--smoke", "opt", short="mini wiring run: short fit/eval",
       action="store_true"),
 ]),
 ("optional - stages & plan", [
    _F("--stages", "opt", short="run ONLY these stages, e.g. fit,save,verify "
                                "(names: --list-stages)",
       default=None, metavar="A,B,..."),
    _F("--skip", "opt", short="run all stages EXCEPT these, e.g. "
                              "base,verify,report",
       default=None, metavar="A,B,..."),
    _F("--test-only", "opt",
       short="plan locked to texts,base,verify (existing artifact; no "
             "fit/refine/save)", action="store_true"),
    _F("--no-cache-verify", "opt", short="skip the 2-chunk log-prob cache "
                                         "self-check (stage 3)",
       action="store_true"),
    _F("--skip-reload-check", "opt", short="same as --skip verify",
       action="store_true"),
    _F("--list-stages", "info", short="print the stage table (does / reads / "
                                      "writes / uses) and exit",
       action="store_true"),
 ]),
 ("optional - artifacts & cleanup", [
    _F("--save-backbone", "opt",
       short="keep = backbone as in the source; bf16 = dequant the backbone "
             "(CPU-friendly artifact)",
       choices=["keep", "bf16"], default="keep"),
    _F("--max-shard", "opt", short="safetensors shard size (default 4GB)",
       default="4GB"),
    _F("--full-dequant", "opt",
       short="build a full dequant checkpoint (~14 GB) instead of on-the-fly "
             "dequant", action="store_true"),
    _F("--keep-dequant", "opt", short="keep the dequant checkpoint after "
                                      "success", action="store_true"),
    _F("--cleanup", "opt", short="erase the GGUF after success (~4.4 GB; the "
                                 "pool cache and artifact stay)",
       action="store_true"),
 ]),
 ("info - print and exit (no torch needed)", [
    _F("--list-flags", "info",
       short="print EVERY flag: required/optional tag, default, full "
             "description", action="store_true"),
    _F("--version", "info",
       short="print what files/versions this run executes (sha256 "
             "fingerprints)", action="store_true"),
 ]),
]

ALL_FLAGS = [f for _, fl in GROUPS for f in fl]

REQ_LEGEND = ("req column:  pick = model source (choose one way; a default "
              "exists)   core = defines the artifact, usually typed "
              "explicitly   opt = safe default   tune = performance/memory "
              "knob   diag = probe, off by default   info = print and exit")

_RECIPES = """\
recipes (every run also prints its effective command as "cmd:"):
  full run, zero-config          python3 hf_pipeline.py --auto
  full run, explicit (13.5)      python3 hf_pipeline.py --rank 64 --u-mode rank4
  new rank from the cached pool  python3 hf_pipeline.py --stages fit,save,verify --rank 64
  du-bank A/B (no collision)     python3 hf_pipeline.py --stages fit,save,verify --u-mode rank4
  t2 polish fit                  python3 hf_pipeline.py --stages fit,save,verify --bank-init t2
  self-distillation              python3 hf_pipeline.py --stages fit,refine,save,verify --refine-rounds 1
  verify an existing artifact    python3 hf_pipeline.py --test-only
  iron test                      python3 hf_pipeline.py --test-only --verify-topk 1,4,8
  reuse only, no network         python3 hf_pipeline.py --skip download
info commands (no torch needed): --help | --version | --list-stages | --list-flags
"""


def _esc(text):
    """argparse %-interpolates help strings - escape literal percent signs."""
    return text.replace("%", "%%")


def build_parser(description):
    """The grouped parser. Flags/dests/defaults/choices are identical to the
    old flat block - only the presentation (groups, tags, short helps) and
    the two new info flags are added."""
    ap = argparse.ArgumentParser(
        description=description, epilog=_RECIPES,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for title, flags in GROUPS:
        g = ap.add_argument_group(title)
        for spec in flags:
            kw = dict(spec["kw"])
            kw.setdefault("help", _esc(spec["short"]))
            g.add_argument(*spec["args"], **kw)
    return ap


def list_flags():
    """--list-flags: every flag, required/optional tag, default, FULL
    description - the complete one-screen reference."""
    import textwrap
    print("\nEVERY FLAG of hf_pipeline.py  (%d flags in %d groups)"
          % (len(ALL_FLAGS), len(GROUPS)))
    print(REQ_LEGEND)
    print("nothing is hard-required: defaults give a full "
          "run; 'pick' marks the model-source decision\n")
    for title, flags in GROUPS:
        print("\n[%s]" % title)
        for spec in flags:
            dflt = spec["kw"].get("default")
            if spec["kw"].get("action") == "store_true":
                dflt = "off"
            elif dflt is None:
                dflt = "(auto)"
            lines = textwrap.wrap(spec["full"], 58) or [""]
            print("  %-22s %-5s %-9s %s" % (spec["args"][0], spec["req"],
                                            dflt, lines[0]))
            for extra in lines[1:]:
                print("  %-22s %-5s %-9s %s" % ("", "", "", extra))
    print("")


def canonical_command(args):
    """The EFFECTIVE command line: only flags that differ from their parser
    default (post-profile/preset values included), so any run can be
    reproduced by copy-paste."""
    parts = []
    for spec in ALL_FLAGS:
        dest = spec["dest"]
        if dest in ("list_stages", "list_flags", "version", "help"):
            continue
        val = getattr(args, dest, None)
        if spec["kw"].get("action") == "store_true":
            if val:
                parts.append(spec["args"][0])
            continue
        if val is None or val == spec["kw"].get("default"):
            continue
        if isinstance(val, (list, tuple)):
            if not val:
                continue
            parts.append("%s %s" % (spec["args"][0],
                                    ",".join(map(str, val))))
        elif isinstance(val, str):
            if val.strip():
                parts.append("%s %s" % (spec["args"][0], shlex.quote(val)))
        else:
            parts.append("%s %s" % (spec["args"][0], shlex.quote(str(val))))
    return " ".join(parts)
