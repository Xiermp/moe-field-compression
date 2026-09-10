# -*- coding: utf-8 -*-
# version: 2026-09-08.4 - UPDATE-13.3 (wh-check): (1) честный (plain-Frobenius,
# неотбеленный) capture рядом с отбеленным — в банках (caps_raw в npz), в
# --verify и в отчёте; большая щель raw<<whitened = захват живёт в
# выбросах ковариации, а не «банк плох». (2) --wh-damp-frac/--wh-damp-base:
# демпфер обратного корня настраиваем (дефолт 1e-3·λmax — как и было; чужой
# damp в кэше пересчитывает факторы, кастомный требует --force для банков).
# version: 2026-09-08.2 - UPDATE-13.2 (tools): вариант 'spk' - проба топлива
# разреженных "всплесков" (AWQ/SpQR-стиль) в остатке после T2: топ-f элементов
# по |R|*sqrt(w) и по |R|, ориентиры (случай=f, гаусс~14.4% при 1%), хранение
# и вердикт (awq@1% >= 25% -> UPDATE-14 оправдан). Ничего не пишет.
# version: 2026-09-08.1 - UPDATE-13 (whbanks): сборка и проверка отбелённых
# банков дельт. ОБА варианта собираются рядом и проверяются одним инструментом.
"""
whbank_build.py — банки дельт в отбелённой (activation-weighted) метрике.

Варианты (каждый в своей подпапке <out>/, npz на блок/тензор):
  t2_k{K}    Tucker-2: общий отбелённый базис + ПЛОТНОЕ ядро K×K на эксперта.
             Контейнерный путь замены: hf_pipeline --bank-init t2 (ядро C
             становится матрицей (n_exp, r, r) вместо диагонали).
  dia_d{D}   легаси: общий базис + ДИАГОНАЛЬ (паритет с bank1/joint-v1,
             чей capture 0.3% известен из логов — якорь санити).
  whsvd_r{R} per-expert отбелённый SVD с масштабом --scale (чемпион по KL
             на игрушке: 0.012 бит/токен при r=16). Потолок качества и
             диагностика: в общий базис контейнера поля НЕ помещается.
  spk        ПРОБА топлива разреженных «всплесков» (AWQ/SpQR-стиль): остаток
             R = dM − U G_e Vᵀ после t2_k{rank}; топ-f элементов по
             |R|·√w (активационно-взвешенный выбор) и по |R| (амплитуда),
             w_ij = Cy_ii·Cx_jj. Ориентиры: случайная поддержка = f;
             топ-f гауссовского шума ≈ 14.4% при f=1%. Топливо есть, если
             AWQ-выбор при f=1% собирает >= 25% энергии остатка. Нужен
             t2_k{rank} (строится автоматически). Ничего не пишет в контейнер.

Дельты — ТО ЖЕ определение, что в expert_basis_init (числа сравнимы с логами
joint-v1):  ΔW_e = dequant(W_e) − mean_e(W_e),  стриминг по одному эксперту
(_iter_expert_w), полный fp32-стек не материализуется.

Ковариации отбеливания — ИЗ КЭША pair pool (ничего не пересобираем):
  gu: Cx = E[x xᵀ]  — вход MoE-блока (точно, это вход gu);
      Cy = E[(mgu·x)(mgu·x)ᵀ] — preact центроида (прокси выхода gu).
  dn: Cx = E[h hᵀ], h = silu(g)·u от центроидного gu (прокси входа down);
      Cy = E[y yᵀ] — выход MoE-блока (прокси выхода down).
Факторы (C^{1/2}, C^{-1/2}) считаются ОДИН раз и кэшируются в
<out>/cov_blk{i}.npz — повторные прогоны и verify их переиспользуют.

Стриминг (2 прохода по экспертам на блок, как в build_whbank):
  A: скаттеры SL=Σ Mw Mwᵀ, SR=Σ Mwᵀ Mw (оба тензора gu/dn за один проход);
     один базис (K = max k) обслуживает ВСЁ семейство t2_k* + dia — свип k
     бесплатный (норм-трюк: ||M − U G Vᵀ||² = ||M||² − ||G||²).
  B: ядра C_e = Uᵀ Mw V и capture.

Память O(кусков ковараций + один эксперт), RAM как в стриминговом пайплайне.

CLI:
  python whbank_build.py --pool <cache_dir> --src <модель> --blocks 0-22 \
      --variants t2,whsvd,dia --k 16,24,32,48,64 --r 16 --scale 3.0
  python whbank_build.py --pool <cache_dir> --report-only
  python whbank_build.py --pool <cache_dir> --src <модель> --blocks 0,1 --verify

Пайплайн (hf_pipeline.py --bank-init t2) вызывает ensure_t2_init(...) в
stage-4: коварации, t2_k{rank} и init-словарь строятся/переиспользуются
автоматически, отдельный вызов не нужен.
"""

import os
import time

import numpy as np

FP = np.float32
WHBANK_T2_VER = "whrank-t2-v1"   # svd_ver init-файлов T2 (сталость по нему)


# ----------------------------- утилиты --------------------------------------

def norm2(A):
    return float(np.sum(A * A))


def raw_capture(dM, U, V, C):
    """Честный (неотбеленный, plain-Frobenius) capture реконструкции U C Vᵀ
    для дельты dM: 1 - ||dM - U C Vᵀ||²_F / ||dM||²_F.

    Считается норм-трюком (без реконструкции m x n):
      ||dM - R||² = ||dM||² - 2<dM, R> + ||R||²,
      <dM, UCVᵀ> = tr(Vᵀ dMᵀ U C)   (один скинни-матмул n x m x k),
      ||UCVᵀ||²   = tr(Cᵀ (UᵀU) C (VᵀV)) = <EC, FC>  (k x k, E=UᵀU, F=VᵀV).

    Это ДРУГАЯ линза, чем отбеленный capture (не лучше и не хуже): отбеленный
    взвешивает направления энергией активаций (= точная доля output-error
    энергии на калибровке), raw меряет долю самой массы дельты. Большой разрыв
    (raw << whitened) показывает, что захват живёт в немногих активационных
    направлениях (выбросы) — сигнал о вырожденности ковариации, а не о баге."""
    U = np.ascontiguousarray(U, dtype=FP)
    V = np.ascontiguousarray(V, dtype=FP)
    C = np.ascontiguousarray(C, dtype=FP)
    A1 = dM.T.astype(FP) @ U                       # (n, k)
    cross = float(np.sum((V.T @ A1) * C.T))        # tr(Vᵀ dMᵀ U C)
    E = U.T @ U                                    # (k, k)
    F = V.T @ V                                    # (k, k)
    r2 = float(np.sum((E @ C) * (F @ C)))          # tr(Cᵀ E C F)
    return (2.0 * cross - r2) / max(norm2(dM), 1e-30)


def _eigh_psd(C):
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    return np.maximum(w, 0.0), V


def sqrtm_psd(C, eps_frac=1e-6):
    w, V = _eigh_psd(C)
    return (V * np.sqrt(w + eps_frac * max(w.max(), 1e-12))) @ V.T


def inv_sqrt(C, eps_frac=1e-3, eps_abs=None):
    w, V = _eigh_psd(C)
    # 13.3: eps_abs — абсолютный демпфер (для --wh-damp-base mean-diag);
    # дефолт-путь (eps_frac·λmax) бит-в-бит как раньше
    eps = eps_abs if eps_abs is not None \
        else eps_frac * max(w.max(), 1e-12)
    return (V * (1.0 / np.sqrt(w + eps))) @ V.T


def top_eig(S, k):
    w, V = _eigh_psd(S)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx].astype(FP)


def mp_expected_capture(r, m, n):
    """Capture топ-r ЧИСТОГО шума (MP edge) — строка-ориентир в отчёте."""
    p, N = min(m, n), max(m, n)
    return (r / p) * (1.0 + np.sqrt(p / N)) ** 2


def human_bytes(x):
    return "%.1f KB" % (x / 1024.0) if x >= 1024 else "%d B" % x


def _silu(x):
    return x / (1.0 + np.exp(-x))


class Whit:
    """Отбеливатель одной стороны: Mw = Cy^{1/2} M Cx^{1/2}; обратный ход
    U = Cy^{-1/2} Uw, V = Cx^{-1/2} Vw (и per-expert A/B для whsvd)."""

    def __init__(self, Sx12, Sx_i, Sy12, Sy_i):
        self.Sx12, self.Sx_i = Sx12, Sx_i
        self.Sy12, self.Sy_i = Sy12, Sy_i

    def fw(self, M):
        return self.Sy12 @ M @ self.Sx12

    def unU(self, Uw):
        return self.Sy_i @ Uw

    def unV(self, Vw):
        return self.Sx_i @ Vw

    def unA(self, Aw):
        return self.Sy_i @ Aw

    def unB(self, Bw):                       # Bw (r, n) -> (r, n)
        return Bw @ self.Sx_i


# ----------------------------- ковариации -----------------------------------

def _pairs_path(pool_dir, i):
    return os.path.join(pool_dir, "pairs_blk%d.pt" % i)


def compute_covs(X, Y, mgu, act="silu"):
    """4 ковариации блока из пула пар + центроида gu (см. шапку файла).
    X (N, d), Y (N, d) bf16-тензоры пула; mgu (2dff, d) fp32."""
    import torch
    N = int(X.shape[0])
    d = int(X.shape[1])
    dff = int(mgu.shape[0]) // 2
    Xf = X.to(torch.float32)
    Cx_gu = (Xf.T @ Xf / N).numpy().astype(np.float64)
    Cy_dn = (Y.to(torch.float32).T @ Y.to(torch.float32) / N).numpy() \
        .astype(np.float64)
    Cy_gu = np.zeros((2 * dff, 2 * dff), dtype=np.float64)
    hsum = np.zeros((dff, dff), dtype=np.float64)
    mguN = mgu.numpy().astype(FP) if torch.is_tensor(mgu) \
        else np.asarray(mgu, dtype=FP)
    ch = max(1, (1 << 25) // max(2 * dff, 1))
    for s in range(0, N, ch):
        xb = Xf[s:s + ch].numpy().astype(FP)
        P = xb @ mguN.T                                   # (B, 2dff) preact
        Cy_gu += P.T.astype(np.float64) @ P.astype(np.float64)
        if act == "silu":
            g, u = P[:, :dff], P[:, dff:]
            hb = _silu(g) * u
        else:
            hb = P[:, :dff]                               # неизвестный act
        hsum += hb.T.astype(np.float64) @ hb.astype(np.float64)
    Cx_dn = hsum / N
    return dict(Cx_gu=Cx_gu, Cy_gu=Cy_gu, Cx_dn=Cx_dn, Cy_dn=Cy_dn, N=N)


def ensure_cov(i, mgu, pool_dir, whdir, geom, force=False, act=None,
               damp_frac=1e-3, damp_base="lmax"):
    """Факторы отбеливания блока i (кэш cov_blk{i}.npz). act берётся из geom
    (hidden_act); не-silu -> вход dn НЕ отбеливаем (Cx_dn = I), с варнингом.

    13.3: демпфер обратного корня настраиваем (он и так был: inv_sqrt
    eps_frac = 1e-3·λmax — при вырожденной ковариации это ЖЁСТЧЕ, чем
    предлагаемый «1% от mean(diag)»; флаги --wh-damp-frac/--wh-damp-base
    позволяют ослабить/перестроить). damp записывается в npz: чужой damp в
    кэше -> факторы пересчитываются (банки T2 при этом надо перестроить
    с --force — см. guard в run())."""
    path = os.path.join(whdir, "cov_blk%d.npz" % i)
    act = act or str(geom.get("hidden_act", "silu"))
    if os.path.isfile(path) and not force:
        z = np.load(path)
        st_damp = float(z["damp"]) if "damp" in z.files else 1e-3
        st_base = str(z["damp_base"]) if "damp_base" in z.files else "lmax"
        if abs(st_damp - damp_frac) < 1e-15 and st_base == damp_base:
            out = {}
            for side in ("gu", "dn"):
                out[side] = Whit(z[side + "_Sx12"], z[side + "_Sx_i"],
                                 z[side + "_Sy12"], z[side + "_Sy_i"])
            out["_N"] = int(z["N"])
            return out
        print("  cov blk%d: damp в кэше (%s·%s) != запрошенному (%s·%s) - "
              "пересчёт факторов" % (i, st_damp, st_base, damp_frac, damp_base),
              flush=True)
    import torch
    d = torch.load(_pairs_path(pool_dir, i), map_location="cpu")
    X, Y = d["X"], d["Y"]                      # {"X","Y"} bf16, как в пуле
    if int(X.shape[0]) == 0:
        raise SystemExit("whbank_build: pairs_blk%d.pt пуст - сначала пул "
                         "(stage 2-3 пайплайна)" % i)
    t0 = time.time()
    covs = compute_covs(X, Y, mgu if torch.is_tensor(mgu)
                        else torch.tensor(np.asarray(mgu, dtype=np.float32)),
                        act=act)
    if act != "silu":
        print("  WARNING: hidden_act=%r не silu - вход dn не отбеливается "
              "(Cx_dn = I)" % act, flush=True)
        covs["Cx_dn"] = np.eye(covs["Cx_dn"].shape[0])
    fac = {}
    for side in ("gu", "dn"):
        Cx, Cy = covs["Cx_" + side], covs["Cy_" + side]
        bx = float(np.mean(np.diag(Cx))) if damp_base == "mean-diag" \
            else float(np.linalg.eigvalsh(Cx)[-1])
        by = float(np.mean(np.diag(Cy))) if damp_base == "mean-diag" \
            else float(np.linalg.eigvalsh(Cy)[-1])
        fac[side] = Whit(sqrtm_psd(Cx).astype(FP),
                         inv_sqrt(Cx, eps_abs=max(damp_frac * bx, 1e-30))
                         .astype(FP),
                         sqrtm_psd(Cy).astype(FP),
                         inv_sqrt(Cy, eps_abs=max(damp_frac * by, 1e-30))
                         .astype(FP))
        print("  cov blk%d/%s: демпфер inv_sqrt eps = %.3g(x) / %.3g(y) "
              "(база %s, потолок усиления ~%.0fх / ~%.0fх)"
              % (i, side, damp_frac * bx, damp_frac * by,
                 "mean(diag)" if damp_base == "mean-diag" else "λmax",
                 1.0 / np.sqrt(max(damp_frac * bx, 1e-30)),
                 1.0 / np.sqrt(max(damp_frac * by, 1e-30))), flush=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:            # хендл: savez не допишет .npz
        np.savez(fh, N=np.int64(covs["N"]),
                 damp=np.float64(damp_frac), damp_base=str(damp_base),
                 gu_Sx12=fac["gu"].Sx12, gu_Sx_i=fac["gu"].Sx_i,
                 gu_Sy12=fac["gu"].Sy12, gu_Sy_i=fac["gu"].Sy_i,
                 dn_Sx12=fac["dn"].Sx12, dn_Sx_i=fac["dn"].Sx_i,
                 dn_Sy12=fac["dn"].Sy12, dn_Sy_i=fac["dn"].Sy_i)
    os.replace(tmp, path)
    print("  cov blk%d: факторы отбеливания посчитаны и закэшированы "
          "(%.1fs)" % (i, time.time() - t0), flush=True)
    fac["_N"] = covs["N"]
    return fac


# ----------------------------- дельты ---------------------------------------

def iter_deltas(block, mgu, mdn):
    """yield (dgu (2dff,d) f32, ddn (d,dff) f32) — ΔW_e = W_e − центроида,
    то же определение, что в expert_basis_init (сравнимо с логами joint-v1)."""
    import torch
    from hf_field_transform import _iter_expert_w
    mgu = mgu.numpy().astype(np.float32) if torch.is_tensor(mgu) \
        else np.asarray(mgu, dtype=np.float32)
    mdn = mdn.numpy().astype(np.float32) if torch.is_tensor(mdn) \
        else np.asarray(mdn, dtype=np.float32)
    for wgu, wdn in _iter_expert_w(block):
        yield (wgu.numpy().astype(np.float32) - mgu,
               wdn.numpy().astype(np.float32) - mdn)


def side_dims(geom):
    d, dff = int(geom["d_model"]), int(geom["d_ff"])
    return {"gu": (2 * dff, d), "dn": (d, dff)}


# ----------------------------- сборка T2 + DIA ------------------------------

def build_block_t2(i, block, mgu, mdn, geom, covs, ks, dia_d, whdir,
                   force=False, log_prefix=""):
    """Один двухпроходный прогон блока собирает ВСЁ семейство t2_k* + dia_d*
    (общий базис K = max k). Возвращает строки отчёта."""
    dims = side_dims(geom)
    E = int(geom["n_exp"])
    K = {kind: min(max(max(ks), int(dia_d or 0)), min(*dims[kind]))
         for kind in ("gu", "dn")}

    def deltas():
        return iter_deltas(block, mgu, mdn)

    # --- проход A: скаттеры в отбелённой метрике (gu и dn за один проход)
    SL = {k: np.zeros((dims[k][0], dims[k][0]), dtype=FP) for k in dims}
    SR = {k: np.zeros((dims[k][1], dims[k][1]), dtype=FP) for k in dims}
    for dgu, ddn in deltas():
        for kind, dM in (("gu", dgu), ("dn", ddn)):
            Mw = covs[kind].fw(dM)
            SL[kind] += Mw @ Mw.T
            SR[kind] += Mw.T @ Mw
    basis = {kind: (top_eig(SL[kind], K[kind]), top_eig(SR[kind], K[kind]))
             for kind in dims}
    del SL, SR

    # --- проход B: коэффициенты + capture (норм-трюк, без реконструкции)
    # 13.3: в ТОМ ЖЕ проходе копятся куски для честного (plain Frobenius)
    # capture: B_e = V_Kᵀ dMᵀ U_K (K,K) и ||dM||² по экспертам — все варианты
    # k получаются префиксами (U_k/V_k = первые k столбцов U_K/V_K), без
    # ре-стриминга. raw capture снимает спор «отбеленная метрика = розовые
    # очки»: два числа рядом, а не одна линза.
    Cs = {k: [] for k in dims}
    nrms = {k: [] for k in dims}
    Bs = {k: [] for k in dims}
    d2s = {k: [] for k in dims}
    UK, VK, EF = {}, {}, {}
    raw_ok = {}
    for kind in dims:
        Uw_K, Vw_K = basis[kind]
        UK[kind] = covs[kind].unU(Uw_K).astype(FP)      # (m, K) raw-пространство
        VK[kind] = covs[kind].unV(Vw_K).astype(FP)      # (n, K) raw-пространство
        EF[kind] = (UK[kind].T @ UK[kind], VK[kind].T @ VK[kind])
        # диагностика дорога при гигантском K — тогда остаёмся в отбеленной линзе
        raw_ok[kind] = K[kind] <= 512
    for dgu, ddn in deltas():
        for kind, dM in (("gu", dgu), ("dn", ddn)):
            Uw, Vw = basis[kind]
            Mw = covs[kind].fw(dM)
            Cs[kind].append(Uw.T.astype(FP) @ Mw @ Vw.astype(FP))
            nrms[kind].append(norm2(Mw))
            if raw_ok[kind]:
                A1 = dM.T.astype(FP) @ UK[kind]          # (n, K)
                Bs[kind].append(VK[kind].T @ A1)         # (K, K)
                d2s[kind].append(norm2(dM))

    variants = [("t2_k%d" % k, k, False) for k in ks]
    if dia_d:
        variants.append(("dia_d%d" % int(dia_d), int(dia_d), True))
    rows = []
    for name, k, diag in variants:
        fdir = os.path.join(whdir, name)
        os.makedirs(fdir, exist_ok=True)
        for kind in ("gu", "dn"):
            fpath = os.path.join(fdir, "blk%02d_%s.npz" % (i, kind))
            if os.path.isfile(fpath) and not force:
                z = np.load(fpath)
                caps = z["caps"].astype(np.float64)
                raw_s = (" | честный (Frobenius) %.1f%%"
                         % (100.0 * float(np.mean(z["caps_raw"].astype(np.float64))))
                         if "caps_raw" in z.files
                         else " | честный (Frobenius): нет (пересборка с --force)")
                print("  %s/%s: reused, capture %.1f%%%s"
                      % (name, kind, 100.0 * float(np.mean(caps)), raw_s),
                      flush=True)
                rows.append((name, k, kind, float(np.mean(caps))))
                continue
            Uw, Vw = basis[kind]
            cores = np.stack(
                [np.diag(np.diag(C[:k, :k])) if diag else C[:k, :k]
                 for C in Cs[kind]])
            caps = np.array([np.sum(c * c) / max(nrms[kind][e], 1e-30)
                             for e, c in enumerate(cores)])
            U = covs[kind].unU(Uw[:, :k]).astype(FP)
            V = covs[kind].unV(Vw[:, :k]).astype(FP)
            caps_raw = None
            if raw_ok[kind]:
                Ek = EF[kind][0][:k, :k]
                Fk = EF[kind][1][:k, :k]
                caps_raw = np.array(
                    [(2.0 * float(np.sum(Bs[kind][e][:k, :k] * c.T))
                      - float(np.sum((Ek @ c) * (Fk @ c))))
                     / max(d2s[kind][e], 1e-30)
                     for e, c in enumerate(cores)], dtype=FP)
            tmp = fpath + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, U=U, V=V, cores=cores.astype(FP),
                         caps=caps.astype(FP),
                         meta=np.array([k, i, 1 if diag else 0],
                                       dtype=np.int64),
                         **({"caps_raw": caps_raw}
                            if caps_raw is not None else {}))
            os.replace(tmp, fpath)
            raw_s = (" | честный (Frobenius) %.1f%%"
                     % (100.0 * float(np.mean(caps_raw)),)
                     if caps_raw is not None else "")
            print("  %s delta energy captured at step 0 - %.1f%%  (%s, "
                  "отбелённая метрика)%s"
                  % (kind, 100.0 * float(np.mean(caps)), name, raw_s),
                  flush=True)
            rows.append((name, k, kind, float(np.mean(caps))))
    return rows


# ----------------------------- сборка WHSVD ---------------------------------

def build_block_whsvd(i, block, mgu, mdn, geom, covs, r, r_log, scale,
                      whdir, force=False):
    """Per-expert отбелённый SVD ранга r с масштабом scale (×3.0 - пиковый
    вариант). Свип r_log бесплатен (один SVD на эксперта)."""
    dims = side_dims(geom)
    fdir = os.path.join(whdir, "whsvd_r%d" % r)
    os.makedirs(fdir, exist_ok=True)
    r_list = sorted(set([int(x) for x in r_log] + [int(r)]))
    rows = []
    have_all = all(os.path.isfile(os.path.join(
        fdir, "blk%02d_%s.npz" % (i, kind))) for kind in ("gu", "dn"))
    if have_all and not force:
        for kind in ("gu", "dn"):
            z = np.load(os.path.join(fdir, "blk%02d_%s.npz" % (i, kind)))
            caps = z["caps_e"][:, list(z["r_list"]).index(r)].astype(np.float64)
            print("  whsvd_r%d/%s: reused, capture %.1f%%"
                  % (r, kind, 100.0 * float(np.mean(caps))), flush=True)
            rows.append(("whsvd_r%d" % r, r, kind, float(np.mean(caps))))
        return rows

    for kind in ("gu", "dn"):
        m, n = dims[kind]
        A = None
        caps_by_r = {rl: [] for rl in r_list}
        scales = []
        nrms = []
        e = 0
        for dgu, ddn in iter_deltas(block, mgu, mdn):
            dM = dgu if kind == "gu" else ddn
            Mw = covs[kind].fw(dM)
            U, S, Vt = np.linalg.svd(Mw, full_matrices=False)
            SS = float(np.sum(S * S))
            for rl in r_list:
                caps_by_r[rl].append(
                    float(np.sum(S[:min(rl, len(S))] ** 2)) / max(SS, 1e-30))
            rr = min(r, len(S))
            Aw = U[:, :rr] * np.sqrt(S[:rr])
            Bw = Vt[:rr, :] * np.sqrt(S[:rr])[:, None]
            Rrec = Aw @ Bw
            alpha = np.sum(Mw * Rrec) / max(norm2(Rrec), 1e-30)
            scales.append(alpha)
            nrms.append(norm2(Mw))
            if A is None:
                A = np.zeros((int(geom["n_exp"]), m, rr), dtype=FP)
                B = np.zeros((int(geom["n_exp"]), n, rr), dtype=FP)
            A[e] = (covs[kind].unA(Aw) * (alpha * scale)).astype(FP)
            B[e] = (covs[kind].unB(Bw)).T.astype(FP)   # (n, r); rec = A @ B.T
            e += 1
        caps_e = np.array([[caps_by_r[rl][j] for rl in r_list]
                           for j in range(e)])
        fpath = os.path.join(fdir, "blk%02d_%s.npz" % (i, kind))
        tmp = fpath + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, A=A.astype(FP), B=B.astype(FP),
                     scales=np.array(scales, dtype=FP),
                     caps_e=caps_e.astype(FP), r_list=np.array(r_list),
                     scale=np.float32(scale), meta=np.array([r, i]))
        os.replace(tmp, fpath)
        caps = float(np.mean(caps_e[:, r_list.index(r)]))
        print("  %s delta energy captured at step 0 - %.1f%%  (whsvd_r%d, "
              "scale=%.2f x LS-alpha %.2f, MP-шум при r=%d ~%.1f%%)"
              % (kind, 100.0 * caps, r, scale, float(np.mean(scales)), r,
                 100.0 * mp_expected_capture(r, m, n)), flush=True)
        for rl in r_list:
            print("      r=%-3d capture %.1f%%"
                  % (rl, 100.0 * float(np.mean(caps_by_r[rl]))), flush=True)
        rows.append(("whsvd_r%d" % r, r, kind, caps))
    return rows


# ----------------------------- verify ---------------------------------------

def verify_block(i, block, mgu, mdn, geom, covs, whdir, t2_ks, dia_d, r):
    """Пересчёт capture из СОХРАНЁННЫХ файлов на свежих дельтах (эксперт 0).
    Расхождение с сохранённым caps > 2% относительных -> FAIL."""
    ok = True
    for name, k, diag in ([("t2_k%d" % k, k, False) for k in t2_ks]
                          + ([("dia_d%d" % int(dia_d), int(dia_d), True)]
                             if dia_d else [])):
        for kind in ("gu", "dn"):
            p = os.path.join(whdir, name, "blk%02d_%s.npz" % (i, kind))
            if not os.path.isfile(p):
                continue
            z = np.load(p)
            U, V, cores, caps = z["U"], z["V"], z["cores"], z["caps"]
            dM = next(iter_deltas(block, mgu, mdn))[0 if kind == "gu" else 1]
            Uw = covs[kind].Sy12 @ U.astype(FP)
            Vw = covs[kind].Sx12 @ V.astype(FP)
            Mw = covs[kind].fw(dM)
            G = Uw.T @ Mw @ Vw
            if diag:
                G = np.diag(np.diag(G))
            cap_v = np.sum(G * G) / max(norm2(Mw), 1e-30)
            good = abs(cap_v - float(caps[0])) < \
                2e-2 * max(float(caps[0]), 1e-9) + 1e-4
            ok &= good
            # 13.3: рядом — честный (raw Frobenius) capture этого же банка
            # (эксперт 0) в RAW-пространстве (U, V из файла — уже un-whitened)
            rcap = raw_capture(dM, U, V, G)
            print("  verify %s/%s: из файла %.2f%% vs пересчёт %.2f%% -> %s "
                  "| честный (Frobenius) %.2f%%"
                  % (name, kind, 100 * float(caps[0]), 100 * cap_v,
                     "OK" if good else "FAIL", 100 * rcap), flush=True)
    if r:
        for kind in ("gu", "dn"):
            p = os.path.join(whdir, "whsvd_r%d" % r, "blk%02d_%s.npz" % (i, kind))
            if not os.path.isfile(p):
                continue
            z = np.load(p)
            A, B = z["A"].astype(FP), z["B"].astype(FP)
            r_list = list(z["r_list"])
            cap_ref = float(z["caps_e"][0, r_list.index(r)])
            dM = next(iter_deltas(block, mgu, mdn))[0 if kind == "gu" else 1]
            Mw = covs[kind].fw(dM)
            R = (covs[kind].Sy12 @ A[0]) @ (B[0].T @ covs[kind].Sx12)
            cap_v = norm2(R) / max(norm2(Mw), 1e-30)
            good = abs(cap_v - cap_ref) < 2e-2 * max(cap_ref, 1e-9) + 1e-4
            ok &= good
            print("  verify whsvd_r%d/%s: из файла %.2f%% vs пересчёт %.2f%% "
                  "-> %s" % (r, kind, 100 * cap_ref, 100 * cap_v,
                             "OK" if good else "FAIL"), flush=True)
    return ok


# ----------------------------- проба спайков (AWQ/SpQR) ---------------------

def gauss_topk_share(f):
    """Доля квадратно-энергии в топ-f элементах |N(0,1)| — ориентир для
    чистошумового остатка (амплитудная линза, равные веса): ~14.4% при f=1%,
    ~8.5% при f=0.5%, ~2.3% при f=0.1%."""
    import math
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - f)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return 2.0 * (phi * z + f)


def spike_block(i, block, mgu, mdn, geom, covs, whdir, rank=None,
                fracs=(0.001, 0.005, 0.01, 0.02), force=False):
    """Проба 'spk': собирает ли топ-f ЭЛЕМЕНТОВ остатка после T2 его энергию.

    Функциональный вес элемента (диагональное приближение отбеливания):
    w_ij = Cy_ii · Cx_jj, capture = доля взвешенной энергии остатка в выборе.
    Два выбора: awq = топ-f по |R|·sqrt(w) (активационная навигация —
    функционально-оптимальный кандидат на деплой), raw = топ-f по |R|
    (SpQR-стиль, чистая амплитуда). Per-expert хранится f·m·n·6 байт
    (индекс 4B + fp16 2B). Возвращает строки для report()."""
    dims = side_dims(geom)
    if rank is None:
        raise ValueError("spike_block: нужен rank")
    # T2-банк должен существовать (строим при отсутствии)
    need = [kind for kind in dims if not os.path.isfile(
        os.path.join(whdir, "t2_k%d" % int(rank), "blk%02d_%s.npz" % (i, kind)))]
    if need:
        print("  spk: нет t2_k%d (%s) - строю..." % (rank, ",".join(need)),
              flush=True)
        build_block_t2(i, block=block, mgu=mgu, mdn=mdn, geom=geom, covs=covs,
                       ks=[int(rank)], dia_d=None, whdir=whdir, force=force)
    bank, wdiag = {}, {}
    for kind in dims:
        z = np.load(os.path.join(whdir, "t2_k%d" % int(rank),
                                 "blk%02d_%s.npz" % (i, kind)))
        U, V = z["U"].astype(FP), z["V"].astype(FP)
        bank[kind] = (U, V, z["cores"].astype(FP),
                      float(np.mean(z["caps"].astype(np.float64))))
        # веса функциональной линзы: w_ij = Cy_ii * Cx_jj (диагональное
        # приближение ||Sy12 S Sx12||^2 для элементарной S)
        wdiag[kind] = (
            np.einsum("ik,ik->i", covs[kind].Sy12, covs[kind].Sy12).astype(FP),
            np.einsum("jk,jk->j", covs[kind].Sx12, covs[kind].Sx12).astype(FP))

    st = {k: {"res": [], "awq": {f: [] for f in fracs},
              "raw": {f: [] for f in fracs}, "cnorm": []} for k in dims}
    e_idx = {k: 0 for k in dims}
    for dgu, ddn in iter_deltas(block, mgu, mdn):
        for kind, dM in (("gu", dgu), ("dn", ddn)):
            U, V, cores, cap = bank[kind]
            e = e_idx[kind]
            e_idx[kind] += 1
            if e >= cores.shape[0]:
                break
            G = cores[e]
            Mw = covs[kind].fw(dM)
            e_delta = norm2(Mw)
            if e_delta < 1e-30:
                continue
            Uw = (covs[kind].Sy12 @ U).astype(FP)      # (m,k): один раз/банк
            Vt = (V.T @ covs[kind].Sx12).astype(FP)    # (k,n): один раз/банк
            Rw = Mw - Uw @ G @ Vt                      # отбелённый остаток
            st[kind]["res"].append(norm2(Rw) / e_delta)
            st[kind]["cnorm"].append(float(np.sqrt(norm2(G))))
            R = dM - (U @ G @ V.T).astype(FP)          # raw-остаток (деплой)
            cy, cx = wdiag[kind]
            W = np.sqrt(cy)[:, None] * np.sqrt(cx)[None, :]
            sw = (R * W)
            np.square(sw, out=sw)                      # R^2 * w
            tot_w = float(sw.sum())
            aR = np.abs(R).ravel()
            swf = sw.ravel()
            for f in fracs:
                k_n = max(1, int(round(f * R.size)))
                sel = np.argpartition(swf, -k_n)[-k_n:]     # awq-выбор
                st[kind]["awq"][f].append(
                    float(swf[sel].sum()) / max(tot_w, 1e-30))
                sel = np.argpartition(aR, -k_n)[-k_n:]      # raw-выбор
                st[kind]["raw"][f].append(
                    float(swf[sel].sum()) / max(tot_w, 1e-30))

    rows = []
    for kind in ("gu", "dn"):
        m_k, n_k = dims[kind]
        res = 100.0 * float(np.median(st[kind]["res"]))
        cn = np.array(st[kind]["cnorm"])
        spread = 100.0 * (cn.std() / max(cn.mean(), 1e-30))
        print("  spk/%s: остаток после t2_k%d = %.1f%% энергии дельты "
              "(отбелённая метрика); ||G_e||: разброс %.0f%% по экспертам"
              % (kind, rank, res, spread), flush=True)
        for f in fracs:
            awq = 100.0 * float(np.median(st[kind]["awq"][f]))
            raw = 100.0 * float(np.median(st[kind]["raw"][f]))
            print("    spk/%s f=%.1f%%: awq-выбор %.1f%% | raw-выбор %.1f%% "
                  "| ориентиры: случай %.1f%%, гаусс %.1f%% | хранение "
                  "%.1f KB/эксперт"
                  % (kind, 100 * f, awq, raw, 100 * f,
                     100 * gauss_topk_share(f),
                     f * m_k * n_k * 6 / 1024.0), flush=True)
        rows.append(("spk(awq)", int(rank), kind,
                     float(np.median(st[kind]["awq"][0.01]))))
        cap1 = float(np.median(st[kind]["awq"][0.01]))
        print("    spk/%s ВЕРДИКТ: топливо %s (awq@1%% = %.1f%% vs порог 25%%)"
              % (kind, "ЕСТЬ — UPDATE-14 оправдан" if cap1 >= 0.25
                 else "НЕ подтверждён", 100 * cap1), flush=True)
    return rows


# ----------------------------- пайплайн: T2 -> init -------------------------

def t2_init_dict(i, geom, rank, whdir):
    """init-словарь поля для stage-4 hf_pipeline из t2_k{rank} npz.
    payload вектор->матрица: C (n_exp, rank, rank) ПЛОТНОЕ ядро Tucker-2."""
    import torch
    init = {"svd_ver": WHBANK_T2_VER, "banks": 1, "core": "dense",
            "capture_metric": "whitened Tucker-2 (whbank_build)"}
    caps = {}
    for side in ("gu", "dn"):
        z = np.load(os.path.join(whdir, "t2_k%d" % rank,
                                 "blk%02d_%s.npz" % (i, side)))
        k = int(z["meta"][0])
        if k < rank:
            raise SystemExit("whbank_build: t2_k%d уже, чем rank %d" % (k, rank))
        init["U" + side] = torch.from_numpy(
            np.ascontiguousarray(z["U"][:, :rank], dtype=np.float32))
        init["V" + side] = torch.from_numpy(
            np.ascontiguousarray(z["V"][:, :rank], dtype=np.float32))
        init["C" + side] = torch.from_numpy(
            np.ascontiguousarray(z["cores"][:, :rank, :rank], dtype=np.float32))
        caps[side] = float(np.mean(z["caps"].astype(np.float64)))
    init["capture"] = caps
    if "caps_raw" in z.files:
        caps_raw = {}
        for side in ("gu", "dn"):
            zz = np.load(os.path.join(whdir, "t2_k%d" % rank,
                                      "blk%02d_%s.npz" % (i, side)))
            if "caps_raw" in zz.files:
                caps_raw[side] = float(
                    np.mean(zz["caps_raw"].astype(np.float64)))
        if caps_raw:
            init["capture_raw"] = caps_raw
    return init


def ensure_t2_init(i, block, mgu, mdn, geom, rank, pool_dir, fit_dir,
                   log_prefix=""):
    """Точка входа hf_pipeline (stage-4, --bank-init t2). Коварации и
    t2_k{rank} строятся при отсутствии и кэшируются; возвращает init-словарь.
    ВЫЗЫВАТЬ внутри with_block(i) (для дельт нужен материал блока)."""
    whdir = os.path.join(fit_dir, "whbank")
    os.makedirs(whdir, exist_ok=True)
    covs = ensure_cov(i, mgu, pool_dir, whdir, geom)
    need = any(not os.path.isfile(os.path.join(
        whdir, "t2_k%d" % rank, "blk%02d_%s.npz" % (i, kind)))
        for kind in ("gu", "dn"))
    if need:
        print("    %s whbank: строю t2_k%d (2 стриминг-прохода по экспертам)..."
              % (log_prefix, rank), flush=True)
        build_block_t2(i, block, mgu, mdn, geom, covs, ks=[int(rank)],
                       dia_d=None, whdir=whdir)
    return t2_init_dict(i, geom, int(rank), whdir)


# ----------------------------- отчёт ----------------------------------------

def report(rows_summary, geom, blocks_built):
    """Сводная таблица: вариант x (gu, dn) capture, байты/эксперт, GATE."""
    d, dff, E = int(geom["d_model"]), int(geom["d_ff"]), int(geom["n_exp"])
    dims = side_dims(geom)
    print("\n==== WHBANK REPORT (%d блоков) ====" % len(blocks_built))
    print("%-12s %6s %14s %9s %9s %9s %6s"
          % ("variant", "k/r", "байт/эксперт", "gu", "dn", "MP-шум", "GATE"))
    agg = {}
    for b, kind, name, k, cap in rows_summary:
        agg.setdefault(name, {"k": k}).setdefault(kind, []).append(cap)
    m_gu, n_gu = dims["gu"]
    for name in sorted(agg):
        k = agg[name]["k"]
        if name.startswith("t2"):
            be = k * k * 2 + (m_gu + n_gu) * k * 2 // E
            mpn = mp_expected_capture(k, m_gu, n_gu)
        elif name.startswith("dia"):
            be = k * 2 + (m_gu + n_gu) * k * 2 // E
            mpn = mp_expected_capture(k, m_gu, n_gu)
        else:
            be = k * (m_gu + n_gu) * 2
            mpn = mp_expected_capture(k, m_gu, n_gu)
        cg = [100 * c for c in agg[name].get("gu", [])]
        cd = [100 * c for c in agg[name].get("dn", [])]
        g = np.mean(cg) if cg else float("nan")
        dd = np.mean(cd) if cd else float("nan")
        gate = "PASS" if min(np.nanmin(cg) if cg else 0,
                             np.nanmin(cd) if cd else 0) >= 45.0 else "-"
        print("%-12s %6d %14s %8.1f%% %8.1f%% %8.1f%% %6s"
              % (name, k, human_bytes(be), g, dd, 100 * mpn, gate))
    print("\nGATE: замена оправдана при gu/dn >= 45% минимум по блокам "
          "(мин по столбцам выше).\n"
          "  dia_d*  — якорь паритета с joint-v1 (в логах пайплайна 0.3%)\n"
          "  t2_k*   — контейнерный путь замены: hf_pipeline --bank-init t2\n"
          "  whsvd_r* — потолок качества (per-expert базис; в контейнер "
          "общего базиса не помещается)")


# ----------------------------- модель ---------------------------------------

def load_model(src, gguf=None, local_path=None):
    """Как в пайплайне: BlockStreamRunner (gguf или safetensors) или полный
    HF-лоад для маленьких моделей. Возвращает (model, stream, blocks)."""
    import torch
    from hf_field_transform import find_moe_blocks
    if local_path:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            local_path, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        return model, None, find_moe_blocks(model)
    from hf_stream import BlockStreamRunner
    model = BlockStreamRunner(src, dtype=torch.bfloat16, device="cpu",
                              gguf=gguf, progress=True)
    return model, model, find_moe_blocks(model)


# ----------------------------- CLI ------------------------------------------

def parse_blocks(s):
    if s in ("all", None):
        return None
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run(args):
    import torch
    from contextlib import nullcontext
    from hf_field_transform import block_geometry
    whdir = args.out
    os.makedirs(whdir, exist_ok=True)
    blocks_sel = parse_blocks(args.blocks)
    ks = [int(x) for x in args.k.split(",")]
    r_log = [int(x) for x in args.r_sweep.split(",")] + [args.r]

    if args.report_only:
        import glob as _g
        rows = []
        for f in sorted(_g.glob(os.path.join(whdir, "*", "blk*_*.npz"))):
            name = os.path.basename(os.path.dirname(f))
            kind = "gu" if "_gu" in os.path.basename(f) else "dn"
            z = np.load(f)
            if name.startswith("whsvd"):
                rl = list(z["r_list"])
                cap = float(np.mean(z["caps_e"][:, rl.index(args.r)]
                                    .astype(np.float64)))
                rows.append((-1, kind, name, args.r, cap))
            else:
                rows.append((-1, kind, name, int(z["meta"][0]),
                             float(np.mean(z["caps"].astype(np.float64)))))
        if not rows:
            print("нет npz в %s - нечего репортить" % whdir)
            return
        geom = None                       # геометрия из первого файла банка
        for pat in ("t2_k*", "whsvd_r*"):
            fs = sorted(_g.glob(os.path.join(whdir, pat, "blk*_gu.npz")))
            if fs:
                z = np.load(fs[0])
                if "cores" in z:          # T2: U (m, K), V (n, K)
                    geom = dict(d_model=int(z["V"].shape[0]),
                                d_ff=int(z["U"].shape[0]) // 2,
                                n_exp=int(z["cores"].shape[0]))
                else:                     # WHSVD: A (E, m, r), B (E, n, r)
                    geom = dict(d_model=int(z["B"].shape[1]),
                                d_ff=int(z["A"].shape[1]) // 2,
                                n_exp=int(z["A"].shape[0]))
                break
        if geom is None:
            raise SystemExit("не удалось восстановить геометрию из npz")
        report(rows, geom, ["-"])
        return

    model, stream, blocks = load_model(args.src, gguf=args.gguf,
                                       local_path=args.local_path)
    if blocks_sel is None:
        blocks_sel = list(range(len(blocks)))
    pool_dir = args.pool
    # 13.3 guard: кастомный демпфер меняет ОТБЕЛИВАНИЕ - старые банки устарели
    if args.wh_damp_frac != 1e-3 or args.wh_damp_base != "lmax":
        import glob as _gg
        _have = (_gg.glob(os.path.join(whdir, "t2_k*", "blk*_*.npz"))
                 + _gg.glob(os.path.join(whdir, "whsvd_r*", "blk*_*.npz")))
        if _have and not args.force:
            raise SystemExit(
                "--wh-damp*: демпфер отбеливания изменён - существующие банки "
                "построены со старым отбеливанием. Перестройте их: добавьте "
                "--force (или удалите <out>/t2_k*, <out>/whsvd_r*).")
    rows_summary = []
    verified = True
    for i in blocks_sel:
        ipath = os.path.join(pool_dir, "init_blk%d.pt" % i)
        if not os.path.isfile(ipath):
            raise SystemExit("нет %s - сначала пул пайплайна (stage 2-4)"
                             % ipath)
        ini = torch.load(ipath, map_location="cpu")
        mgu, mdn, geom = ini["mgu"], ini["mdn"], dict(ini["geom"])
        print("=== block %d (%d exp, d=%d, dff=%d) ==="
              % (i, geom["n_exp"], geom["d_model"], geom["d_ff"]), flush=True)
        covs = ensure_cov(i, mgu, pool_dir, whdir, geom, force=args.force,
                          damp_frac=args.wh_damp_frac,
                          damp_base=args.wh_damp_base)
        ctx = stream.with_block(i) if stream is not None else nullcontext()
        with ctx:
            if "t2" in args.variants or "dia" in args.variants:
                rows_summary += build_block_t2(
                    i, block=blocks[i][1], mgu=mgu, mdn=mdn, geom=geom,
                    covs=covs, ks=ks,
                    dia_d=(args.dia if "dia" in args.variants else None),
                    whdir=whdir, force=args.force)
            if "whsvd" in args.variants:
                rows_summary += build_block_whsvd(
                    i, block=blocks[i][1], mgu=mgu, mdn=mdn, geom=geom,
                    covs=covs, r=args.r, r_log=r_log, scale=args.scale,
                    whdir=whdir, force=args.force)
            if "spk" in args.variants:
                rows_summary += spike_block(
                    i, block=blocks[i][1], mgu=mgu, mdn=mdn, geom=geom,
                    covs=covs, whdir=whdir,
                    rank=(args.spk_rank if args.spk_rank else max(ks)),
                    fracs=[float(x) for x in args.spk_fracs.split(",")],
                    force=args.force)
            if args.verify:
                verified &= verify_block(
                    i, block=blocks[i][1], mgu=mgu, mdn=mdn, geom=geom,
                    covs=covs, whdir=whdir, t2_ks=(ks if "t2" in args.variants
                                                   else []),
                    dia_d=(args.dia if "dia" in args.variants else None),
                    r=(args.r if "whsvd" in args.variants else None))
    report(rows_summary, dict(ini["geom"]), blocks_sel)
    if args.verify:
        print("\nVERIFY: %s" % ("OK - все банки совпадают с файлами"
                                if verified else "FAIL - см. строки выше"))
    if stream is not None:
        stream.close()


def main():
    p = __import__("argparse").ArgumentParser(
        description="whbank_build: отбелённые банки дельт (UPDATE-13)")
    p.add_argument("--pool", required=True,
                   help="кэш пайплайна (pairs_blk*.pt + init_blk*.pt)")
    p.add_argument("--src", default=None,
                   help="HF-каталог или light-catalog (стриминг)")
    p.add_argument("--gguf", default=None, help="gguf-путь для стриминга")
    p.add_argument("--local-path", default=None,
                   help="полный HF-лоад (маленькие модели)")
    p.add_argument("--out", default=None,
                   help="куда класть банки (по умолчанию <pool>/whbank)")
    p.add_argument("--blocks", default="0",
                   help="'0', '0-22', '0,3,7' или 'all'")
    p.add_argument("--variants", default="t2,whsvd,dia")
    p.add_argument("--k", default="16,24,32,48,64", help="свип ядер T2")
    p.add_argument("--r", type=int, default=16, help="ранг whsvd")
    p.add_argument("--r-sweep", default="4,8", help="доп. ранги для свипа")
    p.add_argument("--scale", type=float, default=3.0,
                   help="множитель LS-alpha whsvd (пиковый вариант 3.0)")
    p.add_argument("--dia", type=int, default=128,
                   help="размер легаси-диагонали (0 = выкл)")
    p.add_argument("--spk-rank", type=int, default=None,
                   help="ранг T2-банка для spk-пробы (по умолчанию max из --k)")
    p.add_argument("--spk-fracs", default="0.001,0.005,0.01,0.02",
                   help="доли элементов для спайк-пробы")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="пересчитать capture из файлов на свежих дельтах")
    p.add_argument("--force", action="store_true")
    p.add_argument("--wh-damp-frac", type=float, default=1e-3,
                   help="демпфер обратного корня отбеливания: eps = frac·base "
                        "(дефолт 1e-3 — текущее поведение; 0.01·mean-diag = "
                        "внешний рецепт «1%% от средней диагонали»)")
    p.add_argument("--wh-damp-base", default="lmax",
                   choices=["lmax", "mean-diag"],
                   help="база демпфера: λmax (дефолт, при вырождении ЖЕСТЧЕ) "
                        "или mean(diag)")
    args = p.parse_args()
    if not args.report_only and not (args.src or args.local_path):
        p.error("нужен --src или --local-path (или --report-only)")
    if args.out is None:
        args.out = os.path.join(args.pool, "whbank")
    run(args)


if __name__ == "__main__":
    main()
