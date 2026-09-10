# -*- coding: utf-8 -*-
"""
whrank.py — обобщённая ранговая система для дельт MoE-экспертов.

Отвечает на два запроса:
  (1) «двойной банк -> ранговая система с глубиной до сколько понадобится»
  (2) «диагональ c_e -> плотное ядро G_e (Tucker-2)»

Стадии каскада (любая глубина, авто-стоп по capture):
  T2(d, k)          — общий базис d направлений + ПЛОТНОЕ ядро k×k на эксперта
  T2(d, diag=True)  — легаси: общий базис + диагональ (старый bank1, ловушка 0.3%)
  GR(r)             — per-expert жадный ранг-r банк на остатке (старый bank2)

Ключевая математика:
  G_e = U^T ΔW_e V  (closed form, LS-оптимум при фиксированном базисе)
  — забирает ВСЕ cross-коэффициенты общей подпространства, поэтому step-0
    capture растёт с ~0.3% (диагональ) до доли энергии общей подпространства.

Самотест на синтетике (cross-term сигнал + доминирующий per-expert шум):
    python3 whrank.py

Точки подключения к реальному пайплайну — в конце файла.
"""

import numpy as np


# ----------------------------- утилиты --------------------------------------

def norm2(A):
    return float(np.sum(A * A))


def sqrtm_psd(C, eps_frac=1e-6):
    """C^{1/2} для PSD-матрицы."""
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    w = np.maximum(w, 0.0)
    return (V * np.sqrt(w + eps_frac * max(w.max(), 1e-12))) @ V.T


def inv_sqrt(C, eps_frac=1e-3):
    """Регуляризованная C^{-1/2}: eps страхует мелкие направления от усиления шума."""
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    w = np.maximum(w, 0.0)
    eps = eps_frac * max(w.max(), 1e-12)
    return (V * (1.0 / np.sqrt(w + eps))) @ V.T


def mp_expected_capture(r, m, n):
    """Ожидаемый capture топ-r ЧИСТОГО шума (Marchenko-Pastur edge).
    Per-expert greedy на шумовом остатке собирает именно это — поэтому
    стоп для GR привязан к этому ожиданию, а не к фиксированному порогу."""
    p, N = min(m, n), max(m, n)
    return (r / p) * (1.0 + np.sqrt(p / N)) ** 2


def whiten_deltas(deltas, Cx=None, Cy=None):
    """Mw_e = Cy^{1/2} · Δ_e · Cx^{1/2}.
    Cx — входная ковариация слоя из КЭША pair pool (n×n); Cy — выходная (m×m), опц.
    В whitened-метрике capture = функциональный (activation-weighted) capture."""
    Sy = sqrtm_psd(Cy) if Cy is not None else None
    Sx = sqrtm_psd(Cx) if Cx is not None else None
    out = []
    for M in deltas:
        Mw = Sy @ M if Sy is not None else M
        if Sx is not None:
            Mw = Mw @ Sx
        out.append(Mw)
    return out


def unwhiten_T2(Uw, Vw, Cx=None, Cy=None):
    """Базис из whitened-пространства в исходное:
    Δ̂_e = U G_e V^T,  U = Cy^{-1/2} Ū,  V = Cx^{-1/2} V̄."""
    U = inv_sqrt(Cy) @ Uw if Cy is not None else Uw
    V = inv_sqrt(Cx) @ Vw if Cx is not None else Vw
    return U, V


# ----------------------------- стадии ---------------------------------------

class T2:
    """Общий базис + ядро на эксперта (Tucker-2). diag=True — легаси-диагональ."""

    def __init__(self, d, k=None, diag=False, name=None):
        self.diag = diag
        self.k = d if k is None else k
        self.d = d if diag else self.k   # у плотного ядра базис = размеру ядра
        if name is None:
            name = ("DIA d=%d (diag)" % d) if diag else ("T2 core=%dx%d" % (self.k, self.k))
        self.name = name
        self.U = self.V = self.cores = None
        self.cap = None

    def fit(self, R):
        """R — список дельт/остатков [(m,n)] по экспертам. Возвращает остатки.
        Общий базис (HOSVD-init): скаттеры SL=Σ MM^T, SR=Σ M^T M, суммированные
        по экспертам; топ-d собственных направлений каждого — общий U и V."""
        m, n = R[0].shape
        d = min(self.d, m, n)
        SL = np.zeros((m, m))
        SR = np.zeros((n, n))
        for M in R:
            SL += M @ M.T
            SR += M.T @ M
        wL, UL = np.linalg.eigh(SL)
        wR, VR = np.linalg.eigh(SR)
        Uw = UL[:, np.argsort(wL)[::-1][:d]]
        Vw = VR[:, np.argsort(wR)[::-1][:d]]      # ортонормальны в текущей метрике
        cores, caps, recs = [], [], []
        for M in R:
            C = Uw.T @ M @ Vw                     # (k,k): ВСЕ cross-коэффициенты
            if self.diag:
                C = np.diag(np.diag(C))           # легаси: только совпадающие пары
            rec = Uw @ C @ Vw.T
            caps.append(1.0 - norm2(M - rec) / max(norm2(M), 1e-30))
            cores.append(C)
            recs.append(rec)
        self.U, self.V, self.cores = Uw, Vw, cores
        self.cap = float(np.mean(caps))
        return [M - rec for M, rec in zip(R, recs)]

    def export(self, m, n):
        return (m + n) * self.d, (self.d if self.diag else self.k * self.k)


class GR:
    """Per-expert жадный ранг-r банк на остатке (старый bank2)."""

    def __init__(self, r, name=None):
        self.r = r
        self.name = name or ("GR r=%d (per-expert)" % r)
        self.factors = None
        self.cap = None

    def fit(self, R):
        factors, caps, recs = [], [], []
        for M in R:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            r = min(self.r, len(S))
            Aw = U[:, :r] * S[:r]
            Bw = Vt[:r, :]
            rec = Aw @ Bw
            caps.append(1.0 - norm2(M - rec) / max(norm2(M), 1e-30))
            factors.append((Aw, Bw))
            recs.append(rec)
        self.factors, self.cap = factors, float(np.mean(caps))
        return [M - rec for M, rec in zip(R, recs)]

    def export(self, m, n):
        return 0, self.r * (m + n)


# ----------------------------- каскад ---------------------------------------

def fit_cascade(deltas, stages, min_capture=0.05, tag="?"):
    """Каскад стадий по дельтам одного тензора. Глубина любая:
    стадии применяются по очереди к остатку; стоп, когда очередная стадия
    снимает < min_capture (остаток = шум для этой стадии)."""
    R = [M.copy() for M in deltas]
    total0 = np.mean([norm2(M) for M in R])
    kept = []
    for st in stages:
        # порог шума: для GR — MP-edge (иначе greedy вечно собирает шум),
        # для T2 — фиксированный min_capture (шумовой сбор там ~k^2/(mn), копейки)
        thr = min_capture
        if isinstance(st, GR):
            m_, n_ = R[0].shape
            thr = max(min_capture, 1.3 * mp_expected_capture(min(st.r, min(m_, n_)), m_, n_))
        R = st.fit(R)
        kept.append(st)
        print("  delta energy captured at step 0 - %.2f%%  (%s)"
              % (100.0 * st.cap, st.name))
        if st.cap < thr:
            print("  auto-stop: capture %.1f%% < шумового порога %.1f%% (MP-edge) — остаток шум"
                  % (100.0 * st.cap, 100.0 * thr))
            break
    total = 1.0 - np.mean([norm2(M) for M in R]) / max(total0, 1e-30)
    return kept, total


def report_params(stages, m, n, E):
    shared = per = 0
    for st in stages:
        s, p = st.export(m, n)
        shared += s
        per += p
    tot = shared + E * per
    full = E * m * n
    print("  params: shared %d + %d/эксперт -> %d всего = %.2f%% полного "
          "(bf16 ~%.1f КБ/эксперт)" % (shared, per, tot, 100.0 * tot / full, per * 2.0 / 1024.0))


def run_variant(tag, deltas, stages, min_capture=0.05):
    m, n = deltas[0].shape
    print("\n[%s]" % tag)
    kept, total = fit_cascade(deltas, stages, min_capture=min_capture)
    report_params(kept, m, n, len(deltas))
    print("  ИТОГО capture: %.1f%%" % (100.0 * total))
    return kept, total


# ----------------------------- самотест -------------------------------------

def selftest(seed=7):
    """Синтетика, зеркалящая реальную ситуацию:
    shared-структура с ПЛОТНЫМ ядром (cross-члены) + доминирующий per-expert шум."""
    rng = np.random.default_rng(seed)
    m, n, E = 256, 384, 16
    d0 = 8
    U0, _ = np.linalg.qr(rng.standard_normal((m, d0)))
    V0, _ = np.linalg.qr(rng.standard_normal((n, d0)))
    deltas = []
    for e in range(E):
        Ge = rng.standard_normal((d0, d0)) / np.sqrt(d0)               # ПЛОТНОЕ ядро -> cross-члены
        signal = U0 @ Ge @ V0.T                                        # энергия ~ d0 = 8
        noise = rng.standard_normal((m, n)) * np.sqrt(20.0 / (m * n))  # энергия ~20, свой seed
        deltas.append(signal + noise)

    print("=== selftest: cross-term сигнал (энергия 8) + per-expert шум (энергия 20) ===")
    print("    E=%d экспертов, тензор %dx%d, истинное ядро %dx%d ПЛОТНОЕ" % (E, m, n, d0, d0))

    run_variant("легаси: общий базис + диагональ (bank1)", deltas, [T2(32, diag=True)])
    run_variant("Tucker-2: общий базис + плотное ядро 16x16", deltas, [T2(16)])
    run_variant("Tucker-2: общий базис + плотное ядро 32x32", deltas, [T2(32)])
    run_variant("каскад: T2 32x32 -> GR(8) -> GR(8), авто-стоп", deltas, [T2(32), GR(8), GR(8)])


# ==================== ПОДКЛЮЧЕНИЕ К РЕАЛЬНОМУ ПАЙПЛАЙНУ ====================
#
# 1) Дельты одного тензора (gu и dn отдельно), из стримингового загрузчика:
#      deltas = [dequant(W_art_e) - W_base_e for e in range(E)]        # (m, n)
#
# 2) Ковариации — ИЗ КЭША pair pool (23 блока, ничего не пересобираем):
#      Cx = E[x x^T]   (n, n) — входы слоя
#      Cy = E[y y^T]   (m, m) — опционально, взвешивание выходов
#
# 3) Прогнать варианты в отбелённой метрике (capture = функциональный):
#      Mw = whiten_deltas(deltas, Cx=Cx, Cy=Cy)
#      kept, total = fit_cascade(Mw, [T2(32)], min_capture=0.05, tag="gu")
#
# 4) Экспорт в контейнер (полной смены формата квантования НЕ нужно):
#      U, V = unwhiten_T2(kept[0].U, kept[0].V, Cx=Cx, Cy=Cy)
#      W_e = W0_e + U[:, :k] @ G_e @ V[:, :k].T
#      + (опц.) GR-факторы:  A_e = inv_sqrt(Cy) @ Aw ;  B_e = Bw @ inv_sqrt(Cx)
#      Каскад схлопывается: Σ_i A_i B_i^T = [A_1..A_N][B_1..B_N]^T — один банк.
#
# 5) Гейт успеха (строка в логе):
#      "delta energy captured at step 0 - gu >= 45%"  ->  фит ~50 шагов, IRON eval.
#
# 6) Проверка генератора артефакта: шум Ω_e — свой seed на КАЖДОГО эксперта,
#    иначе дельты гасят друг друга. В самом T2-фите гашение невозможно:
#    ядро G_e считается по своей дельте независимо.
# ============================================================================

if __name__ == "__main__":
    selftest()
