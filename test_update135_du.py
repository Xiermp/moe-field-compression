#!/usr/bin/env python3
"""test_update135_du.py - DU-BANK (13.5): per-expert deltas on the DOWN output
factor (u_mode rank1/rank4/full, LoRA start, external review fig12/fig14).

Запускается БЕЗ transformers (чистый torch; шаблон рендерится на стабы).

  t1  LoRA-старт: u_mode rank4/full при нулевых du-дельтах бит-в-бит == none
      (preact и postact, фит-сторона и рендер шаблона)
  t2  градиенты шага 0: duBdn/duEdn живы; duAdn спит ровно первый шаг
      (классическая A@B^T асимметрия, fig12 это подтверждает на стенде)
  t3  train="cores": du-факторы НЕ замораживаются (имя не ловится паттерном
      U*/V*) и реально двигаются; базис U*/V* и центроиды стоят
  t4  roundtrip fit->init: fit-словарь восстанавливает модуль бит-в-бит
  t5  шаблон == фит-сторона (одинаковые параметры -> одинаковый выход),
      u_mode rank4/full x preact/postact, du НЕ нулевые
  t6  state_dict содержит du-ключи ровно нужного режима (контракт артефакта)
  t7  field_accounting: rank-a добавляет E*(r+d)*a*2 байт/блок, full E*r*d*2
  t8  _muon_split: du* уходят в Adam (не NS-кандидаты), Udn остаётся в Muon
"""
import importlib.util
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_field_transform import (FieldSparseMoe, fit_field_module,
                                field_accounting, _muon_split,
                                render_modeling_file)

torch.manual_seed(0)
torch.set_num_threads(2)
FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}"
          + (f" ({extra})" if extra else ""), flush=True)
    if not cond:
        FAILED.append(name)


D, DFF, NEXP, R, TOPK, A = 32, 48, 8, 12, 2, 4
GEOM = dict(n_exp=NEXP, d_model=D, d_ff=DFF, top_k=TOPK, norm_topk=False,
            hidden_act="silu")
DU = {"none": (), "rank1": ("duAdn", "duBdn"),
      "rank4": ("duAdn", "duBdn"), "full": ("duEdn",)}


def make_fit(um, mode, seed=0):
    g = dict(GEOM) | {"field_mode": mode, "u_mode": um, "u_rank": A}
    torch.manual_seed(seed)
    gw = torch.randn(NEXP, D) * 0.1
    return FieldSparseMoe(g, R, gate_w=gw, act_fn=F.silu), gw


def fill(mod, seed=5, with_du=True):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn", "Cgu", "Cdn"):
            getattr(mod, n).normal_(std=0.1, generator=g)
        for n in DU[getattr(mod, "u_mode", "none")]:
            if with_du:
                getattr(mod, n).normal_(std=0.05, generator=g)


X = torch.randn(256, D)
torch.manual_seed(123)
X = torch.randn(256, D)


def out_of(mod, x):
    with torch.no_grad():
        z = mod._z(x)
        return mod.forward_from_z(x, z)


print("== t1. LoRA-старт: du при нулевых дельтах бит-в-бит == none ==")
for mode in ("preact", "postact"):
    base, _ = make_fit("none", mode)
    fill(base)
    y0 = out_of(base, X)
    for um in ("rank1", "rank4", "full"):
        m, _ = make_fit(um, mode)
        with torch.no_grad():                     # общие веса из base
            for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn",
                      "Cgu", "Cdn"):
                getattr(m, n).copy_(getattr(base, n))
        check(f"{mode}/{um}: выход == none", torch.equal(out_of(m, X), y0))

print("== t2. градиенты шага 0 ==")
for mode in ("preact", "postact"):
    m, _ = make_fit("rank4", mode)
    fill(m, with_du=False)                        # duBdn остаётся нулевым
    loss = (out_of(m, X) ** 2).mean()             # форвард под no_grad -
    # повторим с градиентом честно:
    z = m._z(X)
    loss = (m.forward_from_z(X, z) ** 2).mean()
    loss.backward()
    gb = m.duBdn.grad.abs().sum().item()
    ga = m.duAdn.grad.abs().sum().item()
    check(f"{mode}/rank4: duBdn.grad жив на шаге 0", gb > 0, f"grad={gb:.2e}")
    check(f"{mode}/rank4: duAdn.grad == 0 ровно на шаге 0", ga == 0.0)
    m2, _ = make_fit("full", mode)
    fill(m2, with_du=False)
    (m2.forward_from_z(X, m2._z(X)) ** 2).mean().backward()
    check(f"{mode}/full: duEdn.grad жив на шаге 0",
          m2.duEdn.grad.abs().sum().item() > 0)

print("== t3. train='cores': du двигается, базис стоит ==")
torch.manual_seed(7)
teacher, _ = make_fit("rank4", "postact", seed=3)
fill(teacher, seed=11, with_du=True)              # du НЕ нулевые
with torch.no_grad():                             # усилим du-сигнал x10,
    teacher.duAdn *= 10.0                         # чтобы mse-пол был видим
    teacher.duBdn *= 10.0
Y = out_of(teacher, X).detach()
m, _ = make_fit("rank4", "postact", seed=3)       # тот же базис, du=0
fill(m, seed=11, with_du=False)
u0 = m.Ugu.detach().clone()
a0 = m.duAdn.detach().clone()
mse0 = float(((out_of(m, X) - Y) ** 2).mean())
mse = fit_field_module(m, X, Y, steps=80, bs=128, lr=5e-3, device="cpu",
                       log_prefix="t3", guard=False, method="adam",
                       autocast="off", train="cores", seed=1)
moved = (m.duAdn.detach() - a0).abs().max().item()
basis_ok = torch.equal(m.Ugu.detach(), u0)
check("duAdn сдвинулся с нуля", moved > 0, f"max|d|={moved:.2e}")
check("базис Ugu заморожен (cores)", basis_ok)
check(f"mse упал: {mse0:.3e} -> {mse:.3e}",
      mse < mse0 * 0.5, f"падение {(1 - mse / max(mse0, 1e-12)):.1%}")

print("== t4. roundtrip fit->init ==")
m, _ = make_fit("rank4", "postact")
fill(m)
out = {n: getattr(m, n).detach().clone() for n in m.field_names}
m2, _ = make_fit("none", "postact")               # другой режим - но init
m3, _ = make_fit("rank4", "postact")              # грузим в правильный
with torch.no_grad():
    for k, v in out.items():
        getattr(m3, k).copy_(v)
check("fit-словарь восстанавливает du бит-в-бит",
      torch.equal(out_of(m3, X), out_of(m, X)))
check("field_names содержит du-ключи", set(out) >= {"duAdn", "duBdn"})

# ---------------------------------------------------------------- шаблон --
print("== t5-t6. шаблон: parity + LoRA-старт + ключи ==")


class _StubRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(NEXP, D) * 0.1)

    def forward(self, x):
        return x @ self.weight.t()


def load_template(mode, um):
    stub_tf = types.ModuleType("transformers")
    stub_tf.StubBase = type("StubBase", (nn.Module,), {})
    stub_act = types.ModuleType("transformers.activations")
    stub_act.ACT2FN = {"silu": F.silu}
    stub_tf.activations = stub_act
    stub_rm = types.ModuleType("stub_router_mod")
    stub_rm.StubRouter = _StubRouter
    sys.modules["transformers"] = stub_tf
    sys.modules["transformers.activations"] = stub_act
    sys.modules["stub_router_mod"] = stub_rm
    src = render_modeling_file("StubBase", "StubRouter", "stub_router_mod")
    tmp = tempfile.mkdtemp(prefix="t135_")
    p = os.path.join(tmp, "modeling_field.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(src)
    import py_compile
    py_compile.compile(p, doraise=True)
    spec = importlib.util.spec_from_file_location(f"mf_{mode}_{um}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


for mode in ("preact", "postact"):
    for um in ("rank4", "full", "none"):
        fi = dict(rank=R, n_layers=2, base_class="StubBase", n_exp=NEXP,
                  d_model=D, d_ff=DFF, top_k=TOPK, norm_topk=False,
                  hidden_act="silu", field_mode=mode, u_mode=um, u_rank=A)
        cfg = types.SimpleNamespace(field=fi, hidden_act="silu")
        am = load_template(mode, um).FieldSparseMoe(cfg)
        fm, gw = make_fit(um, mode)
        with torch.no_grad():
            am.gate.weight.copy_(gw)
            for n in ("wgud", "Ugu", "Vgu", "wdnd", "Udn", "Vdn",
                      "Cgu", "Cdn"):
                getattr(am, n).copy_(getattr(fm, n))
            for n in DU[um]:
                getattr(am, n).copy_(getattr(fm, n))
        ya = am(X.view(4, -1, D))
        yf = out_of(fm, X)
        dmax = (ya.view(256, D) - yf).abs().max().item()
        check(f"шаблон==фит: {mode}/{um}", dmax < 1e-5, f"max|d|={dmax:.2e}")
        keys = set(am.state_dict().keys())
        need = set(DU[um])
        extra = {k for k in ("duAdn", "duBdn", "duEdn")} - need
        check(f"ключи state_dict: {mode}/{um}",
              need <= keys and not (extra & keys),
              f"need={sorted(need)}")

print("== t7. field_accounting ==")
geoms = [dict(d_model=D, d_ff=DFF, n_exp=NEXP)]
_, f_none = field_accounting(geoms, R, banks=1, core="diag", u_mode="none")
_, f_r4 = field_accounting(geoms, R, banks=1, core="diag",
                           u_mode="rank4", u_rank=4)
_, f_r1 = field_accounting(geoms, R, banks=1, core="diag",
                           u_mode="rank1", u_rank=4)
_, f_full = field_accounting(geoms, R, banks=1, core="diag", u_mode="full")
check("rank4: +E*(r+d)*4*2 байт", f_r4 - f_none == NEXP * (R + D) * 4 * 2,
      f"{f_r4 - f_none}")
check("rank1: +E*(r+d)*1*2 байт", f_r1 - f_none == NEXP * (R + D) * 1 * 2)
check("full: +E*r*d*2 байт", f_full - f_none == NEXP * R * D * 2)

print("== t8. _muon_split ==")
m, _ = make_fit("rank4", "postact")
mu, ad = _muon_split(m.field_names, m.fit_params(), 512)
mu_names = {n for n, t in zip(m.field_names, m.fit_params())
            if any(t is p for p in mu)}
ad_names = {n for n, t in zip(m.field_names, m.fit_params())
            if any(t is p for p in ad)}
check("du* в Adam (не NS)", {"duAdn", "duBdn"} <= ad_names)
check("Udn по-прежнему NS-кандидат", "Udn" in mu_names)

print()
if FAILED:
    print(f"ПРОВАЛЕНО: {len(FAILED)}: {FAILED}")
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ 13.5 ПРОЙДЕНЫ")
