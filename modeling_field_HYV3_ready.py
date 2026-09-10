# ГОТОВЫЙ ФАЙЛ - см. шапку ниже.
"""Рантайм "поле-движка" — ОБЫЧНАЯ HF-модель, загружается стандартно:
    AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    (рекомендуется dtype="bfloat16", low_cpu_mem_usage=True)
Сгенерирован hf_pipeline.py. Требует transformers>=5 (SparseMoeBlock.forward
-> Tensor). # version: 2026-09-05.4 - dtype-mismatch fix: v5-роутеры возвращают
fp32-логиты/веса (upcast внутри), а чекпойнт бывает смешанным (бэкбон fp32 из
GGUF-декванта + поле bf16): matmul'ы (@) в torch не смешивают dtype и падали
(float != BFloat16). Поле теперь один раз выравнивает свои тензоры под dtype
вычислений хоста и приводит z к нему же (см. UPDATE-10, 10.6).

Идея: явных весов экспертов нет. Есть "поле": центроиды (w*bd) + низкоранговые
факторы U,V и координаты C. "Сид движения" c(z) = z @ C вычисляется из роутера:
    z = topk(gate(x))
    gu(x) = x@wGud^T + (x@Vgu * cgu)@Ugu^T      # fused gate+up
    h = silu(gate_part) * up_part
    y  = h@wDnd^T + (h@Vdn * cdn)@Udn^T         # down
Один проход FFN вместо top-k проходов (FLOPs/2 при top-2, ~x8 при top-8).

hy_v3 (NanoColibri и родня, DeepSeek-стиль роутинга), 2026-09-05.3:
  - роутер берёт порог выбора эксперт-скор аргументом forward(x, bias)
    (transformers 5.16+; на старых билдах bias живёт внутри роутера - вызов
    подстраивается по сигнатуре автоматически); bias приезжает из чекпойнта
    буфером <layer>.mlp.e_score_correction_bias;
  - блок несёт always-on shared_experts: их веса сохранены в артефакте
    (<layer>.mlp.shared_experts.*) и прибавляются к выходу поля в fp32 -
    ровно как в базовой модели и как при фите (таргет фита = блок минус
    shared-ветка, поле учит только routed-часть);
  - у v5-роутеров топк-веса УЖЕ нормализованы и умножены на scaling-фактор
    внутри роутера - поле берёт их как есть (то же делает фит-сторона);
  - dtype живёт по правилу "как у входа": при первом forward все плавающие
    тензоры модуля приводятся к dtype скрытых состояний хоста (смешанный
    чекпойнт или fp32-логиты v5-роутера больше не роняют matmul'ы).

banks (2026-09-08, UPDATE-12):
  - config.field.banks >= 2: второй банк координат (U2gu,V2gu,C2gu) и
    (U2dn,V2dn,C2dn) - дельта-пространство удваивается (фикс H2). Входит
    аддитивно, при нулевых bank2-весах выход бит-в-бит равен одному банку;
    артефакты без ключа banks работают как раньше.
"""
import inspect

import torch
import torch.nn as nn
from transformers import HYV3ForCausalLM
from transformers.models.hy_v3.modeling_hy_v3 import HYV3TopKRouter
from transformers.activations import ACT2FN


class FieldSparseMoe(nn.Module):
    """Замена SparseMoeBlock: тот же роутер, эксперты собираются из поля."""

    def __init__(self, config):
        super().__init__()
        fi = config.field
        d, dff, r = fi["d_model"], fi["d_ff"], fi["rank"]
        self._field_dtype = None                     # выравнивание ещё не делали
        self.top_k = int(fi["top_k"])
        self.field_mode = str(fi.get("field_mode", "preact"))  # 10.9
        self.banks = max(1, int(fi.get("banks", 1)))           # UPDATE-12
        self.core = str(fi.get("core", "diag"))                # UPDATE-13
        self.norm_topk = bool(fi.get("norm_topk"))
        self.gate = HYV3TopKRouter(config)                 # роутер не трогаем (вес из базы)
        self.act_fn = ACT2FN[config.hidden_act]
        # hy_v3: bias выбора в 5.16+ - аргумент forward(x, bias), на старых
        # билдах он внутри роутера. Определяем по фактической сигнатуре хоста
        # (+ подсказка router_kind="sigmoid_bias" из геометрии фита); сам
        # буфер заполняется из чекпойнта при загрузке.
        try:
            takes_bias = "e_score_correction_bias" in inspect.signature(
                type(self.gate).forward).parameters
        except (TypeError, ValueError):   # экзотический хост - смотрим флаг
            takes_bias = False
        self._gate_takes_bias = takes_bias or \
            str(fi.get("router_kind", "") or "") == "sigmoid_bias"
        if self._gate_takes_bias:
            self.register_buffer("e_score_correction_bias",
                                 torch.zeros(fi["n_exp"]))
        dffs = int(fi.get("dff_shexp", 0) or 0)    # hy_v3: always-on ветка
        if dffs:
            se = nn.Module()                       # ключи как у базовой модели:
            se.gate_proj = nn.Linear(d, dffs, bias=False)   # shared_experts.*
            se.up_proj = nn.Linear(d, dffs, bias=False)     # gate/up/down_proj
            se.down_proj = nn.Linear(dffs, d, bias=False)
            self.shared_experts = se
        for nm, out, inp in (("gu", 2 * dff, d), ("dn", d, dff)):
            self.register_parameter(f"w{nm}d", nn.Parameter(torch.zeros(out, inp)))
            self.register_parameter(f"U{nm}", nn.Parameter(torch.zeros(out, r)))
            self.register_parameter(f"V{nm}", nn.Parameter(torch.zeros(inp, r)))
        c_shape = (fi["n_exp"], r, r) if self.core == "dense" \
            else (fi["n_exp"], r)                    # UPDATE-13: Tucker-2
        self.Cgu = nn.Parameter(torch.zeros(c_shape))
        self.Cdn = nn.Parameter(torch.zeros(c_shape))
        if self.banks >= 2:                        # UPDATE-12: второй банк
            for nm, out, inp in (("gu", 2 * dff, d), ("dn", d, dff)):
                self.register_parameter(
                    f"U2{nm}", nn.Parameter(torch.zeros(out, r)))
                self.register_parameter(
                    f"V2{nm}", nn.Parameter(torch.zeros(inp, r)))
            self.C2gu = nn.Parameter(torch.zeros(fi["n_exp"], r))
            self.C2dn = nn.Parameter(torch.zeros(fi["n_exp"], r))
        self.u_mode = str(fi.get("u_mode", "none"))    # 13.5: du-банк на Udn
        self.u_rank = int(fi.get("u_rank", 4))
        if self.u_mode not in ("none", "rank1", "rank4", "full"):
            raise ValueError(f"unknown u_mode: {self.u_mode}")
        if self.u_mode in ("rank1", "rank4"):
            a = 1 if self.u_mode == "rank1" else self.u_rank
            self.duAdn = nn.Parameter(torch.zeros(fi["n_exp"], r, a))
            self.duBdn = nn.Parameter(torch.zeros(fi["n_exp"], d, a))
        elif self.u_mode == "full":
            self.duEdn = nn.Parameter(torch.zeros(fi["n_exp"], r, d))

    def _align_field_dtype(self, dtype):
        """Один раз приводим все плавающие тензоры модуля к dtype вычислений
        хоста. Зачем: (1) v5-роутеры отдают fp32-логиты/веса независимо от
        dtype модели; (2) чекпойнт, загруженный "как сохранён" (dtype=None),
        бывает смешанным - бэкбон fp32 (GGUF-деквант) + поле bf16 (ради
        размера). Сложения в torch приводят типы сами, а matmul'ы (@) - нет,
        отсюда float != BFloat16. Вверх (bf16->fp32) приводим без потерь,
        вниз - только если весь хост уже в этом dtype."""
        for _, p in self.named_parameters(recurse=True):
            if p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype)
        if getattr(self, "_gate_takes_bias", False):
            b = self.e_score_correction_bias
            if b.is_floating_point() and b.dtype != dtype:
                self.e_score_correction_bias = b.to(dtype)
        self._field_dtype = dtype

    def forward(self, hidden_states):
        B, T, d = hidden_states.shape
        x = hidden_states.reshape(-1, d)
        if self._field_dtype != x.dtype:           # один раз на запуск
            self._align_field_dtype(x.dtype)
        if self._gate_takes_bias:                  # hy_v3: sigmoid + bias-порог
            try:
                out = self.gate(x, self.e_score_correction_bias)
            except TypeError:                      # старый хост: bias внутри
                out = self.gate(x)                 # роутера - зовём по-старому
            logits, scores, idx = out[0], out[1], out[2]
        else:
            gout = self.gate(x)
            if isinstance(gout, (tuple, list)):    # v5-роутер: (logits, scores, idx)
                logits, scores, idx = gout[0], gout[1], gout[2]
            else:                                  # обычный Linear-роутер
                logits = gout
                probs = torch.softmax(logits.float(), dim=-1)
                scores, idx = torch.topk(probs, self.top_k, dim=-1)
                if self.norm_topk:
                    scores = scores / scores.sum(-1, keepdim=True)
        z = torch.zeros_like(logits).scatter_(-1, idx, scores).to(x.dtype)
        if self.field_mode == "postact":           # 10.9: per-expert ветки
            zt, ei = z.topk(min(self.top_k, z.shape[-1]), dim=-1)
            gu0 = x @ self.wgud.t()                # общий пре-бейс
            y = None
            for j in range(zt.shape[-1]):
                if self.core == "dense":           # UPDATE-13: ядро эксперта
                    gu = gu0 + torch.einsum(
                        "bkl,bl->bk", self.Cgu[ei[:, j]],
                        x @ self.Vgu) @ self.Ugu.t()
                else:
                    cgu = self.Cgu[ei[:, j]]
                    gu = gu0 + (x @ self.Vgu * cgu) @ self.Ugu.t()
                if self.banks >= 2:                # UPDATE-12: банк 2
                    gu = gu + (x @ self.V2gu
                               * self.C2gu[ei[:, j]]) @ self.U2gu.t()
                g, u = gu.chunk(2, dim=-1)
                h = self.act_fn(g) * u             # НЕЛИНЕЙНОСТЬ ПО ЭКСПЕРТУ
                if self.core == "dense":           # UPDATE-13
                    yj = zt[:, j:j + 1] * (
                        h @ self.wdnd.t()
                        + torch.einsum("bkl,bl->bk", self.Cdn[ei[:, j]],
                                       h @ self.Vdn) @ self.Udn.t())
                else:
                    cdn = self.Cdn[ei[:, j]]
                    yj = zt[:, j:j + 1] * (
                        h @ self.wdnd.t()
                        + (h @ self.Vdn * cdn) @ self.Udn.t())
                if self.banks >= 2:                # UPDATE-12: банк 2
                    yj = yj + zt[:, j:j + 1] * (
                        (h @ self.V2dn * self.C2dn[ei[:, j]]) @ self.U2dn.t())
                if self.u_mode == "full":          # 13.5: du-банк
                    yj = yj + zt[:, j:j + 1] * torch.matmul(
                        (h @ self.Vdn).unsqueeze(1),
                        self.duEdn[ei[:, j]]).squeeze(1)
                elif self.u_mode in ("rank1", "rank4"):
                    yj = yj + zt[:, j:j + 1] * torch.matmul(
                        torch.matmul((h @ self.Vdn).unsqueeze(1),
                                     self.duAdn[ei[:, j]]),
                        self.duBdn[ei[:, j]].mT).squeeze(1)
                y = yj if y is None else y + yj
        else:                                      # preact: c(z) до silu
            if self.core == "dense":               # UPDATE-13: смесь ядер
                gu = x @ self.wgud.t() + torch.einsum(
                    "tkl,tl->tk",
                    torch.einsum("te,ekl->tkl", z, self.Cgu),
                    x @ self.Vgu) @ self.Ugu.t()
            else:
                cgu = z @ self.Cgu                 # сид движения (T,r)
                gu = x @ self.wgud.t() + (x @ self.Vgu * cgu) @ self.Ugu.t()
            if self.banks >= 2:                    # UPDATE-12: банк 2
                gu = gu + (x @ self.V2gu * (z @ self.C2gu)) @ self.U2gu.t()
            g, u = gu.chunk(2, dim=-1)
            h = self.act_fn(g) * u
            if self.core == "dense":               # UPDATE-13
                y = h @ self.wdnd.t() + torch.einsum(
                    "tkl,tl->tk",
                    torch.einsum("te,ekl->tkl", z, self.Cdn),
                    h @ self.Vdn) @ self.Udn.t()
            else:
                cdn = z @ self.Cdn
                y = h @ self.wdnd.t() + (h @ self.Vdn * cdn) @ self.Udn.t()
            if self.banks >= 2:                    # UPDATE-12: банк 2
                y = y + (h @ self.V2dn * (z @ self.C2dn)) @ self.U2dn.t()
            if self.u_mode != "none":              # 13.5: du-банк поверх preact
                zt, ei = z.topk(min(self.top_k, z.shape[-1]), dim=-1)
                hv = h @ self.Vdn                  # h общий -> hv один
                for j in range(zt.shape[-1]):
                    if self.u_mode == "full":
                        y = y + zt[:, j:j + 1] * torch.matmul(
                            hv.unsqueeze(1), self.duEdn[ei[:, j]]).squeeze(1)
                    else:
                        y = y + zt[:, j:j + 1] * torch.matmul(
                            torch.matmul(hv.unsqueeze(1),
                                         self.duAdn[ei[:, j]]),
                            self.duBdn[ei[:, j]].mT).squeeze(1)
        if hasattr(self, "shared_experts"):        # hy_v3: shared-ветка, fp32
            se = self.shared_experts               # combine - как в базе
            ys = se.down_proj(self.act_fn(se.gate_proj(x)) * se.up_proj(x))
            y = (y.float() + ys.float()).to(y.dtype)
        return y.view(B, T, -1)


class FieldForCausalLM(HYV3ForCausalLM):
    """Базовая архитектура, где каждый MoE-блок заменён на поле."""

    def __init__(self, config):
        super().__init__(config)
        n = 0
        for name, mod in list(self.named_modules()):
            if hasattr(mod, "experts") and hasattr(mod, "gate"):
                parent = self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self
                setattr(parent, name.rsplit(".", 1)[-1], FieldSparseMoe(config))
                n += 1
        expected = config.field["n_layers"]
        if n != expected:
            raise RuntimeError(f"заменили {n} MoE-блоков, ожидалось {expected}")
