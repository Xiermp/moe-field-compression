# UPDATE-11 — build 10.9-pre (2026-09-06): probe postact + iron test + --field-mode

Диагностический релиз под задачу 29.8 (архитектурный пол ~31%: KL 0.686,
ppl 52.67 при base ppl 40.00). Ничего не ломает: все новые флаги выключены
по умолчанию, прежние прогоны воспроизводятся бит-в-бит.

## Что нового

### 1. probe_capacity.py — плечо `postact`
Решающий тест гипотезы H1 (кросс-члены SwiGLU при смешивании координат ДО
нелинейности) на РЕАЛЬНЫХ весах одного блока:

    python probe_capacity.py --root <cache> --block 20 --arm postact --steps 800

Идентичные параметры поля, но композиция «Вариант A» из внешнего ревью:
общий пре-бейс `x@wgud^T`, низкоранговые дельты и НЕЛИНЕЙНОСТЬ по каждому
выбранному эксперту, выход = `sum_e z_e * FFN_e(x)`. При top_k=1 совпадает
с preact точно. Сравнивать с плечом `cont` на том же блоке.

### 2. hf_pipeline.py — «железный тест» `--verify-topk K`
После обычного verify перезапускает base vs artifact с MoE top_k=K на ОБЕИХ
моделях (роутеры патчатся в рантайме), KL считается LIVE (кеш lp построен
под top-2 и непригоден):

    python hf_pipeline.py <ваши модельные флаги> --stages verify --verify-topk 1

Авто-вердикт по отношению KL(topK)/KL(top2):
  - < 0.6  — разрыв схлопнулся без смеси -> H1 (композиция) -> постакт-режим;
  - > 0.85 — разрыв остался -> H2 (ёмкость дельт) -> плечо bank2;
  - иначе  — смешанная картина.

### 3. hf_pipeline.py — режим `--field-mode {preact,postact}` (по умолч. preact)
Postact теперь СКВОЗНОЙ: фит (fit_sig включает field_mode — смена режима
честно перефитит), refine-проходы, экспорт артефакта (config.field.field_mode)
и рантайм-шаблон modeling_field.py (читает режим из конфига). Параметры поля
ТЕ ЖЕ самые — инициализация/пул/файлы совместимы между режимами.

    python hf_pipeline.py <флаги> --field-mode postact --stages fit,save,verify

Цена: ~+1 down-GEMM на токен (стоимость остаётся ниже оригинального top-2 MoE).

## Решающая матрица 29.8
| Результат пробы | Вывод | Действие |
|---|---|---|
| postact << cont  или KL(top1) << KL(top2) | H1 подтверждена | полный прогон `--field-mode postact` |
| bank2 << cont | ёмкость связывает | two-bank режим (следующий билд) |
| postact ~= cont и bank2 ~= cont | потолок выразимости дельт | per-expert базы / смена параметризации |

Рекомендуемый порядок (всё можно параллельно):
1. `python probe_capacity.py --root <cache> --block 20 --arm postact --steps 800`
   (и `--arm cont` для контроля; якорь blk20: shipped mse 0.16780)
2. `python hf_pipeline.py <флаги> --stages verify --verify-topk 1`
3. По итогам — либо `--field-mode postact` на весь прогон, либо ждём two-bank.

## Проверено
py_compile; test_template_hy_v3 (регресс preact) ALL; test_postact_parity
(рантайм postact == фит postact, max|d|=3e-8; top-1: preact==postact ровно 0);
test_dtype_mismatch ALL; router_guard 11/11; update10_speed 11/11;
lowram 50/50; smoke_probe — все 4 плеча (eval/cont/bank2/postact).
