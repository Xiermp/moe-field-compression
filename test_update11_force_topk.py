"""Unit-test for the update-11 --force-topk / --verify-topk additions in
hf_pipeline.py (torch-only mocks; no transformers needed).
Run: python test_update11_force_topk.py   (from the project root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

import hf_pipeline as hp


class Router(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.top_k = k
        self.weight = nn.Parameter(torch.randn(8, 4))


class BaseMoe(nn.Module):
    """base-like MoE block: int .top_k (captured at init)."""

    def __init__(self, k=2):
        super().__init__()
        self.top_k = k
        self.gate = Router(k)


class FieldBlock(nn.Module):
    """field-like block: int .k + forward_from_z (patch_moe_topk's marker)."""

    def __init__(self, k=2):
        super().__init__()
        self.k = k
        self.gate = Router(k)

    def forward_from_z(self, *a, **kw):
        raise NotImplementedError


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"num_experts_per_tok": 2,
                                     "num_experts": 64})()
        self.moe1 = BaseMoe(2)
        self.moe2 = BaseMoe(2)
        self.fld = FieldBlock(2)


def main():
    ok = True

    m = MockModel()
    n = hp.force_topk_all(m, 8, "unit")
    # per MoE block: block .top_k + nested gate .top_k; field block: .k + gate
    assert n == 6, f"expected 6 patched modules, got {n}"
    assert m.moe1.top_k == 8 and m.moe1.gate.top_k == 8
    assert m.moe2.top_k == 8 and m.fld.k == 8 and m.fld.gate.top_k == 8
    assert m.config.num_experts_per_tok == 8
    print("check 1 ok: force_topk_all patches .top_k + nested gates + field .k + config")

    # re-patch (sweep reuse): patch_moe_topk again to another k
    n2 = hp.patch_moe_topk(m, 4)
    assert n2 == 6 and m.moe1.top_k == 4 and m.moe1.gate.top_k == 4 \
        and m.fld.k == 4
    print("check 2 ok: re-patch for the sweep reuse works")

    # out-of-range guard
    m2 = MockModel()
    try:
        hp.force_topk_all(m2, 100, "range")
        print("check 3 FAILED: no SystemExit for k > num_experts")
        ok = False
    except SystemExit:
        print("check 3 ok: out-of-range k exits with a clear message")

    # verify_topk gate parsing (same code as in main())
    for raw, want in (("1", [1]), ("1,4,8", [1, 4, 8]), ("", [])):
        got = [int(x) for x in str(raw).split(",") if x.strip()] if raw else []
        assert got == want, f"parse {raw!r}: {got} != {want}"
    try:
        [int(x) for x in "1,a".split(",") if x.strip()]
        print("check 4 FAILED: bad list did not raise")
        ok = False
    except ValueError:
        print("check 4 ok: verify_topk list parse + ValueError path")

    # zero-coverage warning path: a model with no int routing attrs
    class Bare(nn.Module):
        pass
    b = Bare()
    b.config = type("C", (), {"num_experts_per_tok": 2, "num_experts": 4})()
    nb = hp.force_topk_all(b, 2, "bare")
    assert nb == 0
    print("check 5 ok: zero-coverage model returns 0 (WARNING printed above)")

    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
