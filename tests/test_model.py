"""Step 9 quality gates: shapes, param count, tying, generate, checkpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel


def _tiny_config(**kw) -> MyAIConfig:
    args = dict(
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=80,
        max_sequence_length=64,
    )
    args.update(kw)
    return MyAIConfig(**args)


def _expected_params(c: MyAIConfig) -> int:
    hd = c.hidden_size // c.num_heads
    per_block = (
        2 * c.hidden_size  # two RMSNorms
        + c.hidden_size * c.hidden_size  # q_proj
        + 2 * c.hidden_size * c.num_kv_heads * hd  # k/v projs
        + c.hidden_size * c.hidden_size  # o_proj
        + 3 * c.hidden_size * c.intermediate_size  # swiglu
    )
    return c.vocab_size * c.hidden_size + c.num_layers * per_block + c.hidden_size


def test_logits_shape_and_param_count():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = MyAIModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    logits = model(ids)
    assert logits.shape == (2, 10, cfg.vocab_size) and torch.isfinite(logits).all()
    assert model.num_parameters() == _expected_params(cfg), (
        f"param mismatch: got {model.num_parameters()}, want {_expected_params(cfg)}"
    )


def test_embeddings_tied():
    cfg = _tiny_config()
    model = MyAIModel(cfg)
    assert model.lm_head.weight is model.embedding.weight
    untied = _tiny_config(tie_embeddings=False)
    m2 = MyAIModel(untied)
    assert m2.lm_head.weight is not m2.embedding.weight
    assert m2.num_parameters() == _expected_params(cfg) + cfg.vocab_size * cfg.hidden_size


def test_generate_length_and_determinism():
    torch.manual_seed(1)
    cfg = _tiny_config()
    model = MyAIModel(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    out1 = model.generate(prompt, max_new_tokens=8)
    out2 = model.generate(prompt, max_new_tokens=8)
    assert out1.shape == (1, 13)
    assert torch.equal(out1, out2), "greedy generate must be deterministic"
    assert torch.equal(out1[:, :5], prompt), "prompt prefix must be preserved"


def test_generate_matches_prefill_logits():
    """First generated token must equal argmax of forward logits."""
    torch.manual_seed(2)
    cfg = _tiny_config()
    model = MyAIModel(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    want = model(prompt)[:, -1, :].argmax(dim=-1, keepdim=True)
    got = model.generate(prompt, max_new_tokens=1)[:, -1:]
    assert torch.equal(got, want)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    cfg = _tiny_config()
    model = MyAIModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    before = model(ids)
    p = str(tmp_path / "ckpt.pt")
    model.save_checkpoint(p, step=42, tokens_seen=1000)
    model2, meta = MyAIModel.load_checkpoint(p)
    assert meta["step"] == 42 and meta["tokens_seen"] == 1000
    assert meta["config"]["model_name"] == cfg.model_name
    torch.testing.assert_close(model2(ids), before, rtol=0, atol=0)


def test_prefix_invariant_and_gradients():
    torch.manual_seed(4)
    cfg = _tiny_config()
    model = MyAIModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    y1 = model(ids)
    ids2 = ids.clone()
    ids2[:, 5:] = (ids2[:, 5:] + 17) % cfg.vocab_size
    y2 = model(ids2)
    torch.testing.assert_close(y1[:, :5, :], y2[:, :5, :], rtol=1e-4, atol=1e-5)

    model.train()
    emb = model.embedding.weight
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    model(x).pow(2).sum().backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
