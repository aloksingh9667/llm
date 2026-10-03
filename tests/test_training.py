"""Step 11 quality gates: schedule shape, overfit, ckpt resume, eval."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from myai.data.packed import PackedLMDataset, pack_tokens
from myai.foundation.config import MyAIConfig
from myai.foundation.model import MyAIModel
from myai.training.loop import (
    TrainConfig,
    build_optimizer,
    eval_loss,
    load_state,
    lr_at,
    save_state,
    train_step,
)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = MyAIConfig(
        vocab_size=32, hidden_size=24, num_layers=1, num_heads=4,
        num_kv_heads=2, intermediate_size=48, max_sequence_length=32,
    )
    return MyAIModel(cfg)


def _overfit_loader():
    ids = (list(range(30)) * 8)  # trivial repeating pattern
    pairs = pack_tokens(ids, seq_len=16)
    ds = PackedLMDataset(pairs)
    return DataLoader(ds, batch_size=4, shuffle=True)


def test_lr_schedule_shape():
    cfg = TrainConfig(learning_rate=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == 1.0 * 1 / 10
    assert lr_at(9, cfg) == 1.0
    mid = lr_at(60, cfg)
    assert 0.1 < mid < 1.0
    assert lr_at(109, cfg) >= 0.1
    assert lr_at(200, cfg) == 0.1
    # Monotone non-increasing after warmup.
    lrs = [lr_at(s, cfg) for s in range(10, 110)]
    assert all(b <= a + 1e-9 for a, b in zip(lrs, lrs[1:]))


def test_loss_falls_on_learnable_pattern():
    model = _tiny_model()
    tcfg = TrainConfig(learning_rate=3e-3, warmup_steps=5, max_steps=60)
    opt = build_optimizer(model, tcfg)
    loader = _overfit_loader()
    first, last = None, None
    for step, batch in enumerate(_cycle(loader, 40)):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        loss = train_step(model, opt, batch, grad_clip=1.0)
        first = loss if first is None else first
        last = loss
    assert last < first, f"loss did not fall: {first} -> {last}"
    assert last < 2.0, f"40 steps should nearly memorize, got {last}"


def _cycle(loader, n):
    while True:
        for b in loader:
            yield b
            n -= 1
            if n <= 0:
                return


def test_checkpoint_resume_continues(tmp_path):
    model = _tiny_model()
    tcfg = TrainConfig(learning_rate=1e-3)
    opt = build_optimizer(model, tcfg)
    loader = _overfit_loader()
    batch = next(iter(loader))
    train_step(model, opt, batch, grad_clip=1.0)
    p = str(tmp_path / "resume.pt")
    save_state(p, model, opt, step=7, tokens_seen=700, cfg=tcfg)

    model2 = _tiny_model(seed=999)  # different init proves load works
    opt2 = build_optimizer(model2, tcfg)
    meta = load_state(p, model2, opt2)
    assert meta["step"] == 7 and meta["tokens_seen"] == 700
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b), "weights must match after resume"


def test_eval_loss_explicit_device():
    """eval_loss must honor an explicit device (the Kaggle cuda crash)."""
    model = _tiny_model().eval()
    loader = _overfit_loader()
    assert eval_loss(model, loader, device="cpu") > 0


def test_exact_resume_matches_continuous(tmp_path):
    """Audit section 37: 1->60 continuous must equal 1->30 + resume + 31->60."""
    import itertools

    from myai.training.loop import lr_at, set_seed

    torch.use_deterministic_algorithms(True)
    try:
        def run(n_steps, ckpt_in=None, ckpt_out=None, seed=7):
            set_seed(seed)
            model = _tiny_model(seed=seed)
            tcfg = TrainConfig(learning_rate=2e-3, warmup_steps=5, max_steps=60)
            opt = build_optimizer(model, tcfg)
            start, skip = 0, 0
            if ckpt_in:
                meta = load_state(ckpt_in, model, opt)
                start = meta["step"]
                # KEY: resume continues the data position; a fresh cycle
                # would replay batches 0.. and silently diverge (proven by
                # batch fingerprinting during debugging).
                skip = meta.get("data_pos", {}).get("batches_consumed", 0)
            loader = DataLoader(_overfit_loader().dataset, batch_size=4, shuffle=False)
            cycle = itertools.cycle(loader)
            for _ in range(skip):
                next(cycle)
            last = None
            for step in range(start, n_steps):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, tcfg)
                batch = next(cycle)
                last = train_step(model, opt, batch, grad_clip=1.0)
            if ckpt_out:
                save_state(ckpt_out, model, opt, n_steps, n_steps * 64, tcfg,
                           data_pos={"batches_consumed": n_steps, "shuffle": False})
            return model, opt, last

        mA, _, lossA = run(60)
        p = str(tmp_path / "half.pt")
        run(30, ckpt_out=p)
        mB, _, lossB = run(60, ckpt_in=p)
        assert abs(lossA - lossB) < 1e-4, f"{lossA} vs {lossB}"
        for a, b in zip(mA.parameters(), mB.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    finally:
        torch.use_deterministic_algorithms(False)


def test_init_from_loads_weights_only(tmp_path):
    """--init-from semantics: weights transfer, optimizer stays fresh."""
    from myai.foundation.model import MyAIModel

    torch.manual_seed(0)
    src = _tiny_model(seed=0)
    tcfg = TrainConfig(learning_rate=1e-3)
    opt = build_optimizer(src, tcfg)
    p = str(tmp_path / "base.pt")
    save_state(p, src, opt, step=10, tokens_seen=100, cfg=tcfg)
    dst, meta = MyAIModel.load_checkpoint(p)
    for a, b in zip(src.parameters(), dst.parameters()):
        assert torch.equal(a, b)
    assert meta["step"] == 10  # metadata travels; caller resets schedule


def test_eval_loss_matches_manual():
    model = _tiny_model().eval()
    loader = _overfit_loader()
    got = eval_loss(model, loader)
    assert got > 0 and got == got  # finite, not NaN
    import torch.nn.functional as F

    batch = next(iter(loader))
    manual = F.cross_entropy(
        model(batch["input_ids"]).view(-1, 32), batch["labels"].view(-1)
    ).item()
    assert abs(got - manual) < 1.0  # mean over batches ~= single batch
