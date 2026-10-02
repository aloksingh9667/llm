"""CPU smoke test: 1-step dummy training + checkpoint round-trip.

Validates base env before any real MyAI-20M work (doc section 43 item 8-10).
"""
import os
import torch
import torch.nn as nn

ckpt_path = os.path.join("checkpoints", "smoke-test.pt")


def main():
    torch.manual_seed(1337)
    model = nn.Sequential(nn.Embedding(128, 32), nn.Linear(32, 128))
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    x = torch.randint(0, 128, (2, 16))
    logits = model[1](model[0](x))
    loss = nn.functional.cross_entropy(logits.view(-1, 128), x.view(-1))
    print(f"loss_before={loss.item():.4f}")
    opt.zero_grad()
    loss.backward()
    opt.step()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 1}, ckpt_path)
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert loaded["step"] == 1, "checkpoint step mismatch"
    print(f"OK smoke test passed, checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
