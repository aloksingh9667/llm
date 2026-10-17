# User Preferences (persistent)

- **Language**: Always reply in **English** (not Hinglish/Hindi), in every session.
- Modes: keep momentum — don't ask step-by-step approvals; proceed autonomously;
  be honest in verdicts ("make it perfect" = no fake wins).
- Long Kaggle GPU jobs: split into legged parts; monitor continuously; retry on
  quota without stopping; run should not stop until the target is reached
  (current target: 2.0B tokens on MyAI-100M).
- Secrets stay out of the repo: Neon DB URL in local gitignored `.env`;
  Kaggle creds in a private Kaggle dataset (rotate after the 2B run);
  HF token via Kaggle Secrets, and rotated when pasted in chat.
