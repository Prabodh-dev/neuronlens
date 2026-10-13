# Changelog

## 0.1.0

Initial version. No tag or release has been made for it yet.

- TopK sparse autoencoder code (k = 32, 12,288 features) for layer 8 of GPT-2 small: training, evaluation (FVU, L0, dead features, cross-entropy recovered), feature indexing into a SQLite database, and a comparison script against OpenAI's published GPT-2 SAE.
- FastAPI server (`/api/v1`) with a mock mode on a synthetic sample database and a real mode that loads the checkpoint and runs the model: features, metrics, Inspect, Steer, saved runs and a feature search by words.
- Steering checks: re-reading the feature on the finished text, side-by-side comparison with the unsteered text, a sweep over strengths, a prompt-only baseline, a degeneration flag, and steering two or three features together.
- Web app with Microscope, Features, Inspect, Steer (including Combine), Auto-Lab and Metrics pages, shareable links for features and steering results, and a model picker.
- Auto-Lab: pick a prompt and a goal in words; it tries candidate features at six strengths and reports whether any worked.
- Automatic feature naming with Gemini (`ml/scripts/auto_label.py`); names are not checked by hand and uncertain features are marked ambiguous.
- Experimental second model, GPT-Neo-125M, trained and indexed with the same recipe (37.5 million tokens).
- Docs: feature cards, decisions log, data and model notes, contributing guide, code of conduct, security policy.
- CI for the server tests, the ML tests on CPU (without `ml/tests/test_evals.py`), and the web typecheck and build.
