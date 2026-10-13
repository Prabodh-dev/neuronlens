## What changed and why

Fixes #

## How I tested it
- [ ] Server tests: `python -m pytest server/tests`
- [ ] ML tests, if `ml/` changed: `cd ml && python -m pytest tests --ignore=tests/test_evals.py`
- [ ] Web, if `web/` changed: `npm run typecheck && npm run build`
- [ ] Tried it in mock mode, or in real mode (say which), or n/a

## Checklist
- [ ] No change to `contracts/`, or the change was agreed in an issue
- [ ] No `.db`, `.pt`, `.env`, API key or `node_modules` committed
- [ ] Docs updated where behaviour changed
- [ ] Wording follows CONTRIBUTING.md: steering is a test, no "proves" or "detects", no comparison between models
