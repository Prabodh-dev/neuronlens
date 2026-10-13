# Demo data

Recorded files that the mock server, the tests and the web app read. Nothing here is edited by hand.

| File | What it is | Read by |
| --- | --- | --- |
| `autolab_examples.json` | Auto-Lab searches and sweeps recorded on a real GPT-2 server: 6 goals × 3 prompts, seed 42 | mock server (`server/app/services/mock.py`), `server/tests/test_autolab.py` |
| `autolab_fillers.json` | GPT-2 features that come back from the word search for many unrelated words; Auto-Lab drops them | `server/scripts/autolab_check.py`, copied into the web app by `web/scripts/copy-fillers.mjs` |
| `autolab_fillers_gpt-neo-125m.json` | The same list for GPT-Neo-125M | same |
| `phase4_examples.json` | Recorded steering-check runs (X1 to X4, W5) | mock server, `server/tests/` |

They are written by `server/scripts/autolab_check.py` and `ml/scripts/phase4_demo.py`; see [server/README.md](../server/README.md) and [ml/README.md](../ml/README.md). Recordings are specific to one model and one checkpoint.
