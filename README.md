# NeuronLens

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A sparse autoencoder explorer for language models. NeuronLens finds a model's internal features, names them, and lets you turn one up to see what changes. It currently ships with GPT-2 small, plus an experimental GPT-Neo-125M run.

## What it does

- **Inspect.** See which features fire on each token of a sentence.
- **Features.** Browse and search all 12,288 features, each with its strongest example sentences and its next-word effects.
- **Steer.** Turn a feature up at a chosen strength (or switch it off), or combine two or three features, with a fixed seed, and compare the text with the unsteered version. This is a test of what a feature does, not a fix.
- **Auto-Lab.** Pick a prompt and a goal in plain words. NeuronLens finds the features that fire most on the goal's words, tries up to 5 candidates at six strengths each (plus an unsteered reading), and recommends a strength only when a neighbouring strength also worked. If nothing worked, it says so.
- **Shareable links.** A link reopens the same prompt, seed and strength.
- **Two models.** GPT-2 small, plus an experimental run on GPT-Neo-125M. A model picker switches between them.

## How it works

1. The language model stays frozen (GPT-2 small in the main run). At layer 8, each token is a vector of 768 numbers.
2. A TopK sparse autoencoder (k = 32, 12,288 features) is trained to rebuild that vector from only its 32 strongest features. It was trained on about 100 million tokens of FineWeb-Edu.
3. A scan over 3 million tokens records where each feature fires hardest. Next-word effects come straight from the weights.
4. An LLM (Gemini) names each feature from its strongest sentences and next-word effects. These labels are auto-generated and not hand-checked. Uncertain ones are marked ambiguous.
5. Steering adds a feature's direction at layer 8 while GPT-2 generates. A check then re-encodes the finished text without steering to confirm the feature really rose.

## Measured results

Each model is stated on its own. The two models' numbers are not comparable with each other.

| | GPT-2 small, layer 8 | GPT-Neo-125M, layer 8 (experimental) |
| --- | --- | --- |
| Features, k | 12,288, 32 | 12,288, 32 |
| Training | about 100M tokens | 37.5M tokens |
| Next-word ability kept when layer 8 is replaced | 98.6% | 97.5% |
| Features named / ambiguous | 10,390 / 1,897 | 8,383 / 3,840 |

## Limits

- Names are a guide, not a guarantee. In the recorded Auto-Lab runs for "weather rain", the feature named "weather – atmospheric conditions" (#11299) did not rise at any strength on any of the 3 prompts; "rain – rainforest" (#9821) rose at a single strength on 2 of the 3 prompts, which Auto-Lab does not count as reliable.
- Every result is one prompt and one seed. NeuronLens is a test bench, not a controller.
- Auto-Lab was checked on six goals for each model; the second model offers three of them as buttons. Other goals run but have not been checked.
- Labels are auto-generated. The second model has more ambiguous features, and we have not looked into why.

## Run it

The repository has three parts: `server/` (FastAPI), `web/` (Next.js) and `ml/` (training and indexing). The databases and checkpoints are not in the repository because of their size; see [docs/data-and-models.md](docs/data-and-models.md).

Mock mode needs no GPU, model or checkpoint. It reads a small synthetic database (`contracts/fixtures/sample.db`) and simulates Inspect and Steer, and the app says so on every page:

```
pip install -r server/requirements.txt
uvicorn app.main:app --app-dir server --port 8000
```

In a second terminal:

```
cd web
npm install
npm run dev
```

Open http://localhost:3000. Real mode needs a GPU, a checkpoint and a database built with `ml/`.

- Real-mode start-up: [docs/demo-startup.md](docs/demo-startup.md)
- Server and API: [server/README.md](server/README.md)
- Training, indexing and the second model: [ml/README.md](ml/README.md) and [docs/features/MultiModel.md](docs/features/MultiModel.md)
- Web app: [web/README.md](web/README.md)
- Data and models, and their terms: [docs/data-and-models.md](docs/data-and-models.md)
- Design decisions: [docs/decisions.md](docs/decisions.md)
- Third-party licenses: [THIRD_PARTY.md](THIRD_PARTY.md)

## Repository layout

| Folder | Contents |
| --- | --- |
| `contracts/` | API contract and database schema |
| `server/` | FastAPI server, Auto-Lab check script, tests |
| `web/` | Next.js app |
| `ml/` | Training, evaluation and index-building code |
| `demo/` | Recorded examples and Auto-Lab filler lists |
| `docs/` | Feature cards, decisions, data and model notes |

## Team

- **Prabodh Raj U R** (ML, integration)
- **Mirudull D** (server, auto-labelling)
- **Anand** (web app)

## AI assistance

AI tools used include Claude (coding sessions and drafting text and documentation) and Gemini (automatic feature labelling). The team reviewed the results and ran the code and tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Wording matters in this project: say "tuned", not "improved", and describe steering as a test. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md); report security problems as described in [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
