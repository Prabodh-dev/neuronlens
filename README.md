# NeuronLens

A sparse autoencoder explorer for GPT-2 small. NeuronLens finds the internal features of a language model, names them, and lets you turn one up to see what changes.

Built by Team Yvl at HACKERING 2.0 (Open Innovation track). Released under the MIT license.

## What it does

- **Inspect.** See which features fire on each token of a sentence.
- **Features.** Browse and search all 12,288 features, each with its strongest example sentences and its next-word effects.
- **Steer.** Turn one feature up at a chosen strength, with a fixed seed, and compare the text with the unsteered version. This is a test of what a feature does, not a fix.
- **Auto-Lab.** Pick a prompt and a goal in plain words. NeuronLens searches the feature names, tries the best candidates at seven strengths each, and recommends a strength only when a neighbouring strength also worked. If nothing worked, it says so.
- **Shareable links.** A link reopens the same prompt, seed and strength.
- **Two models.** GPT-2 small, plus an experimental run on GPT-Neo-125M. A model picker switches between them.

## How it works

1. GPT-2 small stays frozen. At layer 8, each token is a vector of 768 numbers.
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

- Names are a guide, not a guarantee. For example, a feature named "weather" did not respond when steered, while one named "rain" did.
- Every result is one prompt and one seed. NeuronLens is a test bench, not a controller.
- Auto-Lab was checked on six goals for GPT-2 and three for the second model. Other goals work but have not been checked.
- Labels are auto-generated. The second model has more ambiguous features, and we have not looked into why.

## Run it

The repository has three parts: `server/` (FastAPI), `web/` (Next.js) and `ml/` (training and indexing). The databases and checkpoints are not in the repository because of their size.

- Back end and front end start-up: [docs/demo-startup.md](docs/demo-startup.md)
- Server and API: [server/README.md](server/README.md)
- Training, indexing and the second model: [ml/README.md](ml/README.md) and [docs/features/MultiModel.md](docs/features/MultiModel.md)
- Web app: [web/README.md](web/README.md)
- Design decisions: [docs/decisions.md](docs/decisions.md)

## Repository layout

| Folder | Contents |
| --- | --- |
| `contracts/` | API contract and database schema |
| `server/` | FastAPI server, Auto-Lab check script, tests |
| `web/` | Next.js app |
| `ml/` | Training, evaluation and index-building code |
| `demo/` | Recorded examples and Auto-Lab filler lists |
| `docs/` | Feature cards, decisions and planning notes |

## Team

- **Prabodh Raj U R** (ML, integration)
- **Mirudull D** (server, auto-labelling)
- **Anand** (web app)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Wording matters in this project: say "tuned", not "improved", and describe steering as a test.

## License

MIT. See [LICENSE](LICENSE).
