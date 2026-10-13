# Third-party software

The direct dependencies of NeuronLens and their licenses. NeuronLens itself is MIT ([LICENSE](LICENSE)). Models and data have their own terms: see [docs/data-and-models.md](docs/data-and-models.md).

Licenses come from each package's metadata (PyPI for Python, `node_modules/*/package.json` for npm, checked with `license-checker-rseidelsohn`). Metadata can be wrong or out of date, and the Python requirements are not pinned, so check the license of the version you install. Transitive dependencies are not listed except where noted under Flags.

## Python

| Package | Used by | License |
| --- | --- | --- |
| torch | ml, server (real mode) | Wheel metadata lists Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0 and MIT for the bundled components; see the CUDA note below |
| transformers | ml | Apache-2.0 |
| datasets | ml | Apache-2.0 |
| numpy | ml | BSD-3-Clause (the wheel also bundles 0BSD, MIT, Zlib and CC0-1.0 components) |
| pyyaml | ml | MIT |
| tensorboard | ml | Apache-2.0 |
| fastapi | server | MIT |
| uvicorn | server | BSD-3-Clause |
| pydantic | server | MIT |
| pytest | server and ml tests | MIT |
| httpx | server tests | BSD-3-Clause |

Imported by the code but not listed in the main requirements files (`google-genai` has its own file):

| Package | Used by | License |
| --- | --- | --- |
| pyarrow | `ml/neuronlens/activations.py` (installed as a dependency of `datasets`) | Apache-2.0 |
| huggingface_hub | `ml/scripts/compare_published.py` (installed as a dependency of `transformers`) | Apache-2.0 |
| google-genai | `ml/scripts/auto_label.py`, only when you label with Gemini; listed in `ml/requirements-labelling.txt` | Apache-2.0 |
| umap-learn | `ml/scripts/build_map.py`, optional | BSD (PyPI gives no more detail than "BSD"; check the license file of the version you install) |

## Web (`web/package.json`)

| Package | Kind | License |
| --- | --- | --- |
| next 16.4.0 | dependency | MIT |
| react 19.3.0 | dependency | MIT |
| react-dom 19.3.0 | dependency | MIT |
| @types/node ^22 | dev | MIT |
| @types/react ^19 | dev | MIT |
| @types/react-dom ^19 | dev | MIT |
| openapi-typescript 7.13.0 | dev | MIT |
| typescript 5.9.3 | dev | Apache-2.0 |

## Flags

No direct dependency has a copyleft or unknown license. Items that need a look:

- **LGPL-3.0-or-later, transitive (npm).** `@img/sharp-libvips-linux-x64` (and `@img/sharp-wasm32`, which is Apache-2.0 AND LGPL-3.0-or-later AND MIT) come in through `next`'s optional image library `sharp`. They are prebuilt binaries that the app does not modify. The app does not use Next's image optimiser, but the packages are installed. Check whether LGPL matters for how you distribute a build.
- **CC-BY-4.0, transitive (npm).** `caniuse-lite` is browser-support data, used at build time. It requires attribution.
- **Python-2.0, transitive (npm).** `argparse@2.0.1`, a build-time dependency. Permissive.
- **PyTorch CUDA wheels.** The default `pip install torch` for Linux pulls `nvidia-*` CUDA packages under NVIDIA's own license terms, not an open-source license. The CPU build from `https://download.pytorch.org/whl/cpu` does not. Read the NVIDIA terms before redistributing an environment or container image that includes them. This was not reviewed here.
- **`web/package.json` had no `license` field**, so license checkers reported the web app as UNLICENSED. It now says MIT.
- **Unpinned Python requirements.** A new release can change a license.
