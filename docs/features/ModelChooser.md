# ModelChooser: run the app on another model
Owner: Anand (UI); Prabodh (the second model and its server)
Branch: dev-anand (side experiment; merged only if it fully works)
User sees: with one model's server running, the app looks as it did before, with no chooser. When more than one server answers, the status pill in the top bar (Mock, Live, Sample data) gets a ▾ and opens a menu titled "Run every page on". Each row is a model: its name, layer, number of features and mode, all read from that server's /health, and an "experimental" tag for a model marked experimental. Choosing one reloads the same page on that model. Then every page (Features, Inspect, Steer, Auto-Lab, Microscope, Metrics, the home page) calls that model's server. The footer has one "Model" line, "Model: <model_name> · layer <layer> · <d_sae> features · k=<k>", read from /health; a part the server did not send is left out. On an experimental model the "experimental" tag also shows in the status pill and next to that line, with one more line under it: "Smaller training run than GPT-2, so expect weaker features." No numbers about the second model are typed into the app. A chosen model whose server stops answering stays in the menu as "not responding", so there is a way back.
Model picker: opening `/` with no `?model=` while two or more servers answer shows a "Pick a model." screen first, in the landing page's look: one card per model with its name, an "experimental" tag where marked, a short description and the details its server reports (layer, features, fire per token, width, data kind; a row the server did not send is hidden). Choosing a card opens the landing page on that model (`/?model=<id>`, also for the default model, which is how the home page knows a choice was made). The landing page then has an "About this model" block with the same description. With one server answering, `/` is the landing page as before. A link that names a model, and every page other than `/`, skips the picker. Descriptions are in `web/src/lib/model-info.ts`, keyed by model id and with no numbers in them; nothing ranks or compares the models.
Endpoint / data used: no new endpoint. Each model is its own server with its own database, answering the same endpoints. `GET /api/v1/health` on every model when the app loads, with a 2.5 s limit; a model that does not answer is not offered.

How it works
- `web/src/lib/models.ts` lists the models: `{ id, label, apiBase, experimental }`.
  - `gpt2` on `http://127.0.0.1:8000`, the default. `NEXT_PUBLIC_API_BASE`, when set, still overrides it.
  - `neo` on `http://127.0.0.1:8001`, experimental. Its optional `fillers` entry (`gpt-neo-125m`) is the name in its filler file.
  - `label` is for people reading the file. The screen shows `model_name` and `hook_layer` from /health. `modelLabel()` in the same file only makes the name friendly: `gpt2` reads "GPT-2 small", any other name is shown as the server returned it. Every call, the health probe included, goes to `<apiBase>/api/v1/...`, built in one place (`apiV1`). Cross-port calls need no new header: reads send none that triggers a CORS preflight, and each server must list the page's origin in `NEURONLENS_CORS_ORIGINS` (default `http://localhost:3000`).
- The model is in the address as `?model=<id>`.
  - No `?model=` (or an unknown id) means the default model, so existing links, the demo and shared links made before the chooser open as they did.
  - The model is read once when the page loads and stays fixed until it reloads. Choosing another model reloads the page, so one page never mixes two servers' answers or cached results.
  - Links inside the app do not carry `?model=`; for a non-default model the app puts it back into the address after every page change.
  - Copy link (S5) adds `model=<id>` for a non-default model.
- Choosing a model keeps the page but clears its settings, because a feature number means a different feature on each model. `/features/123` goes to `/features`.
- Pages are prerendered without the address, so anything that depends on the model waits until the page has loaded in the browser. A second-model page never flashes the default model's presets first.

Keyed per model (a model without an entry gets none of another model's)
- `web/src/lib/model-data.ts`: picks and recorded runs, all from GPT-2.
  - the home page's "one neuron, many ideas" features
  - the landing tour's stops and its Steering chapter
  - the Steer page's saved runs
  - the Combine tab's recorded pair and its note
  - Without an entry: the home page and tour use the model's strongest named features, the tour has no Steering chapter, and the Steer and Combine one-click buttons are not shown.
- `web/src/lib/autolab-presets.ts`: Auto-Lab's prompt and goal buttons, the "about N seconds" estimate, and whether the ✓ thresholds were tuned on that model.
  - Without an entry there are no buttons: the page says no prompts or goals were tested on this model, asks for a typed prompt and goal, and runs them live.
  - The time estimate is left out, because that model was not timed.
  - The foot says the ✓ thresholds were tuned on another model's runs and are untested on this one.
- Filler lists: `web/public/autolab-fillers/<name>.json`, where `<name>` is the model's `fillers` entry in `models.ts`, or its id when it has none (`gpt2` for the default model).
  - `web/scripts/copy-fillers.mjs` copies them unchanged from `demo/autolab_fillers.json` (the default model) and `demo/autolab_fillers_<name>.json`. It runs before `npm run dev` and `npm run build`, and only reads `demo/`.
  - Auto-Lab and the Features page ("Your words") load the current model's list.
  - With no file yet, nothing is dropped, and both screens say the model has no filler list yet. When Prabodh adds `demo/autolab_fillers_gpt-neo-125m.json`, the next `npm run dev` or `npm run build` picks it up with no code change.
- Not keyed: example sentences that are only inputs, run live on whichever model is chosen: the home page sentence and the Inspect pair. Saved steering examples come from each server's own database.

Real demo example (feature ids, prompt, strength): none for the second model; nothing was recorded on it. The default model keeps all of its recordings unchanged.
Honest limit shown on screen:
- the "experimental" tag on the second model
- "No prompts or goals were tested on this model yet…" in Auto-Lab
- "this model has no filler list yet"
- "The ✓ thresholds were tuned on another model's runs and are untested on this one."
- The app makes no comparison between the models.

How to test without the second model: start the usual server on 8000, and a second mock server on 8001 from the same code, for example on the sample database:

```
NEURONLENS_MODE=mock NEURONLENS_DB=contracts/fixtures/sample.db uvicorn app.main:app --app-dir server --port 8001
```

The chooser, the experimental tag and `?model=` switching then show. Both test servers report the model name of their own database. To see two different names, point the second one at a copy of the sample database with `meta.model_name` changed, kept outside the repository. A mock server also replays GPT-2's recorded Auto-Lab runs for the recorded goal words, so on such a test server use other words.

Status: working on dev-anand (typecheck and build pass; tested with mock servers on 8000 and 8001)
