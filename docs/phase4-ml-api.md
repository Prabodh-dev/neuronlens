# Phase 4: ML functions for the new endpoints

For Mirdul. These are the ML-side functions behind the Phase 4 steering endpoints (X1 to X4). They live in `ml/neuronlens/steering_checks.py`. This file lists the exact inputs and outputs so the fields can be added to `contracts/` and `server/` by the people who own them. Nothing in `contracts/` was edited.

Steering is a test of what a feature does, not a fix. A steered output is never presented as an improvement. The steering maths is unchanged: amplify adds `strength x max_activation x decoder direction` at every position except position 0; ablate subtracts the feature's own activation.

The example JSON below is real model output from one GPU run on GPT-2 small: feature 13 (label "upside – upside down", `max_activation` 21.4249), `seed` 42, `max_new_tokens` 60. Long texts are shortened with "..." for readability. The texts, strengths, the divergence index and the degenerate flag shown are from that run. Fields that were not recorded in the run are left out of the examples and each example says which. A real response contains every field listed in the tables. The only text in the examples that did not come from the model is the `reason` wording in X2 and the fixed `percent_changed_raw_note` string.

## Common rules

- **Same seed for both runs.** Every function that compares runs generates the baseline and the steered text with the same `seed`, so a difference comes from the intervention. Sampling settings are fixed in `ml/neuronlens/steering.py` (temperature 0.8, top-p 0.95).
- **How to call.** Each function takes the loaded runtime `rt`, then the inputs listed per function. It needs `rt.generate(prompt, max_new_tokens, seed, steering)`, `rt.d_sae`, and for X1 also `rt.feature_activations(token_ids, feature_id)`. `neuronlens.runtime.Runtime` has both. `server/app/services/real.py` (`RealRuntime`) wraps `Runtime` and does not forward them yet, so it needs `generate` and `feature_activations` (under its lock) and `d_sae` passed through. Mock mode needs its own stand-in; the functions do not run without a `generate`.
- **Where `max_activation` comes from.** The server passes it, read from `features.max_activation` for the feature, exactly as `/steer` does today.
- **Text fields.** `baseline_text`, `steered_text`, `text` hold the full decoded text, prompt included (same format as `/steer`). Token lists hold one display string per token from `textdecode.token_pieces`: a character split across tokens appears whole on its last token and as `""` on the earlier ones, so no list contains U+FFFD for valid UTF-8.
- **Ablate.** `strength` is ignored for ablate and is reported as `0.0`.
- **Time.** Each generation is a full model run. X1 and X3 run two, X2 runs one plus one per strength, X4 runs one. The functions hold the model for that whole time, so the server should call them under the same lock as `/steer`. For rough times see Timing.

### Errors (all functions)

Bad input raises `neuronlens.steering_checks.SteeringCheckError`, a `ValueError` with a `code` attribute that uses the API's existing codes. The server should map it to the matching error response.

| `code` | When |
| --- | --- |
| `INVALID_INPUT` | empty or blank `prompt`; `mode` not `amplify` or `ablate`; `strength` not a finite number from 0 to `STRENGTH_MAX` (10); `max_new_tokens` not an integer from 1 to 120; `seed` not an integer from 0 to 2147483647; `max_activation` negative or not finite; `max_activation` 0 with `amplify` (a dead feature); a sweep with no strengths or more than `MAX_SWEEP_POINTS` (12) |
| `FEATURE_NOT_FOUND` | `feature_id` negative or not below `rt.d_sae` |

Not handled in ML (the server keeps doing them): text length (`TEXT_TOO_LONG`), reading `max_activation` from the database, dead-feature checks against the database, `MODEL_NOT_LOADED`.

### Strength range

Decision: the cap on `SteerRequest.strength` stays at 3 and the contract does not change. In GPU runs on the demo feature, the useful range was about 0.5 to 1, and repetition loops started around 1 to 1.5. That is one feature on GPT-2 small, so other features may differ. The default sweep is therefore `[0, 0.25, 0.5, 0.75, 1, 1.5, 2]`, all within the cap. The ML functions still accept `strength` from 0 to 10 (`STRENGTH_MAX`); the server keeps rejecting values above 3 on its own endpoints.

---

## X1. Steering check

`steering_check(rt, prompt, feature_id, mode, strength, max_activation, max_new_tokens=60, seed=42)`

Baseline and steered text for one feature, a flag for whether they differ, and the target feature's activation on each text.

| Input | Type | Notes |
| --- | --- | --- |
| `prompt` | str | not blank |
| `feature_id` | int | 0 to `d_sae - 1` |
| `mode` | str | `"amplify"` or `"ablate"` |
| `strength` | float | 0 to 10; ignored for ablate |
| `max_activation` | float | from the database |
| `max_new_tokens` | int | 1 to 120, default 60 |
| `seed` | int | 0 to 2147483647, default 42 |

| Output field | Type | Notes |
| --- | --- | --- |
| `feature_id` | int | echoed |
| `mode` | str | echoed |
| `strength` | float | echoed; `0.0` for ablate |
| `seed` | int | echoed |
| `max_new_tokens` | int | echoed |
| `baseline_text` | str | no steering, prompt included |
| `steered_text` | str | steered, prompt included |
| `differs` | bool | `baseline_text != steered_text` |
| `target_activation_baseline` | float | the target feature's maximum activation over the new tokens of the baseline text |
| `target_activation_steered` | float | the same for the steered text |
| `target_activation_baseline_mean` | float | mean activation over the new tokens of the baseline text |
| `target_activation_steered_mean` | float | mean activation over the new tokens of the steered text |
| `activation_change` | float | `(target_activation_steered - target_activation_baseline) / max_activation`, so it is comparable across features; `0.0` when `max_activation` is 0 (possible for ablate) |
| `activation_verdict` | str | `"higher"`, `"lower"` or `"about the same"`, from `activation_change` (see below) |

Amplify at strength 0 adds nothing, so `differs` should be `false` (a control).

**How the activation fields are measured.** Both texts are generated; then their token ids are fed back through the model with no steering hook attached, along the same SAE path that Inspect uses (`Runtime.feature_activations` and `Runtime.inspect` share one helper), and the target feature's activation is read at every new token. The prompt tokens and position 0 are left out, and the ids are used as generated, without re-tokenising the text. The maximum and the mean are taken over the new tokens. The feature is not measured inside the steered run, which would be circular for amplify.

`activation_verdict` is `"higher"` when `activation_change` is at least `ACTIVATION_SAME_FRACTION` (0.05), `"lower"` when it is at most minus 0.05, and `"about the same"` when its absolute value is below 0.05. The 0.05 is a starting value and has not been calibrated on real outputs. When the two texts are identical, the baseline numbers are reused, so `activation_change` is `0.0` and the verdict is `"about the same"`.

The limit: this reads the feature on the finished text, re-encoded without steering. It says nothing about whether the output is better, and a higher or lower reading is not a quality judgement. The maximum is a single token's value, so it can move while the mean barely does.

The sweep (X2) does not include activations yet.

The call was `steering_check(rt, "The acrobat hung", 13, "amplify", 0.5, 21.4249, 60, 42)`:

```json
{
  "feature_id": 13,
  "mode": "amplify",
  "strength": 0.5,
  "seed": 42,
  "max_new_tokens": 60,
  "baseline_text": "The acrobat hung his head and walked toward the door, but had no idea where he was going.\n\n\"Well, it's all fine,\" he said.\n\n\"Let's go,\" ...",
  "steered_text": "The acrobat hung his head and walked up to the roof and closed the window, only to have it fall down on him. The roof was flipped down, ...",
  "differs": true
}
```

A second real example, same feature, strength and seed, prompt `He turned the picture`:

```json
{
  "feature_id": 13,
  "mode": "amplify",
  "strength": 0.5,
  "seed": 42,
  "max_new_tokens": 60,
  "baseline_text": "He turned the picture over to me and said, \"You are a good guy, sir.\"...",
  "steered_text": "He turned the picture down to 10 seconds, but the camera still had the image in it...",
  "differs": true
}
```

These two real examples were recorded before the activation fields existed, so they do not show them. No real activation values have been recorded yet.

Example values for the new fields. These numbers are example values, not measured, and are not tied to the texts above (they assume a `max_activation` of 20.0, also an example value):

```json
{
  "target_activation_baseline": 2.3,
  "target_activation_steered": 11.8,
  "target_activation_baseline_mean": 0.4,
  "target_activation_steered_mean": 3.1,
  "activation_change": 0.475,
  "activation_verdict": "higher"
}
```

### Limits of the activation check

Observed on the GPU laptop, seed 42: one feature each and a few prompts. These are observations, not general rules. The numbers are the maximum activation over the new tokens, `target_activation_baseline` to `target_activation_steered`.

- A feature that already fires near its maximum cannot show an increase. Feature 1863 on `The lawyers argued before the` had a baseline activation of 18.57 (its `max_activation` is about 19.7), so amplify reads "about the same".
- For ablate, the check re-encodes the finished text, so if the concept's word is still in the text the feature still fires. Feature 1863 with ablate read "about the same" on two prompts (18.57 to 18.57, and 18.89 to 18.89).
- Results depend on the prompt and the strength. Feature 3446 on `The band walked on stage and` went from 0.00 to 12.44 at strength 0.5, but read 0.00 at 1.0, where the text drifted to "video". Feature 1863 on `She opened the letter and read` went from 0.00 to 17.27 at 0.5 and read 0.00 at 1.0.
- The check says whether the feature appears in the output text, not whether the output is better.

---

## X3. Seed-matched compare

`seed_matched_compare(rt, prompt, feature_id, mode, strength, max_activation, max_new_tokens=60, seed=42)`

Runs the baseline and the steered generation with the same seed and compares the new tokens one by one. It also has an opt-in `with_activations` flag (off by default, used by the demo script) that adds the six X1 activation fields; they are not part of the documented response. Same inputs and same errors as X1. The pure part is `compare_generations(baseline, steered)` (two `Generation` objects).

Indices are 0-based and count newly generated tokens only (not the prompt). A position past the end of the shorter output counts as different.

| Output field | Type | Notes |
| --- | --- | --- |
| `feature_id`, `mode`, `strength`, `seed`, `max_new_tokens` | as in X1 | echoed |
| `baseline_text`, `steered_text` | str | full text, prompt included |
| `prompt_tokens` | int | number of prompt tokens |
| `baseline_new_tokens`, `steered_new_tokens` | int | number of generated tokens in each run |
| `diverges` | bool | `false` when the two outputs are identical |
| `first_divergence_index` | int or null | the earliest position where the tokens differ; `null` when `diverges` is `false` |
| `identical_prefix_tokens` | list of str | tokens before that position (same in both runs; taken from the baseline's pieces). The whole output when there is no divergence |
| `identical_prefix_text` | str | the same, joined |
| `baseline_from_divergence`, `steered_from_divergence` | list of str | tokens from that position onward in each run (empty lists when there is no divergence) |
| `baseline_from_divergence_text`, `steered_from_divergence_text` | str | the same, joined |
| `tokens_changed` | int | positions that differ, counted only from `first_divergence_index`; `0` when there is no divergence |
| `tokens_compared_from_divergence` | int | positions from `first_divergence_index` to the end of the longer output; `0` when there is no divergence |
| `percent_changed_from_divergence` | float or null | `tokens_changed / tokens_compared_from_divergence`, as a percent with one decimal; `null` when there is no divergence |
| `percent_changed_raw` | float | `tokens_changed` over all generated positions (the longer output), as a percent with one decimal; `0.0` when there is no divergence |
| `percent_changed_raw_note` | str | fixed text saying the raw percent overstates the direct effect, because everything after the first divergence drifts. Show it next to the raw percent if that number is shown at all |

A character split across tokens at the divergence point can appear whole in one list and as `""` pieces in the other. That is expected.

```json
{
  "feature_id": 13,
  "mode": "amplify",
  "strength": 0.5,
  "seed": 42,
  "max_new_tokens": 60,
  "baseline_text": "The acrobat hung his head and walked toward the door, but had no idea where he was going.\n\n\"Well, it's all fine,\" he said.\n\n\"Let's go,\" ...",
  "steered_text": "The acrobat hung his head and walked up to the roof and closed the window, only to have it fall down on him. The roof was flipped down, ...",
  "diverges": true,
  "first_divergence_index": 4,
  "identical_prefix_text": " his head and walked",
  "baseline_from_divergence_text": " toward the door, but had no idea where he was going.\n\n\"Well, it's all fine,\" he said.\n\n\"Let's go,\" ...",
  "steered_from_divergence_text": " up to the roof and closed the window, only to have it fall down on him. The roof was flipped down, ...",
  "percent_changed_raw_note": "percent_changed_raw counts every differing token against all generated tokens. It overstates the direct effect of the intervention: after the first divergence the two runs continue from different text, so later tokens drift even where the intervention no longer acts. Use first_divergence_index and tokens_changed instead."
}
```

The call was `seed_matched_compare(rt, "The acrobat hung", 13, "amplify", 0.5, 21.4249, 60, 42)`. Left out of this example because they were not recorded in the run: `prompt_tokens`, `baseline_new_tokens`, `steered_new_tokens`, the two token lists (`identical_prefix_tokens`, `baseline_from_divergence`, `steered_from_divergence`), `tokens_changed`, `tokens_compared_from_divergence` and the two percent fields. A real response has all of them. The `percent_changed_raw_note` text is the fixed text returned by the code.

No divergence (for example amplify at strength 0): `diverges` is `false`, `first_divergence_index` is `null`, `identical_prefix_*` hold the whole output, the `*_from_divergence*` fields are empty, `tokens_changed` and `tokens_compared_from_divergence` are `0`, `percent_changed_from_divergence` is `null`, `percent_changed_raw` is `0.0`.

---

## X2. Strength sweep

`strength_sweep(rt, prompt, feature_id, max_activation, strengths=DEFAULT_SWEEP_STRENGTHS, max_new_tokens=60, seed=42)`

Amplifies one feature at several strengths with the same seed. The baseline is generated once.

The documented output has no activations yet. The function has an opt-in `with_activations` flag (off by default, used by `ml/scripts/phase4_demo.py`): it adds the six X1 activation fields to each result and `target_activation_baseline` and `target_activation_baseline_mean` at the top level. That is not part of the documented response.

| Input | Type | Notes |
| --- | --- | --- |
| `prompt`, `feature_id`, `max_activation`, `max_new_tokens`, `seed` | as in X1 | mode is always amplify |
| `strengths` | list of float | default `[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`; 1 to 12 values, each 0 to 10 in ML (the server keeps its own cap of 3); order is kept |

| Output field | Type | Notes |
| --- | --- | --- |
| `feature_id` | int | echoed |
| `mode` | str | always `"amplify"` |
| `seed`, `max_new_tokens` | int | echoed |
| `baseline_text` | str | no steering, prompt included |
| `baseline_degenerate` | bool | the rule below applied to the baseline |
| `baseline_reason` | str or null | |
| `results` | list | one entry per strength, in the order given |

Each `results` entry:

| Field | Type | Notes |
| --- | --- | --- |
| `strength` | float | |
| `text` | str | full text, prompt included |
| `differs_from_baseline` | bool | |
| `first_divergence_index` | int or null | as in X3; `null` when identical to the baseline |
| `degenerate` | bool | see the rule below |
| `reason` | str or null | why `degenerate` is true; `null` otherwise |
| `metrics` | object | the measured values (see below) |

`metrics`: `new_tokens` (int), `repeated_token_fraction` (float, 0 to 1), `repeated_trigram_fraction` (float, 0 to 1), `non_word_fraction` (float, 0 to 1), `replacement_chars` (int).

```json
{
  "feature_id": 13,
  "mode": "amplify",
  "seed": 42,
  "max_new_tokens": 60,
  "baseline_text": "The acrobat hung his head and walked toward the door, but had no idea where he was going.\n\n\"Well, it's all fine,\" he said.\n\n\"Let's go,\" ...",
  "results": [
    {
      "strength": 0.5,
      "text": "The acrobat hung his head and walked up to the roof and closed the window, only to have it fall down on him. The roof was flipped down, ...",
      "differs_from_baseline": true,
      "first_divergence_index": 4
    },
    {
      "strength": 1.0,
      "text": "The acrobat hung his head down and fell down the aisle. \"It's down, man,\" she said, \"but down, up!\"\n\nThe fall landed him in his arse-dow...",
      "differs_from_baseline": true,
      "degenerate": true,
      "reason": "repeating loop (measured values not recorded)"
    }
  ]
}
```

The call was `strength_sweep(rt, "The acrobat hung", 13, 21.4249, ...)`. Only two results are shown. Left out because they were not recorded in the run: `baseline_degenerate`, `baseline_reason`, `metrics`, and the other strengths of the sweep. The `reason` string is shortened here; a real one includes the measured percentages. A strength-0 result is expected to match the baseline, but none was recorded, so none is shown.

### Degeneration rule

`assess_degeneration(token_ids, text)` is applied to the newly generated tokens and text only, never the prompt. An output is `degenerate` when any of these holds. `reason` lists every rule that fired, with the measured value.

| Rule | Constant | Value |
| --- | --- | --- |
| Rules 2 and 3 are skipped for outputs shorter than this many tokens | `MIN_TOKENS_FOR_RULES` | 8 |
| 1. No new text was generated | | |
| 2a. Mostly repeated tokens: `1 - distinct tokens / total tokens` is at least | `REPEAT_UNIGRAM_FRACTION` | 0.6 |
| 2b. Repeating loop: the share of token trigrams that repeat an earlier trigram is at least (needs at least `MIN_TRIGRAMS_FOR_RULE` = 6 trigrams) | `REPEAT_TRIGRAM_FRACTION` | 0.3 |
| 3. Mostly non-words: the share of whitespace-separated chunks that are not words is at least (needs at least `MIN_CHUNKS_FOR_RULE` = 5 chunks) | `NON_WORD_FRACTION` | 0.5 |
| 4. U+FFFD: at least this many replacement characters, or more than this share of the characters | `REPLACEMENT_CHAR_MIN_COUNT` / `REPLACEMENT_CHAR_FRACTION` | 3 / 0.02 |

A chunk counts as a word when, after stripping surrounding punctuation, it is not empty, has no U+FFFD, contains a letter or digit, and at least `NON_WORD_LETTER_RATIO` (0.5) of its characters are letters, digits, apostrophes or hyphens. Letters of any script count, so the rule does not judge language. One U+FFFD at the very end of the text is ignored, because cutting generation at `max_new_tokens` can end inside a multi-byte character.

The limits are heuristics. They have been seen on only a few real outputs (see Observed behaviour below) and are not calibrated; `ml/scripts/phase4_demo.py` prints the measured values so they can be adjusted. A `degenerate` flag marks an output that is mostly repeats, noise or broken characters. It is not a quality score and says nothing about outputs that are not flagged. The rule catches repetition loops and non-word runs, not all low-quality text, and a result marked ok can still be poor.

### Observed behaviour (one feature, one seed)

One real run of a sweep: feature 13 (label "upside – upside down"), prompt `The acrobat hung`, `seed` 42, `max_new_tokens` 60, amplify. "Diverges at new token N" is `first_divergence_index`. Values that were not recorded in the run are marked as such.

| Strength | Compared with the baseline | Degenerate flag |
| --- | --- | --- |
| 0.25 | identical for 25 new tokens (positions 0 to 24), then drifts slightly | not recorded |
| 0.5 | diverges at new token 4 | not recorded |
| 0.75 | diverges at new token 4 | not recorded |
| 1.0 | diverges at new token 2 | flagged as a repeating loop |
| 1.5 | diverges at new token 0 | flagged as mostly repeated tokens |
| 2.0 | diverges at new token 0 | flagged as mostly repeated tokens |

Text from the run, shortened with "..." (the 0.5 and 1.0 texts are in the X2 example above):

- 0.25 continues like the baseline, then drifts: `" his head and walked toward the door, but had no idea where he was going.\n\n\"Well, it's all right, I'll go for a ride, I "...` The baseline at the same point reads `"Well, it's all fine," he said.`
- 1.5 and 2.0 (not separated in the notes) begin like `" down the down--down-down-down--..."`.

This is one feature, one prompt and one seed. It is not a general rule: other features, prompts and seeds can behave differently, and the strengths at which loops start are not fixed. The default sweep covers 0 to 2 and is dense below 1 because, for this feature, the changes and the earliest loops appeared in that range.

---

## X4. Prompt-only baseline

`prompt_only_baseline(rt, prompt, max_new_tokens=60, seed=42)`

What the model writes from the prompt by itself: no steering, same seed. Errors: `INVALID_INPUT` for a blank prompt, bad `max_new_tokens` or bad `seed`.

| Output field | Type | Notes |
| --- | --- | --- |
| `prompt` | str | echoed |
| `seed`, `max_new_tokens` | int | echoed |
| `text` | str | prompt plus the new text |
| `continuation` | str | the new text only |
| `new_tokens` | int | number of generated tokens |

With the same prompt, seed and `max_new_tokens`, `text` equals the `baseline_text` of X1, X2 and X3.

```json
{
  "prompt": "He turned the picture",
  "seed": 42,
  "max_new_tokens": 60,
  "text": "He turned the picture over to me and said, \"You are a good guy, sir.\"...",
  "continuation": " over to me and said, \"You are a good guy, sir.\"..."
}
```

Left out because it was not recorded in the run: `new_tokens`.

---

## W5. Multi-concept steering

`multi_steering_check(rt, prompt, steers, max_new_tokens=60, seed=42, with_single_controls=False)`

Steers 2 or 3 features together and compares the result with the baseline. The endpoint is not decided: Mirdul proposes it, Prabodh signs off. Nothing here was added to `contracts/` or `server/`.

**How the directions combine.** The directions are summed in the same forward hook. For each steer, `strength x max_activation x decoder direction` is added at every position except position 0, and the terms are added together. Amplify only; there is no ablate in a combination. The terms are summed in order of feature id, so the order of `steers` in the request does not change the result. With one non-zero strength (the others at 0) the hook output is identical to the single-feature hook, and the result should match X1 for that feature; this was tested bitwise on a toy model and on random tensors, not on the real model. The single-feature maths and X1 to X4 are unchanged.

**What the runtime must provide.** `rt.generate(prompt, max_new_tokens, seed, steering)` must accept `neuronlens.steering_checks.MultiSteeringSpec(steers=((feature_id, strength, max_activation), ...))` as `steering` (`neuronlens.runtime.Runtime` does), plus `rt.d_sae` and `rt.feature_activations(token_ids, feature_id)`, as for X1. `RealRuntime` and the mock need to forward or stand in for them.

| Input | Type | Notes |
| --- | --- | --- |
| `prompt` | str | not blank |
| `steers` | list of 2 or 3 objects | each `{"feature_id": int, "strength": float, "max_activation": float}`; the feature ids must be different |
| `steers[].feature_id` | int | 0 to `d_sae - 1` |
| `steers[].strength` | float | 0 to 10 in ML (the server keeps its own cap of 3); 0 adds nothing |
| `steers[].max_activation` | float | from the database; must be above 0 (a feature with 0 is dead) |
| `max_new_tokens` | int | 1 to 120, default 60 |
| `seed` | int | 0 to 2147483647, default 42; used for the baseline and the steered run |
| `with_single_controls` | bool | default `false`; ML-side option, see below |

| Output field | Type | Notes |
| --- | --- | --- |
| `prompt` | str | echoed |
| `seed`, `max_new_tokens` | int | echoed |
| `steers` | list | echoed, with `strength` and `max_activation` as floats, in the order given |
| `baseline_text` | str | no steering, prompt included |
| `steered_text` | str | all steers applied together, prompt included |
| `differs` | bool | `baseline_text != steered_text` |
| `first_divergence_index` | int or null | as in X3 (0-based, new tokens only); `null` when the texts are identical |
| `degenerate` | bool | the degeneration rule of X2 applied to the steered text |
| `reason` | str or null | why `degenerate` is true; `null` otherwise |
| `activations` | list | one entry per steered feature, in the order of `steers` |
| `activations[].feature_id` | int | |
| `activations[].target_activation_baseline` | float | that feature's maximum activation over the new tokens of the baseline text |
| `activations[].target_activation_steered` | float | the same for the steered text |
| `activations[].activation_verdict` | str | `"higher"`, `"lower"` or `"about the same"`, from `(steered max - baseline max) / max_activation` with the X1 threshold (`ACTIVATION_SAME_FRACTION`, 0.05, uncalibrated) |

The activations are measured as in X1: both texts are generated, then re-encoded through the model without any steering hook, and each feature is read over the new tokens (the prompt and position 0 are left out). When the two texts are identical the baseline numbers are reused. This costs one forward pass per feature per text.

**Comparing with the single-feature runs.** Either call X1 for each feature with the same `prompt`, `seed`, `max_new_tokens`, `strength` and `max_activation`, or pass `with_single_controls=true`. The option runs each feature alone at its own strength (one more generation per feature) and adds `single_feature_runs`, a list in the order of `steers`: `{feature_id, strength, steered_text, differs, first_divergence_index, degenerate, reason, target_activation_baseline, target_activation_steered, activation_verdict}`. It is not part of the default response.

### Errors

Same error type as the other functions (`SteeringCheckError`, with a `code`). Nothing is generated for a bad request.

| `code` | When |
| --- | --- |
| `INVALID_INPUT` | fewer than 2 or more than 3 steers, or `steers` not a list; an entry missing `feature_id`, `strength` or `max_activation`; duplicate feature ids; a strength that is not a finite number from 0 to 10; a `max_activation` that is negative, not finite or 0 (a dead feature); a blank prompt; a bad `seed` or `max_new_tokens` |
| `FEATURE_NOT_FOUND` | a `feature_id` that is negative or not below `rt.d_sae` |

### Example (all values are example values)

Every value below is an example value, not output of a real run. No real W5 run has been recorded yet; real values will replace these when they are available. The feature ids and the text are placeholders.

```json
{
  "prompt": "(example prompt)",
  "seed": 42,
  "max_new_tokens": 60,
  "steers": [
    {"feature_id": 101, "strength": 0.5, "max_activation": 20.0},
    {"feature_id": 202, "strength": 0.5, "max_activation": 15.0}
  ],
  "baseline_text": "(example baseline text)",
  "steered_text": "(example steered text)",
  "differs": true,
  "first_divergence_index": 4,
  "degenerate": false,
  "reason": null,
  "activations": [
    {"feature_id": 101, "target_activation_baseline": 0.0, "target_activation_steered": 9.5, "activation_verdict": "higher"},
    {"feature_id": 202, "target_activation_baseline": 2.0, "target_activation_steered": 2.1, "activation_verdict": "about the same"}
  ]
}
```

### Limits

- 2 to 3 features, amplify only. A feature cannot be ablated inside a combination.
- A combination can break the text at strengths where each feature alone is fine, so a combination should be set beside the single-feature runs. This has not been measured in a real run yet.
- The activation fields are measured on the finished text, re-encoded without steering. They say whether each feature appears in the output text, not whether the output is better, and the limits listed for X1 apply to each feature.
- The `degenerate` flag has the limits of the X2 rule: it catches repetition loops and non-word runs, not all low-quality text, and a result marked ok can still be poor.
- Cost: two generations (three or four with `with_single_controls`) plus two forward passes per feature. It has not been timed.
- The combined output is a test of what the features do together; it is not presented as an improvement.

## Timing

Two measurements, both on one GPU laptop:

- A script run of 8 generations of 60 new tokens (one baseline plus 7 strengths) took 15.1 seconds of wall time. That run included loading the model, so it overstated the cost of a call. It is not used for the estimates below.
- Through the running server in real mode with the model already loaded (warm): a sweep for feature 13, prompt `The acrobat hung`, strengths `[0, 0.5, 1]`, 60 new tokens, seed 42 (one baseline plus three steered generations, 4 generations in total) took 2.66 seconds of wall time for the whole HTTP request. That is about 0.65 seconds per generation.

The estimates below come from the warm-server measurement. They are estimates, not benchmarks:

| Call | Generations | Estimate |
| --- | --- | --- |
| one generation (X4) | 1 | roughly 0.65 seconds |
| X1 or X3 check | 2 | about 1.5 seconds |
| X2 sweep with 7 strengths | 8 (baseline plus 7) | about 5 seconds |

A call made right after the server starts, before the model has been used, will be slower because of model loading; that was not measured. The numbers will change on other hardware, and they were measured at 60 new tokens only. X1 also re-encodes the two texts (two short forward passes after the generations); that extra time was not measured separately.

The server should still use generous timeouts, and the UI should show a progress state for a sweep.

## Open items for the server and contract owners

- Forward `generate` and `d_sae` from `RealRuntime`, and provide a mock stand-in (see Common rules).
- Decide the response models and endpoint paths; the field names above are what the ML functions return.
- Read `max_activation` from the database and pass it in, as `/steer` does.
