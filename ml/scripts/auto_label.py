"""Auto-name SAE features with an LLM (C6). Nothing touches the DB until `apply`:

    cd ml
    # 1. see exactly what would be sent, no LLM call
    python scripts/auto_label.py propose --db ../data/neuronlens.db --dry-run
    # 2. one Gemini slot (one model, one key; --slot NAME, default the first in GEMINI_MODELS).
    #    Prints the labels and writes the JSON; the DB is not touched without --apply
    python scripts/auto_label.py propose --db ../data/neuronlens.db \
        --start 1800 --end 1900 --out ../data/auto_labels.json        # or --ids 1863,3446
    # 3. same command with --apply writes the labels into the DB, after copying it to
    #    neuronlens.db.bak3 (refuses if that file exists). Answered ids are not re-asked.
    #    A reviewed JSON can also be written with:
    python scripts/auto_label.py apply --db ../data/neuronlens.db --labels ../data/auto_labels.json
    # 4. the 50-feature spot-check sheet for Prabodh
    python scripts/auto_label.py sheet --db ../data/neuronlens.db --out ../data/spotcheck.csv

Rules: only live features with label_status 'unlabelled' and no label are ever proposed or
written, so a label a human set is never overwritten. The model may answer 'ambiguous'
instead of forcing a name. Auto labels are marked in `notes` with the model name. The --out
file holds only label and status (plus the model name once); each answer's `reason` goes to
a side file, <out>.reasons.json, which `apply` never reads. Re-running `propose` with the
same --out skips features already answered.
"""
import argparse
import csv
import json
import os
import queue
import random
import re
import shutil
import signal
import sqlite3
import sys
import threading
import time

# ---------- settings: edit these ----------

# Gemini. Leave empty to read GEMINI_API_KEY from the environment or the repo's .env
# (which git ignores). If you paste a key here, never commit this file.
GEMINI_API_KEY = ""

# One entry per model slot; workers rotate across slots like the old local-server list.
# name   : shown in logs so you can tell slots apart (defaults to "slot-N" if omitted).
# api_key: overrides GEMINI_API_KEY for that slot (useful for multiple project keys).
GEMINI_MODELS = []

# ---------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUTO_NOTE = "auto-labelled"
MAX_LABEL_LEN = 32
WINDOW = 12          # tokens shown on each side of the peak
MAX_ATTEMPTS = 3     # tries per feature across all servers
MAX_SERVER_FAILS = 5 # consecutive failures before a server is dropped

# Prabodh's label standard. Labels never contain quote marks, so neither do the examples.
SYSTEM = """You name features of a sparse autoencoder trained on GPT-2 small (layer 8).
For each feature you get its top text examples (the token where it fires is marked with [[ ]]),
and the tokens it makes more or less likely to come next.

Return JSON only, with these keys:
  "label": the key word, then a short context, joined by " – " (space, en dash, space).
           Not a sentence. 2 to 5 words, at most 32 characters, lowercase, no quote marks,
           no parentheses, no ending punctuation. The key word is the word or word piece the
           feature FIRES ON; the context says when or where (the phrase it sits in, the topic,
           or the word it predicts next). If it fires on a topic or a text format rather than
           one word, give just that topic in 1 to 3 words. Do not say what the model "knows".
  "status": "named" if the examples clearly agree, "ambiguous" if they are a mix or have
            unrelated uses, in which case give your best short guess.
  "reason": one sentence citing evidence from the examples (this is not shown to users).

Rules: be specific; "language", "text" and "words" alone are not labels.
- Key word: the exact token that is marked [[ ]], not a neighbour (in 3[[D]] the key is d).
  If it is a piece of a word, end it with a hyphen (hon-). If it is punctuation, a symbol or
  a space, write its name in words (open bracket, semicolon, comma, space), never the symbol.
  Small marked tokens (a space, a comma, a pronoun, a word piece) ARE the key word: never
  swap in a nearby bigger word. In "than[[ we]] assumed" the key is we, not than.
- Context must add something the key word does not already say. Never repeat the key word
  ("old – the word old") and never use filler like "word start", "word fragment" or
  "prefix" on its own. For a word piece, name the words it is part of (hon- – honing, honk).

Examples of good labels: alzheimer – disease, after – named after, in – before a year,
hon- – honing, honk, upside – upside down, la – anti-slavery, surnames – citations,
open bracket – function call, semicolon – between clauses, space – after a heading,
we – after than, food lists, legal contracts."""

# context words that say nothing on their own ("ul- – word fragment", "old – the word old")
FILLER = {"the", "a", "an", "of", "word", "words", "token", "tokens", "piece", "part",
          "fragment", "prefix", "suffix", "start", "end", "letter", "letters", "sequence"}

QUOTES = "\"'`‘’“”«»"
BRACKETS = "()[]{}"
# joining words a 32-char cut can leave dangling at the end
DANGLING = {"or", "and", "of", "the", "a", "an", "to", "in", "on", "for", "with", "by",
            "from", "as", "at", "nor", "but", "–"}
END_PUNCT = ".,;:!?"
DASH = " – "  # word – context
SPACED_DASH = re.compile(r"\s+(?:-+|–|—|:)\s+")
GENERIC = {"text", "language", "languages", "word", "words", "token", "tokens", "various",
           "misc", "miscellaneous", "general", "mixed", "topics", "things", "stuff"}

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "status": {"type": "string", "enum": ["named", "ambiguous"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "status", "reason"],
}

# --batch N: N features per request; each one still goes through clean() on its own
BATCH_RULES = """

You get several features in one message, each starting with "Feature <id>.". Label each
feature only from its own examples and tokens; never let one feature's examples influence
another's label. Return JSON only: {"features": [{"id": <id>, "label": ..., "status": ...,
"reason": ...}, ...]} with exactly one entry per feature, in the order given."""

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, **SCHEMA["properties"]},
                "required": ["id"] + SCHEMA["required"],
            },
        },
    },
    "required": ["features"],
}


# ---------- reading the DB ----------

def connect_ro(db_path):
    uri = "file:" + os.path.abspath(db_path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def parse_ids(text):
    """'1800-1900,3446' -> [1800, ..., 1900, 3446]."""
    ids = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad range {part!r}")
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(int(part))
    return ids


def candidate_ids(con, ids=None):
    """Live features that have never been labelled. Human labels are never candidates."""
    sql = ("SELECT feature_id FROM features WHERE is_dead = 0 "
           "AND label_status = 'unlabelled' AND label IS NULL")
    rows = [r["feature_id"] for r in con.execute(sql + " ORDER BY feature_id")]
    if ids is not None:
        wanted = ids if isinstance(ids, range) else set(ids)
        rows = [f for f in rows if f in wanted]
    return rows


def snippet(tokens, peak):
    lo, hi = max(0, peak - WINDOW), min(len(tokens), peak + WINDOW + 1)
    text = "".join(tokens[lo:peak]) + "[[" + tokens[peak] + "]]" + "".join(tokens[peak + 1:hi])
    return ("..." if lo > 0 else "") + text.replace("\n", " ") + ("..." if hi < len(tokens) else "")


def feature_evidence(con, fid, n_examples=10, n_logits=10):
    examples = [snippet(json.loads(r["tokens_json"]), r["peak_index"]) for r in con.execute(
        "SELECT tokens_json, peak_index FROM feature_examples WHERE feature_id = ? "
        "ORDER BY rank LIMIT ?", (fid, n_examples))]
    logits = {"promoted": [], "suppressed": []}
    for r in con.execute("SELECT direction, token FROM feature_logits WHERE feature_id = ? "
                         "AND rank <= ? ORDER BY direction, rank", (fid, n_logits)):
        logits[r["direction"]].append(r["token"])
    return {"feature_id": fid, "examples": examples, **logits}


def build_prompt(ev):
    lines = [f"Feature {ev['feature_id']}.", "", "Top snippets:"]
    lines += [f"{i}. {s}" for i, s in enumerate(ev["examples"], 1)]
    lines += ["", f"Promoted next tokens: {ev['promoted']!r}",
              f"Suppressed next tokens: {ev['suppressed']!r}"]
    return "\n".join(lines)


# ---------- the LLMs ----------

def read_key(name, hardcoded):
    """Hardcoded value first, then the environment, then the repo's .env file."""
    if hardcoded:
        return hardcoded
    if os.environ.get(name):
        return os.environ[name]
    env = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env):
        with open(env, encoding="utf-8") as f:
            for line in f:
                k, sep, v = line.partition("=")
                if sep and k.strip() == name:
                    return v.split("#", 1)[0].strip().strip("\"'")
    return ""


def parse_answer(text):
    """JSON from a model reply, tolerating ```json fences and <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON in reply: {text[:120]!r}")
        return json.loads(text[start:end + 1])


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = 0

    def add(self, inp, out):
        with self.lock:
            self.calls += 1
            self.input_tokens += inp or 0
            self.output_tokens += out or 0


class RateLimiter:
    """Spaces calls evenly so all workers together stay under `rpm` requests a minute."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        time.sleep(max(0.0, at - now))


class GeminiLabeller:
    """Gemini through the google-genai SDK; JSON output enforced by response_json_schema."""

    def __init__(self, model, rpm, api_key=None, slot=0, name=None):
        from google import genai  # imported here so tests need no SDK installed
        from google.genai import types
        key = api_key or read_key("GEMINI_API_KEY", GEMINI_API_KEY)
        if not key:
            raise SystemExit("no Gemini key: set GEMINI_API_KEY in this script, the "
                             "environment, or the repo's .env")
        self.types = types
        self.client = genai.Client(api_key=key)
        self.model = model
        self.name = f"gemini:{name or f'slot-{slot}'}({model})"
        self.usage = Usage()
        self.limiter = RateLimiter(rpm)

    def __call__(self, system, prompt, schema=SCHEMA):
        self.limiter.wait()
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system, temperature=0,
                    response_mime_type="application/json", response_json_schema=schema))
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from e
        u = response.usage_metadata
        tokens = (0, 0)
        if u is not None:
            tokens = (u.prompt_token_count or 0,
                      (u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
            self.usage.add(*tokens)
        # answer None: the worker parses raw, after counting the tokens this call spent
        return None, response.text, tokens



def clean_label(text):
    """Enforce the label standard: lowercase `word – context`, no quotes/backticks/brackets,
    single spaces, any spaced dash or colon made one en dash, at most MAX_LABEL_LEN chars
    (cut at a word boundary), no ending punctuation."""
    text = str(text or "").lower()
    text = text.translate({ord(c): None for c in QUOTES})
    text = text.translate({ord(c): " " for c in BRACKETS})
    text = SPACED_DASH.sub(DASH, " ".join(text.split()))
    if len(text) > MAX_LABEL_LEN:
        cut = text[:MAX_LABEL_LEN + 1].rfind(" ")
        text = text[:cut] if cut > 0 else text[:MAX_LABEL_LEN]
        words = text.split(" ")  # a cut must not leave "... life or"
        while len(words) > 1 and words[-1] in DANGLING:
            words.pop()
        text = " ".join(words)
    return text.rstrip(END_PUNCT + " –—")


def clean(result):
    label = clean_label(result.get("label"))
    status = result.get("status")
    if status not in ("named", "ambiguous"):
        raise ValueError(f"bad status: {result!r}")
    if not label or all(w in GENERIC for w in label.replace("–", " ").split()):
        raise ValueError(f"empty or generic label: {result.get('label')!r}")
    key, sep, context = label.partition(DASH)
    if label.startswith("–") or (sep and not key.strip()):
        raise ValueError(f"no key word: {result.get('label')!r}")
    if sep and set(context.split()) - FILLER <= set(key.split()) | {k.rstrip("-") for k in key.split()}:
        raise ValueError(f"context adds nothing: {result.get('label')!r}")
    return {"label": label, "status": status,
            "reason": " ".join(str(result.get("reason", "")).split())[:300]}


# ---------- propose ----------

def load_json(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def by_id(d):
    return dict(sorted(d.items(), key=lambda kv: int(kv[0])))


def reasons_path(path):
    return os.path.splitext(path)[0] + ".reasons.json"


def load_run(path):
    """(model, labels, reasons) from a propose output; empty if the file is missing."""
    data, reasons = load_json(path), load_json(reasons_path(path))
    if data and "labels" not in data:  # older files: {fid: {...}} with no run header
        return None, data, reasons
    return data.get("model"), data.get("labels", {}), reasons


def usage_path(path):
    return os.path.splitext(path)[0] + ".usage.json"


def save_run(path, model, labels, reasons, usage=None):
    save_json(path, {"model": model, "labels": by_id(labels)})
    save_json(reasons_path(path), by_id(reasons))
    if usage is not None:
        save_json(usage_path(path), by_id(usage))


def usage_summary(usage, done_ids, calls=None, failed=None):
    """Totals and per-labelled-feature averages. `calls`/`failed` are the real request
    counts (a batched request appears in every one of its features' usage)."""
    if calls is None:
        calls = sum(u["calls"] for u in usage.values())
    if failed is None:
        failed = sum(u["failed_calls"] for u in usage.values())
    inp = sum(u["input_tokens"] for u in usage.values())
    out = sum(u["output_tokens"] for u in usage.values())
    n = max(1, len(done_ids))
    return (f"run total: {calls} calls ({failed} failed), {inp:.0f} input tokens, {out:.0f} "
            f"output tokens; per labelled feature: {calls / n:.2f} calls, {inp / n:.0f} in, "
            f"{out / n:.0f} out")


def usage_lines(labellers):
    seen, lines = set(), []
    for lab in labellers:
        if id(lab) in seen:
            continue
        seen.add(id(lab))
        u = lab.usage
        lines.append(f"usage {lab.name}: {u.calls} calls, {u.input_tokens} input tokens, "
                     f"{u.output_tokens} output tokens")
    return lines


def batch_prompt(prompts, fids):
    return (f"Label these {len(fids)} features.\n\n"
            + "\n\n".join(prompts[f] for f in fids))


def split_batch(answer, fids):
    """{fid: answer dict} from a batched reply. Ids the model left out are simply missing."""
    items = answer.get("features") if isinstance(answer, dict) else answer
    if not isinstance(items, list):
        raise ValueError(f"batched reply has no features list: {str(answer)[:120]!r}")
    wanted, out = set(fids), {}
    for item in items:
        try:
            fid = int(item.get("id"))
        except (AttributeError, TypeError, ValueError):
            continue
        if fid in wanted and fid not in out:
            out[fid] = item
    return out


def propose(db_path, out_path, labellers, limit=None, ids=None, log=print, verbose=False,
            batch=1, progress=None):
    """Label every candidate not already in out_path. One worker thread per entry in
    `labellers` (pass the same object twice for two workers on one key). Workers pull
    requests from a shared queue; each request carries `batch` features. Features whose
    answer fails (bad call, bad JSON, missing from a batch, or rejected by `clean`) go back
    on the queue on their own, up to MAX_ATTEMPTS tries; a key that keeps failing is
    dropped. Every answer is saved as it arrives. All workers must run the same model.
    `progress(state)`, if given, is called after every request and replaces the per-feature
    lines. Returns (done, failed)."""
    models = {getattr(lab, "model", lab.name) for lab in labellers}
    if len(models) != 1:
        raise SystemExit(f"one model per run, got {sorted(models)}")
    model = models.pop()
    batch = max(1, int(batch))
    prev_model, results, reasons = load_run(out_path)
    usage = load_json(usage_path(out_path))  # fid -> calls and tokens spent on it
    if results and prev_model not in (None, model):
        raise SystemExit(f"{out_path} was made with {prev_model}, not {model}; use another --out")
    con = connect_ro(db_path)
    system = SYSTEM if batch == 1 else SYSTEM + BATCH_RULES
    todo = [f for f in candidate_ids(con, ids) if str(f) not in results]
    if limit is not None:
        todo = todo[:limit]
    prompts = {f: build_prompt(feature_evidence(con, f)) for f in todo}
    con.close()
    log(f"{len(todo)} features to label with {len(labellers)} worker(s), "
        f"{batch} per call ({len(results)} already in {out_path})")
    if verbose:
        log(f"\n{'='*60}\nSYSTEM PROMPT\n{'='*60}\n{system}\n{'='*60}\n")

    work = queue.Queue()
    for i in range(0, len(todo), batch):
        work.put((tuple(todo[i:i + batch]), 1, frozenset()))  # (features, attempt, failed keys)
    lock = threading.Lock()
    state = {"done": 0, "failed": 0, "in_flight": 0, "calls": 0, "failed_calls": 0,
             "total": len(todo), "input_tokens": 0, "output_tokens": 0}
    alive = {}  # key name -> live workers on it
    for lab in labellers:
        alive[lab.name] = alive.get(lab.name, 0) + 1

    stop = threading.Event()

    def _sigint(sig, frame):
        if not stop.is_set():
            log("  Ctrl+C: finishing in-flight requests then stopping (press again to force quit)")
            stop.set()

    old_sigint = signal.signal(signal.SIGINT, _sigint)

    def names(fids):
        return str(fids[0]) if len(fids) == 1 else f"{len(fids)} features ({fids[0]}..{fids[-1]})"

    def retry_or_give_up(fids, attempt, tried, lab, err):  # call with lock held
        for f in fids:
            usage[str(f)]["failed_calls"] += 1
        if attempt < MAX_ATTEMPTS:
            work.put((fids, attempt + 1, tried | {lab.name}))
            log(f"  {lab.name} failed on {names(fids)} (try {attempt}): {err}")
        else:
            state["failed"] += len(fids)
            log(f"  gave up on {names(fids)}: {err}")

    def worker(lab):
        fails = 0
        while fails < MAX_SERVER_FAILS:
            if stop.is_set():
                return
            try:
                fids, attempt, tried = work.get(timeout=0.05)
            except queue.Empty:
                with lock:  # finished only when nothing is queued or being worked on
                    if state["in_flight"] == 0 and work.empty():
                        return
                continue
            with lock:
                # work this key already failed goes to another live key first
                retry_elsewhere = lab.name in tried and any(
                    n not in tried for n, c in alive.items() if c > 0)
                if not retry_elsewhere:
                    state["in_flight"] += 1
                    state["calls"] += 1
                    for f in fids:
                        usage.setdefault(str(f), {"calls": 0, "failed_calls": 0,
                                                  "input_tokens": 0, "output_tokens": 0})
                        usage[str(f)]["calls"] += 1
            if retry_elsewhere:
                work.put((fids, attempt, tried))
                time.sleep(0.01)
                continue
            prompt = prompts[fids[0]] if batch == 1 else batch_prompt(prompts, fids)
            try:
                if verbose:
                    log(f"\n{'='*60}\nREQUEST {names(fids)}  ({lab.name})\n{'='*60}\n"
                        f"{prompt}\n{'-'*60}")
                if batch == 1:
                    parsed, raw_text, (t_in, t_out) = lab(system, prompt)
                else:
                    parsed, raw_text, (t_in, t_out) = lab(system, prompt, BATCH_SCHEMA)
                with lock:  # a batch's tokens are shared evenly by its features
                    state["input_tokens"] += t_in
                    state["output_tokens"] += t_out
                    for f in fids:
                        usage[str(f)]["input_tokens"] += t_in / len(fids)
                        usage[str(f)]["output_tokens"] += t_out / len(fids)
                if verbose:
                    log(f"GEMINI RESPONSE:\n{raw_text}\n{'='*60}\n")
                answer = parsed if parsed is not None else parse_answer(raw_text)
                answers = {fids[0]: answer} if batch == 1 else split_batch(answer, fids)
            except Exception as e:  # one bad reply never stops the run
                fails += 1
                with lock:
                    state["in_flight"] -= 1
                    state["failed_calls"] += 1
                    retry_or_give_up(fids, attempt, tried, lab, e)
                    save_json(usage_path(out_path), by_id(usage))
                    if progress:
                        progress(state)
                continue
            good, bad = {}, {}
            for f in fids:
                try:
                    if f not in answers:
                        raise ValueError("missing from the reply")
                    good[f] = clean(answers[f])
                except Exception as e:
                    bad[f] = e
            if good:
                fails = 0
            else:
                fails += 1
            with lock:
                state["in_flight"] -= 1
                if not good:
                    state["failed_calls"] += 1
                for f, res in good.items():
                    results[str(f)] = {"label": res["label"], "status": res["status"]}
                    reasons[str(f)] = res["reason"]
                    state["done"] += 1
                    u = usage[str(f)]
                    if not progress:
                        log(f"  [{state['done']}/{len(todo)}] {f} ({lab.name}): "
                            f"{res['status']} - {res['label']}  "
                            f"[{u['calls']} call{'s' if u['calls'] != 1 else ''}"
                            f"{f' of {len(fids)}' if batch > 1 else ''}, "
                            f"{u['input_tokens']:.0f} in / {u['output_tokens']:.0f} out]")
                if bad:
                    err = "; ".join(f"{f}: {e}" for f, e in bad.items())
                    retry_or_give_up(tuple(bad), attempt, tried, lab, err)
                save_run(out_path, model, results, reasons, usage)
                if progress:
                    progress(state)
        with lock:
            alive[lab.name] -= 1
            log(f"  dropped a {lab.name} worker after {MAX_SERVER_FAILS} failures in a row")

    threads = [threading.Thread(target=worker, args=(lab,), daemon=True) for lab in labellers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    signal.signal(signal.SIGINT, old_sigint)
    left = 0  # features still queued: only if every key was dropped or the run was stopped
    while not work.empty():
        left += len(work.get()[0])
    if stop.is_set():
        log(f"  stopped early: {state['done']} labelled so far, {left} features not done")
    for line in usage_lines(labellers):
        log(line)
    run_usage = {f: usage[str(f)] for f in todo if str(f) in usage}
    log(usage_summary(run_usage, [f for f in todo if str(f) in results],
                      state["calls"], state["failed_calls"]))
    log(f"per-feature calls and tokens -> {usage_path(out_path)}")
    return state["done"], state["failed"] + left


# ---------- apply ----------

BACKUP_SUFFIX = ".bak3"  # neuronlens.db -> neuronlens.db.bak3


def check_backup_free(db_path):
    bak = db_path + BACKUP_SUFFIX
    if os.path.exists(bak):
        raise SystemExit(f"{bak} already exists; move it away first")


def apply_labels(db_path, labels_path, backup=True):
    """Write proposed labels into the DB in one transaction. Only rows that are still
    unlabelled with no label are touched. Returns (written, skipped)."""
    model, labels, _ = load_run(labels_path)  # reasons stay out of the DB
    if backup:
        check_backup_free(db_path)
        shutil.copyfile(db_path, db_path + BACKUP_SUFFIX)
    con = sqlite3.connect(db_path)
    written = skipped = 0
    try:
        with con:
            for fid, v in labels.items():
                note = f"{AUTO_NOTE} ({model or v.get('model', 'unknown')})"
                cur = con.execute(
                    "UPDATE features SET label = ?, label_status = ?, "
                    "notes = CASE WHEN notes IS NULL OR notes = '' THEN ? "
                    "ELSE notes || ' | ' || ? END "
                    "WHERE feature_id = ? AND is_dead = 0 AND label_status = 'unlabelled' "
                    "AND label IS NULL",
                    (v["label"], v["status"], note, note, int(fid)))
                if cur.rowcount:
                    written += 1
                else:
                    skipped += 1
    finally:
        con.close()
    return written, skipped


# ---------- spot-check sheet ----------

def write_sheet(db_path, out_path, n=50, seed=0):
    """CSV for the 50-feature spot-check (Handbook labelling sheet columns)."""
    con = connect_ro(db_path)
    ids = [r["feature_id"] for r in con.execute(
        "SELECT feature_id FROM features WHERE notes LIKE ? ORDER BY feature_id",
        (f"%{AUTO_NOTE}%",))]
    picked = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_id", "auto_label", "auto_status", "top_examples",
                    "promoted", "top examples make sense?", "my_label",
                    "status (named / ambiguous)", "verdict"])
        for fid in picked:
            row = con.execute("SELECT label, label_status FROM features WHERE feature_id = ?",
                              (fid,)).fetchone()
            ev = feature_evidence(con, fid, n_examples=3, n_logits=5)
            w.writerow([fid, row["label"], row["label_status"], " || ".join(ev["examples"]),
                        " ".join(ev["promoted"]), "", "", "", ""])
    con.close()
    return len(picked)


# ---------- CLI ----------

def _short(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else f"{n:.0f}"


def _dur(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


class ProgressLine:
    """One status line rewritten in place; other messages print above it."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.start = time.monotonic()
        self.last = ""

    def _write(self, text):
        pad = max(0, len(self.last) - len(text))
        self.stream.write("\r" + text + " " * pad)
        self.stream.flush()

    def update(self, st):
        done, total = st["done"], max(1, st["total"])
        took = time.monotonic() - self.start
        left = st["total"] - done - st["failed"]
        eta = _dur(took / done * left) if done else "?"
        line = (f"labelling {done:,}/{st['total']:,} ({100 * done / total:.0f}%) | "
                f"{st['calls']:,} calls, {st['failed']:,} failed | "
                f"{_short(st['input_tokens'])} in / {_short(st['output_tokens'])} out | "
                f"{_dur(took)} elapsed, ETA {eta}")
        self._write(line)
        self.last = line

    def log(self, msg):
        if self.last:
            self._write("")
            self.stream.write("\r")
        print(msg, file=self.stream)
        if self.last:
            self._write(self.last)

    def close(self):
        if self.last:
            self.stream.write("\n")
            self.last = ""


# ---------- CLI (continued) ----------

def make_labellers(workers, slot=None, all_slots=False):
    """By default one slot (one model, one key) for the whole run: the named slot, else the
    first. With all_slots, every slot in GEMINI_MODELS gets `workers` workers with its own
    rate limit; propose() still refuses slots that run different models."""
    slots = [s for s in GEMINI_MODELS if s.get("model")]
    if not slots:
        raise SystemExit("GEMINI_MODELS is empty")
    if all_slots:
        if slot is not None:
            raise SystemExit("use --slot or --all-slots, not both")
        labellers = []
        for i, s in enumerate(slots):
            gem = GeminiLabeller(s["model"], s.get("rpm", 10), s.get("api_key") or None, i,
                                 s.get("name"))
            labellers.extend([gem] * max(1, workers))
        return labellers
    if slot is not None:
        slots = [s for s in slots if s.get("name") == slot]
        if not slots:
            names = ", ".join(str(s.get("name")) for s in GEMINI_MODELS)
            raise SystemExit(f"no slot named {slot!r}; slots: {names}")
    s = slots[0]
    # one instance so all workers share its RateLimiter
    gem = GeminiLabeller(s["model"], s.get("rpm", 10), s.get("api_key") or None, 0,
                         s.get("name"))
    return [gem] * max(1, workers)


def select_ids(a):
    """--ids list, or the inclusive --start/--end range, or None for every candidate."""
    if a.ids and (a.start is not None or a.end is not None):
        raise SystemExit("use --ids or --start/--end, not both")
    if a.ids:
        return parse_ids(a.ids)
    if a.start is None and a.end is None:
        return None
    lo, hi = a.start or 0, a.end if a.end is not None else 10 ** 9
    if hi < lo:
        raise SystemExit("--end is before --start")
    return range(lo, hi + 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("propose", help="ask the LLM(s), write labels to a JSON file")
    pr.add_argument("--db", required=True)
    pr.add_argument("--out", default="auto_labels.json")
    pr.add_argument("--limit", type=int)
    pr.add_argument("--ids", help="feature ids and ranges, e.g. 1800-1900,3446,3587")
    pr.add_argument("--start", type=int, help="first feature id (inclusive)")
    pr.add_argument("--end", type=int, help="last feature id (inclusive)")
    pr.add_argument("--workers", type=int, default=1, help="workers on the one key")
    pr.add_argument("--slot", help="GEMINI_MODELS entry to use (default: the first)")
    pr.add_argument("--all-slots", action="store_true",
                    help="use every GEMINI_MODELS entry at once (same model, separate keys); "
                         "a key that keeps failing is dropped and its work moves to the others")
    pr.add_argument("--progress", action="store_true",
                    help="one status line updated in place instead of a line per feature")
    pr.add_argument("--batch", type=int, default=1,
                    help="features per request (default 1); 10 cuts requests about 10x")
    pr.add_argument("--apply", action="store_true",
                    help=f"also write the labels into the DB (backs it up to *{BACKUP_SUFFIX} "
                         "first). Without it the DB is never touched")
    pr.add_argument("--dry-run", action="store_true",
                    help="print the prompt for the first candidate and stop; no LLM call")
    pr.add_argument("--verbose", action="store_true",
                    help="print the full prompt and raw Gemini response for every feature")
    ap = sub.add_parser("apply", help="write a labels JSON into the DB")
    ap.add_argument("--db", required=True)
    ap.add_argument("--labels", default="auto_labels.json")
    ap.add_argument("--no-backup", action="store_true")
    sh = sub.add_parser("sheet", help="write the spot-check CSV")
    sh.add_argument("--db", required=True)
    sh.add_argument("--out", default="spotcheck.csv")
    sh.add_argument("--n", type=int, default=50)
    sh.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.cmd == "propose":
        ids = select_ids(a)
        if a.dry_run:
            con = connect_ro(a.db)
            cands = candidate_ids(con, ids)
            print(f"{len(cands)} candidate features")
            if cands:
                first = cands[:max(1, a.batch)]
                prompts = {f: build_prompt(feature_evidence(con, f)) for f in first}
                if a.batch > 1:
                    print(SYSTEM + BATCH_RULES)
                    print("\n---\n" + batch_prompt(prompts, first))
                else:
                    print(SYSTEM)
                    print("\n---\n" + prompts[first[0]])
            con.close()
            return 0
        if a.apply:
            check_backup_free(a.db)  # fail before spending any tokens
        bar = ProgressLine() if a.progress and not a.verbose else None
        done, failed = propose(a.db, a.out, make_labellers(a.workers, a.slot, a.all_slots), a.limit,
                               ids,
                               verbose=a.verbose, batch=a.batch,
                               **({"log": bar.log, "progress": bar.update} if bar else {}))
        if bar:
            bar.close()
        print(f"done: {done} labelled, {failed} failed -> {a.out}")
        if a.apply:
            written, skipped = apply_labels(a.db, a.out)
            print(f"applied to {a.db} (backup {a.db + BACKUP_SUFFIX}): "
                  f"written {written}, skipped (already labelled or dead) {skipped}")
        else:
            print("DB not touched (add --apply to write these labels)")
        return 1 if failed and not done else 0
    if a.cmd == "apply":
        written, skipped = apply_labels(a.db, a.labels, backup=not a.no_backup)
        print(f"written: {written}, skipped (already labelled or dead): {skipped}")
        return 0
    n = write_sheet(a.db, a.out, a.n, a.seed)
    print(f"wrote {n} rows to {a.out}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
