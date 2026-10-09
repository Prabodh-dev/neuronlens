"""X6 sentence report and X7 text compare. Both aggregate Runtime.inspect; no new ML.

inspect is called with the SAE's k (every active feature on every token), then ranked and
trimmed to the requested top_n, so tokens_active and the ranking are not undercounted.
Token 0 has no features (an outlier position, as in /inspect). Optional max_density hides
features that fire on more than that share of tokens (features.density) before ranking, so
very common features do not crowd out specific ones; it is off by default. sort="relative"
ranks by relative_to_max (activation / the feature's dictionary max) instead of raw
activation; for compare, by |relative_to_max in b - in a|. Features with no dictionary max
(relative_to_max null) go last.

Sentence split (X6): a sentence ends after a token ending in . ! or ? when the next token
starts with whitespace or the text ends, and after any token containing a newline. Known
wrong splits: abbreviations followed by a space ("Mr. Smith", "U.S. troops", "e.g. this")
split early, and sentences with no space after the mark ("end.Next") are not split. Mock
mode tokens are a word split, so boundaries can differ slightly from real mode.
"""
from fastapi import APIRouter, Depends

from ..db import Database
from ..deps import get_db, get_meta, get_runtime
from ..errors import ApiError
from ..models import (ReportFeature, ReportRequest, ReportResponse, ReportSentence,
                      SearchFeature, SearchRequest, SearchResponse,
                      TextCompareFeature, TextCompareRequest, TextCompareResponse,
                      error_responses)
from .model import check_length

router = APIRouter(tags=["inspect analysis"], responses=error_responses(413, 422, 500, 503))

ENDINGS = (".", "!", "?")


def all_active(runtime, meta: dict, text: str) -> list[dict]:
    return runtime.inspect(text, int(meta["k"]))


def split_sentences(tokens: list[dict]) -> list[tuple[int, int]]:
    """[(start, end)] token ranges, end exclusive. Whitespace-only ranges are dropped."""
    spans, start = [], 0
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1]["text"] if i + 1 < len(tokens) else None
        ends = t["text"].rstrip().endswith(ENDINGS) and (nxt is None or nxt[:1].isspace())
        if ends or "\n" in t["text"] or nxt is None:
            spans.append((start, i + 1))
            start = i + 1
    return [(a, b) for a, b in spans if "".join(t["text"] for t in tokens[a:b]).strip()]


def aggregate(tokens: list[dict]) -> dict[int, dict]:
    """feature_id -> {max, count, peak} over the given tokens."""
    out: dict[int, dict] = {}
    for t in tokens:
        for f in t["features"]:
            if f["activation"] <= 0:
                continue
            agg = out.setdefault(f["feature_id"], {"max": 0.0, "count": 0, "peak": t["index"]})
            agg["count"] += 1
            if f["activation"] > agg["max"]:
                agg["max"], agg["peak"] = f["activation"], t["index"]
    return out


def dictionary(db: Database, ids) -> dict[int, tuple]:
    """feature_id -> (label, max_activation) from the DB."""
    ids = sorted(set(ids))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["feature_id"]: (r["label"], r["max_activation"]) for r in db.query(
        f"SELECT feature_id, label, max_activation FROM features WHERE feature_id IN ({marks})",
        tuple(ids))}


def drop_dense(db: Database, agg: dict, max_density: float | None) -> dict:
    """agg without features whose density is above max_density (None: unchanged)."""
    if max_density is None or not agg:
        return agg
    ids = sorted(agg)
    marks = ",".join("?" * len(ids))
    dense = {r["feature_id"] for r in db.query(
        f"SELECT feature_id FROM features WHERE feature_id IN ({marks}) AND density > ?",
        (*ids, max_density))}
    return {fid: v for fid, v in agg.items() if fid not in dense}


def relative(value: float, fid: int, info: dict) -> float | None:
    feature_max = info.get(fid, (None, None))[1]
    return round(value / feature_max, 4) if feature_max else None


NO_RELATIVE = float("-inf")  # sorts features without relative_to_max last


def report_key(sort: str, info: dict):
    if sort == "relative":
        def key(kv):
            rel = relative(kv[1]["max"], kv[0], info)
            return (-(rel if rel is not None else NO_RELATIVE), -kv[1]["max"], kv[0])
        return key
    return lambda kv: (-kv[1]["max"], kv[0])


def compare_key(sort: str, info: dict):
    if sort == "relative":
        def key(row):
            fid, a, b, _ = row
            ra, rb = relative(a["max"], fid, info), relative(b["max"], fid, info)
            change = abs(rb - ra) if ra is not None and rb is not None else NO_RELATIVE
            return (-change, -abs(row[3]), fid)
        return key
    return lambda row: (-abs(row[3]), row[0])


@router.post("/inspect/report", response_model=ReportResponse)
def report(req: ReportRequest, db: Database = Depends(get_db), meta: dict = Depends(get_meta),
           runtime=Depends(get_runtime)):
    """X6. For each sentence, the features with the highest activation on its tokens."""
    check_length(runtime, req.text, "text")
    tokens = all_active(runtime, meta, req.text)
    spans = [(a, b, drop_dense(db, aggregate(tokens[a:b]), req.max_density))
             for a, b in split_sentences(tokens)]
    ranking_info = (dictionary(db, (fid for *_, agg in spans for fid in agg))
                    if req.sort == "relative" else {})
    key = report_key(req.sort, ranking_info)
    per_sentence = [(a, b, sorted(agg.items(), key=key)[:req.top_n]) for a, b, agg in spans]
    info = dictionary(db, (fid for _, _, top in per_sentence for fid, _ in top))
    sentences = [
        ReportSentence(index=n, text="".join(t["text"] for t in tokens[a:b]), start_token=a,
                       end_token=b, features=[
                           ReportFeature(feature_id=fid, label=info.get(fid, (None,))[0],
                                         max_activation=v["max"],
                                         relative_to_max=relative(v["max"], fid, info),
                                         tokens_active=v["count"], peak_token=v["peak"])
                           for fid, v in top])
        for n, (a, b, top) in enumerate(per_sentence)]
    return ReportResponse(model_name=meta["model_name"],
                          simulated=bool(getattr(runtime, "is_simulated", False)),
                          sentences=sentences)


@router.post("/inspect/compare", response_model=TextCompareResponse)
def compare_texts(req: TextCompareRequest, db: Database = Depends(get_db),
                  meta: dict = Depends(get_meta), runtime=Depends(get_runtime)):
    """X7. Features whose max activation differs most between two texts."""
    check_length(runtime, req.text_a, "text_a")
    check_length(runtime, req.text_b, "text_b")
    tok_a = all_active(runtime, meta, req.text_a)
    tok_b = all_active(runtime, meta, req.text_b)
    agg_a = drop_dense(db, aggregate(tok_a), req.max_density)
    agg_b = drop_dense(db, aggregate(tok_b), req.max_density)
    empty = {"max": 0.0, "count": 0}
    rows = []
    for fid in set(agg_a) | set(agg_b):
        a, b = agg_a.get(fid, empty), agg_b.get(fid, empty)
        rows.append((fid, a, b, b["max"] - a["max"]))
    ranking_info = dictionary(db, (r[0] for r in rows)) if req.sort == "relative" else {}
    rows.sort(key=compare_key(req.sort, ranking_info))
    rows = rows[:req.top_n]
    info = dictionary(db, (r[0] for r in rows))
    features = [TextCompareFeature(
        feature_id=fid, label=info.get(fid, (None,))[0], max_activation_a=a["max"],
        max_activation_b=b["max"], difference=round(diff, 6), tokens_active_a=a["count"],
        tokens_active_b=b["count"], relative_to_max_a=relative(a["max"], fid, info),
        relative_to_max_b=relative(b["max"], fid, info)) for fid, a, b, diff in rows]
    return TextCompareResponse(model_name=meta["model_name"],
                               simulated=bool(getattr(runtime, "is_simulated", False)),
                               tokens_a=len(tok_a), tokens_b=len(tok_b), features=features)


# ---- W8: features that fired most on the query ----

SEARCH_PREFIX = "\n"     # scored as token 0 (the outlier position), then hidden
SEARCH_MAX_TOKENS = 32   # counts the query's own tokens only


def strip_prefix(tokens: list[dict], prefix: str) -> list[dict]:
    """The inspect tokens of prefix + query, without the prefix's characters. A token that
    straddles the end of the prefix keeps its query part. Indices restart at 0."""
    out, pos = [], 0
    for t in tokens:
        start, pos = pos, pos + len(t["text"])
        if pos <= len(prefix):
            continue
        text = t["text"][max(0, len(prefix) - start):]
        out.append({**t, "index": len(out), "text": text})
    return out


def densities(db: Database, ids) -> dict[int, float]:
    ids = sorted(set(ids))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["feature_id"]: r["density"] for r in db.query(
        f"SELECT feature_id, density FROM features WHERE feature_id IN ({marks})", tuple(ids))}


@router.post("/features/search", response_model=SearchResponse)
def search(req: SearchRequest, db: Database = Depends(get_db), meta: dict = Depends(get_meta),
           runtime=Depends(get_runtime)):
    """W8. The features that fired most on the words of the query. A fixed prefix is scored
    as token 0 (an outlier position with no features) so every query word is scored; it is
    never returned. Shows which features fired on the query, not what the query means."""
    n = runtime.count_tokens(req.query)
    if n > SEARCH_MAX_TOKENS:
        raise ApiError("TEXT_TOO_LONG",
                       f"query is {n} tokens; the limit is {SEARCH_MAX_TOKENS}")
    recorded = getattr(runtime, "recorded_search", None)  # mock: demo/autolab_examples.json
    out = recorded(req.query, req.top_n, req.max_density, req.sort) if recorded else None
    if out is not None:
        return SearchResponse(**out)
    tokens = strip_prefix(all_active(runtime, meta, SEARCH_PREFIX + req.query), SEARCH_PREFIX)
    agg = drop_dense(db, aggregate(tokens), req.max_density)
    info = dictionary(db, agg) if req.sort == "relative" else {}
    top = sorted(agg.items(), key=report_key(req.sort, info))[:req.top_n]
    info = dictionary(db, (fid for fid, _ in top))
    dens = densities(db, (fid for fid, _ in top))
    return SearchResponse(
        query=req.query, model_name=meta["model_name"],
        simulated=bool(getattr(runtime, "is_simulated", False)),
        tokens=[t["text"] for t in tokens],
        features=[SearchFeature(feature_id=fid, label=info.get(fid, (None,))[0],
                                max_activation=v["max"],
                                relative_to_max=relative(v["max"], fid, info),
                                tokens_active=v["count"], peak_token=v["peak"],
                                density=dens.get(fid, 0.0))
                  for fid, v in top])
