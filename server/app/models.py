from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

LabelStatus = Literal["unlabelled", "named", "ambiguous"]


class ErrorBody(BaseModel):
    code: Literal["INVALID_INPUT", "TEXT_TOO_LONG", "FEATURE_NOT_FOUND", "RUN_NOT_FOUND",
                  "NOT_FOUND", "MODEL_NOT_LOADED", "SCHEMA_MISMATCH", "INTERNAL"]
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class Health(BaseModel):
    status: Literal["ok"]
    mode: Literal["mock", "real"]
    is_synthetic: bool
    schema_version: str
    model_name: str
    hook_layer: int
    d_model: int
    d_sae: int
    k: int
    run_id: str


class FeatureListItem(BaseModel):
    feature_id: int
    label: Optional[str]
    label_status: LabelStatus
    density: float = Field(description="Fraction of tokens where active, 0 to 1")
    max_activation: float
    is_dead: bool


class FeatureList(BaseModel):
    total: int = Field(description="Matches after filters, before pagination")
    limit: int
    offset: int
    items: list[FeatureListItem]


class Example(BaseModel):
    rank: int
    tokens: list[str] = Field(description="Leading spaces kept exactly as tokenised")
    activations: list[float]
    peak_index: int
    max_activation: float


class LogitEntry(BaseModel):
    token: str
    score: float


class Logits(BaseModel):
    promoted: list[LogitEntry]
    suppressed: list[LogitEntry]


class FeatureDetail(FeatureListItem):
    mean_activation: Optional[float]
    notes: Optional[str]
    examples: list[Example]
    logits: Logits


class InspectRequest(BaseModel):
    text: str
    top_n: int = Field(default=5, ge=1, le=20)

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class TokenFeature(BaseModel):
    feature_id: int
    activation: float
    label: Optional[str]


class InspectToken(BaseModel):
    index: int
    text: str = Field(description="Leading space kept exactly as tokenised")
    is_outlier_position: bool
    features: list[TokenFeature]


class InspectResponse(BaseModel):
    model_name: str
    tokens: list[InspectToken]


class SteerRequest(BaseModel):
    prompt: str
    feature_id: int
    mode: Literal["amplify", "ablate"]
    strength: float = Field(default=1.0, ge=0, le=3,
                            description="1.0 = the feature's own max activation; amplify only")
    max_new_tokens: int = Field(default=60, ge=1, le=120)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v


class SteerFeature(BaseModel):
    feature_id: int
    label: Optional[str]


class SteerResponse(BaseModel):
    feature: SteerFeature
    mode: Literal["amplify", "ablate"]
    strength: float
    seed: int
    baseline_text: str
    steered_text: str


# ---- X6 sentence report and X7 text compare (aggregate /inspect) ----

class _AnalysisFeature(BaseModel):
    feature_id: int
    label: Optional[str]


class ReportRequest(BaseModel):
    text: str
    top_n: int = Field(default=5, ge=1, le=20, description="features per sentence")
    max_density: Optional[float] = Field(
        default=None, gt=0, le=1,
        description="hide features whose density (share of tokens they fire on) is above this; "
                    "off by default")
    sort: Literal["activation", "relative"] = Field(
        default="activation",
        description="activation: rank by raw max activation (default). relative: rank by "
                    "relative_to_max, how strongly each feature fired for that feature")

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class ReportFeature(_AnalysisFeature):
    max_activation: float = Field(description="highest activation on this sentence's tokens")
    relative_to_max: Optional[float] = Field(
        description="max_activation / the feature's max_activation in the dictionary")
    tokens_active: int = Field(description="tokens in the sentence where the feature was active")
    peak_token: int = Field(description="index (as in /inspect) of the token with the highest "
                                        "activation")


class ReportSentence(BaseModel):
    index: int
    text: str
    start_token: int = Field(description="first token index, as in /inspect")
    end_token: int = Field(description="one past the last token index")
    features: list[ReportFeature]


class ReportResponse(BaseModel):
    """X6: the features active in each sentence, ranked by max activation (token 0 skipped,
    as in /inspect). Shows which features were active, not what the sentence means."""
    model_name: str
    simulated: bool
    sentences: list[ReportSentence]


class TextCompareRequest(BaseModel):
    text_a: str
    text_b: str
    top_n: int = Field(default=10, ge=1, le=50)
    max_density: Optional[float] = Field(
        default=None, gt=0, le=1,
        description="hide features whose density (share of tokens they fire on) is above this; "
                    "off by default")
    sort: Literal["activation", "relative"] = Field(
        default="activation",
        description="activation: rank by raw max activation (default). relative: rank by "
                    "relative_to_max, how strongly each feature fired for that feature")

    @field_validator("text_a", "text_b")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class TextCompareFeature(_AnalysisFeature):
    max_activation_a: float = Field(description="0 when not active in text_a")
    max_activation_b: float
    difference: float = Field(description="max_activation_b - max_activation_a")
    tokens_active_a: int
    tokens_active_b: int
    relative_to_max_a: Optional[float]
    relative_to_max_b: Optional[float]


class TextCompareResponse(BaseModel):
    """X7: features whose max activation differs most between two texts (|difference|)."""
    model_name: str
    simulated: bool
    tokens_a: int = Field(description="number of tokens in text_a")
    tokens_b: int
    features: list[TextCompareFeature]


# ---- W8 search: features that fired most on the query ----

class SearchRequest(BaseModel):
    query: str = Field(description="1 to 32 tokens of the query itself")
    top_n: int = Field(default=10, ge=1, le=50)
    max_density: Optional[float] = Field(
        default=None, gt=0, le=1,
        description="hide features whose density (share of tokens they fire on) is above this; "
                    "off by default")
    sort: Literal["activation", "relative"] = Field(
        default="activation",
        description="activation: rank by raw max activation (default). relative: rank by "
                    "relative_to_max")

    @field_validator("query")
    @classmethod
    def _query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v


class SearchFeature(_AnalysisFeature):
    max_activation: float = Field(description="highest activation on the query's tokens")
    relative_to_max: Optional[float]
    tokens_active: int
    peak_token: int = Field(description="index into `tokens` of the strongest token")
    density: float = Field(description="share of tokens this feature fires on in the dictionary")


class SearchResponse(BaseModel):
    """W8: the features that fired most on the words of the query. It shows which features
    fired on the query, not what the query means."""
    query: str
    model_name: str
    simulated: bool
    tokens: list[str] = Field(description="the query's own tokens, as the model split them")
    features: list[SearchFeature]


# ---- W4 co-fire counts (fields in docs/phase4-data-files.md) ----

class CofirePartner(BaseModel):
    feature_id: int
    label: Optional[str]
    co_count: int = Field(description="tokens in the sample on which both features were active")
    jaccard: float = Field(description="co_count / (count_a + count_b - co_count), 4 decimals")
    in_dictionary: bool = Field(description="the partner exists in this server's database; "
                                            "the UI links a partner only when true")


class CofireResponse(BaseModel):
    """W4: the features that fired on the same tokens as this one in a token sample, best
    (highest Jaccard) first, up to 10. Shared tokens only; not a statement about meaning."""
    feature_id: int
    available: bool = Field(description="false when the co-fire file is missing, belongs to "
                                        "another run, or has no entry for this feature")
    simulated: bool
    tokens_scanned: Optional[int] = Field(description="sample size; null when not available")
    seed: Optional[int]
    min_count: Optional[int] = Field(description="a pair needs at least this many shared tokens")
    partners: list[CofirePartner]


# ---- Phase 4 steering checks (fields as in docs/phase4-ml-api.md) ----

DEFAULT_SWEEP_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


class _PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=60, ge=1, le=120)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v


class SteerCheckResponse(BaseModel):
    """X1: baseline vs steered (same seed) and the target feature's activation on each
    finished text, re-encoded without steering. It reports whether the feature fires on the
    resulting text; it does not say the output is better."""
    feature: SteerFeature
    mode: Literal["amplify", "ablate"]
    strength: float
    seed: int
    max_new_tokens: int
    baseline_text: str
    steered_text: str
    differs: bool
    target_activation_baseline: float = Field(
        description="max activation of the target feature over the new tokens of the baseline")
    target_activation_steered: float
    target_activation_baseline_mean: float
    target_activation_steered_mean: float
    activation_change: float = Field(
        description="(steered max - baseline max) / the feature's max_activation")
    activation_verdict: Literal["higher", "lower", "about the same"]


class SteerCompareResponse(BaseModel):
    """X3: baseline and steered with the same seed, compared token by token. Indices are
    0-based over newly generated tokens. In mock mode tokens are a word split."""
    feature: SteerFeature
    mode: Literal["amplify", "ablate"]
    strength: float
    seed: int
    max_new_tokens: int
    baseline_text: str
    steered_text: str
    prompt_tokens: int
    baseline_new_tokens: int
    steered_new_tokens: int
    diverges: bool
    first_divergence_index: Optional[int]
    identical_prefix_tokens: list[str]
    identical_prefix_text: str
    baseline_from_divergence: list[str]
    baseline_from_divergence_text: str
    steered_from_divergence: list[str]
    steered_from_divergence_text: str
    tokens_changed: int
    tokens_compared_from_divergence: int
    percent_changed_from_divergence: Optional[float]
    percent_changed_raw: float
    percent_changed_raw_note: str


class SteerSweepRequest(_PromptRequest):
    feature_id: int
    strengths: list[float] = Field(default_factory=lambda: list(DEFAULT_SWEEP_STRENGTHS),
                                   min_length=1, max_length=12,
                                   description="amplify strengths, each 0 to 3, order kept")

    @field_validator("strengths")
    @classmethod
    def _strengths_in_range(cls, v: list[float]) -> list[float]:
        bad = [s for s in v if not 0 <= s <= 3]
        if bad:
            raise ValueError(f"each strength must be from 0 to 3, got {bad}")
        return v


class SweepMetrics(BaseModel):
    new_tokens: int
    repeated_token_fraction: float
    repeated_trigram_fraction: float
    non_word_fraction: float
    replacement_chars: int


class SweepResult(BaseModel):
    strength: float
    text: str
    differs_from_baseline: bool
    first_divergence_index: Optional[int]
    degenerate: bool
    reason: Optional[str]
    metrics: SweepMetrics


class SteerSweepResponse(BaseModel):
    """X2: amplify at several strengths with one seed. `degenerate` flags repetition loops,
    non-word runs and broken characters; it is not a quality score."""
    feature: SteerFeature
    mode: Literal["amplify"]
    seed: int
    max_new_tokens: int
    baseline_text: str
    baseline_degenerate: bool
    baseline_reason: Optional[str]
    results: list[SweepResult]


class PromptOnlyRequest(_PromptRequest):
    pass


class PromptOnlyResponse(BaseModel):
    """X4: what the model writes from the prompt alone, same seed, no steering."""
    prompt: str
    seed: int
    max_new_tokens: int
    text: str
    continuation: str
    new_tokens: int


class SteeringExample(BaseModel):
    example_id: int
    feature_id: int
    mode: Literal["amplify", "ablate"]
    strength: float
    prompt: str
    seed: int
    baseline_text: str
    steered_text: str
    is_curated: bool


class SteeringExampleList(BaseModel):
    items: list[SteeringExample]


class Run(BaseModel):
    run_id: str
    model_name: str
    hook_layer: int
    d_model: int
    d_sae: int
    k: int
    n_train_activations: Optional[int]
    checkpoint_name: Optional[str]
    created_at: str
    notes: Optional[str]
    is_primary: bool


class RunList(BaseModel):
    items: list[Run]


class MetricValue(BaseModel):
    name: str
    value: float = Field(description="fvu and ce_recovered are fractions; dead_pct is PERCENT 0-100")


class DensityBin(BaseModel):
    bin_lo: float = Field(description="log10(density), live features only")
    bin_hi: float
    count: int


class CurvePoint(BaseModel):
    step: int
    loss: float
    fvu: Optional[float]
    dead_pct: Optional[float] = Field(description="PERCENT 0-100")


class MetricsResponse(BaseModel):
    run_id: str
    metrics: list[MetricValue]
    density_hist: list[DensityBin]
    training_curve: list[CurvePoint]


# Fixed descriptions so contracts/openapi.json does not change with the Python version
# (3.13 renamed the 413 and 422 reason phrases).
_ERROR_DESCRIPTIONS = {404: "Not Found", 413: "Content Too Large", 422: "Unprocessable Content",
                       500: "Internal Server Error", 503: "Service Unavailable"}


def error_responses(*statuses: int) -> dict:
    return {s: {"model": ErrorResponse, "description": _ERROR_DESCRIPTIONS[s]} for s in statuses}
