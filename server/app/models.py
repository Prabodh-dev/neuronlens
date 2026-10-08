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
