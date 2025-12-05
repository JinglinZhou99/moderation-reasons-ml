from pydantic import BaseModel
from typing import List, Dict, Optional

LABELS = ["violence", "sexual", "hate"]

class PredictRequest(BaseModel):
    text: str
    model: Optional[str] = "lr"  # "lr"|"bert"

class PredictResponse(BaseModel):
    labels: List[str]
    probs: List[float]
    preds: List[int]

class Span(BaseModel):
    start: int
    end: int
    text: str

class ExplainRequest(BaseModel):
    text: str
    model: Optional[str] = "lr"

class ExplainResponse(BaseModel):
    spans: Dict[str, List[Span]]
    reasons: Dict[str, str]

class MetricsResponse(BaseModel):
    model: str
    labels: List[str]
    micro_f1: float
    macro_f1: float
    per_label: Dict[str, Dict[str, float]]

class BatchRow(BaseModel):
    text: str
    violence: Optional[int] = None
    sexual: Optional[int] = None
    hate: Optional[int] = None

class BatchPredictResponse(BaseModel):
    labels: List[str]
    rows: List[Dict]
    model: str

class BatchEvalResponse(BaseModel):
    model: str
    labels: List[str]
    micro_f1: float
    macro_f1: float
    per_label: Dict[str, Dict[str, float]]
    rows: List[Dict]  # includes predictions
