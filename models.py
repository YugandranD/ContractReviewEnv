from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum
from dataclasses import dataclass

class RiskLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    NONE   = "none"

class RiskType(str, Enum):
    LIABILITY_CAP        = "liability_cap"
    IP_OWNERSHIP         = "ip_ownership"
    TERMINATION          = "termination"
    INDEMNIFICATION      = "indemnification"
    GOVERNING_LAW        = "governing_law"
    DATA_PRIVACY         = "data_privacy"
    NON_COMPETE          = "non_compete"
    AMBIGUOUS_LANGUAGE   = "ambiguous_language"
    MISSING_PROTECTION   = "missing_protection"
    UNILATERAL_CHANGE    = "unilateral_change"
    AUTO_RENEWAL         = "auto_renewal"
    PAYMENT_TERMS        = "payment_terms"
    CONFIDENTIALITY      = "confidentiality"
    DISPUTE_RESOLUTION   = "dispute_resolution"
    CLEAN                = "clean"

@dataclass
class Clause:
    clause_id: str
    clause_type: str
    text: str
    risk_level: RiskLevel
    risk_type: RiskType
    is_missing_protection: bool
    annotation: str
    is_buried: bool = False

@dataclass
class Contract:
    contract_id: str
    contract_type: str
    title: str
    parties: dict
    clauses: list

class FlaggedRisk(BaseModel):
    clause_id: str
    risk_level: RiskLevel
    risk_type: RiskType
    explanation: str = Field(..., min_length=20, max_length=500)
    suggested_fix: str = Field(..., min_length=10, max_length=400)
    is_missing_protection: bool = False

class Action(BaseModel):
    clause_id: str
    is_flagged: bool
    flagged_risk: Optional[FlaggedRisk] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.is_flagged and self.flagged_risk is None:
            raise ValueError("flagged_risk required when is_flagged=True")
        if not self.is_flagged and self.flagged_risk is not None:
            raise ValueError("flagged_risk must be None when is_flagged=False")

class Observation(BaseModel):
    contract_id: str
    contract_type: str
    contract_title: str
    parties: dict
    current_clause: dict
    clause_index: int
    total_clauses: int
    clauses_reviewed: int
    running_f1: float
    flags_so_far: list[dict]
    time_step: int
    done: bool = False
    reward: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class EnvState(BaseModel):
    contract_id: str
    clauses_reviewed: int
    total_clauses: int
    true_positives: int
    false_positives: int
    false_negatives: int
    running_precision: float
    running_recall: float
    running_f1: float
    done: bool
