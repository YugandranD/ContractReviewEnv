"""
Contract Review OpenEnv Environment
OpenEnv spec: step() / reset() / state() with typed Pydantic models
"""

from __future__ import annotations
from typing import Optional, Any
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models import FlaggedRisk, Action, Observation, StepResult, EnvState, RiskLevel, RiskType, Clause, Contract
sys.path.insert(0, os.path.dirname(__file__))
from contracts.contract_data import ALL_CONTRACTS

def clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1 as required by OpenEnv validator."""
    return max(0.01, min(0.99, score))

class ContractReviewEnv:
    metadata = {
        "name": "ContractReviewEnv",
        "version": "1.0.0",
        "description": "AI agent reviews legal contracts clause by clause, flagging risks scored against expert annotations using Precision/Recall/F1.",
    }
    SEVERITY_ORDER = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]

    def __init__(self, contract_id: str = "C-001"):
        if contract_id not in ALL_CONTRACTS:
            raise ValueError(f"Unknown contract_id: {contract_id}. Valid: {list(ALL_CONTRACTS.keys())}")
        self.contract_id = contract_id
        self.contract: Contract = ALL_CONTRACTS[contract_id]
        self._clause_index = 0
        self._tp = self._fp = self._fn = 0
        self._time_step = 0
        self._done = False
        self._flags: list[dict] = []

    def reset(self) -> Observation:
        self._clause_index = 0
        self._tp = self._fp = self._fn = 0
        self._time_step = 0
        self._done = False
        self._flags = []
        return self._make_observation(done=False, reward=None, info={})

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        clause = self.contract.clauses[self._clause_index]
        if action.clause_id != clause.clause_id:
            raise ValueError(f"Action clause_id '{action.clause_id}' != current '{clause.clause_id}'")
        reward_breakdown = self._compute_reward(action, clause)
        if action.is_flagged:
            self._flags.append({
                "clause_id": clause.clause_id,
                "risk_level": action.flagged_risk.risk_level,
                "risk_type": action.flagged_risk.risk_type,
                "explanation": action.flagged_risk.explanation,
            })
        self._clause_index += 1
        self._time_step += 1
        self._done = self._clause_index >= len(self.contract.clauses)
        
        info = {
            "reward_breakdown": reward_breakdown,
            "ground_truth": {
                "risk_level": clause.risk_level,
                "risk_type": clause.risk_type,
                "should_flag": clause.risk_level != RiskLevel.NONE,
                "is_missing_protection": clause.is_missing_protection,
                "annotation": clause.annotation,
                "is_buried": clause.is_buried,
            },
            "tp": self._tp, "fp": self._fp, "fn": self._fn,
            "running_f1": self._f1(),
        }
        
        step_reward = reward_breakdown["step_reward"]
        obs = self._make_observation(done=self._done, reward=step_reward, info=info) if not self._done else self._make_terminal_obs(done=True, reward=step_reward, info=info)
        
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=self._done,
            info=info
        )

    def state(self) -> EnvState:
        return EnvState(
            contract_id=self.contract_id,
            clauses_reviewed=self._clause_index,
            total_clauses=len(self.contract.clauses),
            true_positives=self._tp,
            false_positives=self._fp,
            false_negatives=self._fn,
            running_precision=clamp_score(self._precision()),
            running_recall=clamp_score(self._recall()),
            running_f1=clamp_score(self._f1()),
            done=self._done,
        )

    def _compute_reward(self, action: Action, clause: Clause) -> dict:
        should_flag = clause.risk_level != RiskLevel.NONE
        if action.is_flagged and should_flag:
            self._tp += 1; detection = 0.40
        elif action.is_flagged and not should_flag:
            self._fp += 1; detection = -0.20
        elif not action.is_flagged and should_flag:
            self._fn += 1; detection = -0.30
        else:
            detection = 0.10

        severity_acc = type_acc = missing_prot = explanation_qual = buried_bonus = 0.0

        if action.is_flagged and should_flag and action.flagged_risk:
            pred_sev = action.flagged_risk.risk_level
            true_sev = clause.risk_level
            if pred_sev == true_sev:
                severity_acc = 0.25
            else:
                dist = abs(self.SEVERITY_ORDER.index(pred_sev) - self.SEVERITY_ORDER.index(true_sev))
                severity_acc = max(0.0, 0.25 - dist * 0.10)
            if action.flagged_risk.risk_type == clause.risk_type:
                type_acc = 0.20
            if action.flagged_risk.is_missing_protection == clause.is_missing_protection:
                missing_prot = 0.05
            expl = action.flagged_risk.explanation.lower()
            explanation_qual = min(0.05, len(expl) / 4000) + min(0.05, len(action.flagged_risk.suggested_fix) / 2000)

        if clause.is_buried and action.is_flagged:
            buried_bonus = 0.15

        step_reward = round(detection + severity_acc + type_acc + missing_prot + explanation_qual + buried_bonus, 4)
        return {
            "step_reward": step_reward,
            "detection": round(detection, 4),
            "severity_accuracy": round(severity_acc, 4),
            "type_accuracy": round(type_acc, 4),
            "missing_protection": round(missing_prot, 4),
            "explanation_quality": round(explanation_qual, 4),
            "buried_bonus": round(buried_bonus, 4),
            "running_f1": round(self._f1(), 4),
        }

    def _precision(self):
        d = self._tp + self._fp; return self._tp / d if d > 0 else 1.0
    def _recall(self):
        d = self._tp + self._fn; return self._tp / d if d > 0 else 1.0
    def _f1(self):
        p, r = self._precision(), self._recall()
        return round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0

    def _clause_to_dict(self, clause: Clause) -> dict:
        return {"clause_id": clause.clause_id, "clause_type": clause.clause_type, "text": clause.text}

    def _make_observation(self, done: bool = False, reward: Optional[float] = None, info: Optional[dict] = None) -> Observation:
        clause = self.contract.clauses[self._clause_index]
        return Observation(
            contract_id=self.contract.contract_id, contract_type=self.contract.contract_type,
            contract_title=self.contract.title, parties=self.contract.parties,
            current_clause=self._clause_to_dict(clause), clause_index=self._clause_index,
            total_clauses=len(self.contract.clauses), clauses_reviewed=self._clause_index,
            running_f1=self._f1(), flags_so_far=self._flags.copy(), time_step=self._time_step,
            done=done, reward=reward, metadata=info or {}
        )

    def _make_terminal_obs(self, done: bool = True, reward: Optional[float] = None, info: Optional[dict] = None) -> Observation:
        return Observation(
            contract_id=self.contract.contract_id, contract_type=self.contract.contract_type,
            contract_title=self.contract.title, parties=self.contract.parties,
            current_clause={"clause_id": "DONE", "clause_type": "Review Complete", "text": "All clauses reviewed."},
            clause_index=self._clause_index, total_clauses=len(self.contract.clauses),
            clauses_reviewed=self._clause_index, running_f1=self._f1(),
            flags_so_far=self._flags.copy(), time_step=self._time_step,
            done=done, reward=reward, metadata=info or {}
        )
