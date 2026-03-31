"""
3 Tasks with programmatic graders. Easy → Medium → Hard.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from env import ContractReviewEnv, Action, FlaggedRisk, RiskLevel, RiskType

class Task1_NDA_HighRisk:
    task_id = "task_1_nda_high_risk"
    difficulty = "easy"
    description = "Review NDA. Catch all 3 HIGH-risk clauses: $1k liability cap, archival copy loophole, biased arbitrator."
    HIGH_RISK_IDS = {"C001-04", "C001-07", "C001-08"}

    def run(self, agent_fn) -> dict:
        env = ContractReviewEnv(contract_id="C-001")
        obs = env.reset()
        flagged_high = set()
        step_rewards = []
        while not env.state().done:
            action = agent_fn(obs)
            result = env.step(action)
            step_rewards.append(result.reward)
            if action.is_flagged and action.flagged_risk and action.flagged_risk.risk_level == RiskLevel.HIGH and action.clause_id in self.HIGH_RISK_IDS:
                flagged_high.add(action.clause_id)
            obs = result.observation
        recall = len(flagged_high) / len(self.HIGH_RISK_IDS)
        return {"task_id": self.task_id, "score": round(recall, 3), "high_risk_caught": list(flagged_high), "high_risk_missed": list(self.HIGH_RISK_IDS - flagged_high), "passed": recall >= 0.67}

class Task2_SaaS_FullReview:
    task_id = "task_2_saas_full_review"
    difficulty = "medium"
    description = "Full SaaS Agreement review. F1 scored. Must catch one-sided indemnification (C002-07) for +0.10 bonus."

    def run(self, agent_fn) -> dict:
        env = ContractReviewEnv(contract_id="C-002")
        obs = env.reset()
        step_rewards = []
        indemnification_flagged = False
        while not env.state().done:
            action = agent_fn(obs)
            result = env.step(action)
            step_rewards.append(result.reward)
            if action.clause_id == "C002-07" and action.is_flagged and action.flagged_risk and action.flagged_risk.risk_level == RiskLevel.HIGH:
                indemnification_flagged = True
            obs = result.observation
        state = env.state()
        bonus = 0.10 if indemnification_flagged else 0.0
        score = round(min(state.running_f1 + bonus, 1.0), 3)
        return {"task_id": self.task_id, "score": score, "base_f1": state.running_f1, "precision": state.running_precision, "recall": state.running_recall, "tp": state.true_positives, "fp": state.false_positives, "fn": state.false_negatives, "indemnification_flagged": indemnification_flagged}

class Task3_Employment_Adversarial:
    task_id = "task_3_employment_adversarial"
    difficulty = "hard"
    description = "Employment Agreement with 1 buried clause (C003-07). Score = F1 × buried_multiplier + key_recall + severity_accuracy."
    BURIED_CLAUSE_ID = "C003-07"
    KEY_HIGH_RISK = {"C003-02", "C003-03", "C003-06", "C003-07"}

    def run(self, agent_fn) -> dict:
        env = ContractReviewEnv(contract_id="C-003")
        obs = env.reset()
        step_rewards = []
        buried_caught = False
        key_risks_caught = set()
        severity_correct = severity_total = 0
        while not env.state().done:
            action = agent_fn(obs)
            result = env.step(action)
            step_rewards.append(result.reward)
            gt = result.info["ground_truth"]
            if action.is_flagged and action.flagged_risk:
                if action.clause_id == self.BURIED_CLAUSE_ID:
                    buried_caught = True
                if action.clause_id in self.KEY_HIGH_RISK:
                    key_risks_caught.add(action.clause_id)
                if gt["should_flag"]:
                    severity_total += 1
                    if action.flagged_risk.risk_level == gt["risk_level"]:
                        severity_correct += 1
            obs = result.observation
        state = env.state()
        buried_mult = 1.25 if buried_caught else 0.80
        key_recall = len(key_risks_caught) / len(self.KEY_HIGH_RISK)
        sev_acc = (severity_correct / severity_total) if severity_total > 0 else 0.0
        score = round(min(state.running_f1 * buried_mult * 0.5 + key_recall * 0.3 + sev_acc * 0.2, 1.0), 3)
        return {"task_id": self.task_id, "score": score, "base_f1": state.running_f1, "buried_clause_caught": buried_caught, "buried_multiplier": buried_mult, "key_risks_caught": list(key_risks_caught), "key_recall": round(key_recall, 3), "severity_accuracy": round(sev_acc, 3)}

ALL_TASKS = [Task1_NDA_HighRisk, Task2_SaaS_FullReview, Task3_Employment_Adversarial]
