"""
Baseline inference using OpenAI API. Reads OPENAI_API_KEY from env.
Run: python baseline.py
"""

import os, json
from dotenv import load_dotenv
from openai import OpenAI
import openai
from models import Action, FlaggedRisk, RiskLevel, RiskType, Observation
from server.tasks import ALL_TASKS

load_dotenv()

# Configurable model mapping, defaults to OpenAI's gpt-4o-mini as required by judges
MODEL = os.environ.get("MODEL", "gpt-4o-mini")

# The OpenAI client automatically picks up OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
client = OpenAI()
SYSTEM_PROMPT = """You are a senior legal contract review specialist with 15+ years of experience.
Review each contract clause and respond ONLY with valid JSON — no markdown, no preamble.

JSON schema:
{
  "clause_id": "<exact clause_id provided>",
  "is_flagged": true or false,
  "confidence": 0.0 to 1.0,
  "flagged_risk": {
    "clause_id": "<same clause_id>",
    "risk_level": "high" | "medium" | "low",
    "risk_type": "<one of: liability_cap, ip_ownership, termination, indemnification, governing_law, data_privacy, non_compete, ambiguous_language, missing_protection, unilateral_change, auto_renewal, payment_terms, confidentiality, dispute_resolution, clean>",
    "explanation": "<detailed risk explanation, min 30 words>",
    "suggested_fix": "<specific recommended change, min 15 words>",
    "is_missing_protection": true or false
  }
}

If clause is clean: is_flagged=false, flagged_risk=null.

Risk guide:
- HIGH: significant financial loss, legal liability, or loss of rights
- MEDIUM: unfavorable terms, vague language, missing standard protections
- LOW: minor issues, overly broad but typical language

Watch for: dense cross-referencing legalese that buries obligations, 'without limitation' or 'at sole discretion', short liability caps, one-sided indemnification, unilateral modification rights, undefined Exhibits or Policies."""

VALID_RISK_TYPES = {rt.value for rt in RiskType}
VALID_RISK_LEVELS = {rl.value for rl in RiskLevel}

def build_prompt(obs: Observation) -> str:
    clause = obs.current_clause
    flags_summary = ""
    if obs.flags_so_far:
        flags_summary = "\n\nFlagged so far:\n" + "\n".join(f"  - {f['clause_id']}: {f['risk_level']} {f['risk_type']}" for f in obs.flags_so_far)
    return f"""CONTRACT: {obs.contract_title} ({obs.contract_type})
Parties: {json.dumps(obs.parties)}
Progress: Clause {obs.clause_index + 1} of {obs.total_clauses} | Running F1: {obs.running_f1:.3f}
{flags_summary}

--- CLAUSE TO REVIEW ---
ID:   {clause['clause_id']}
Type: {clause['clause_type']}
Text: {clause['text']}
--- END CLAUSE ---

Respond with JSON only."""

def parse_action(raw: str, clause_id: str) -> Action:
    try:
        raw = raw.strip()
        # Handle markdown JSON blocks
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        # Clean potential non-JSON preamble/postamble
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]

        data = json.loads(raw)
        is_flagged = bool(data.get("is_flagged", False))
        confidence = float(data.get("confidence", 0.5))
        if is_flagged and data.get("flagged_risk"):
            fr = data["flagged_risk"]
            rl = fr.get("risk_level", "medium")
            rt = fr.get("risk_type", "ambiguous_language")
            if rl not in VALID_RISK_LEVELS: rl = "medium"
            if rt not in VALID_RISK_TYPES: rt = "ambiguous_language"
            flagged_risk = FlaggedRisk(
                clause_id=clause_id,
                risk_level=RiskLevel(rl),
                risk_type=RiskType(rt),
                explanation=str(fr.get("explanation", "Risk identified."))[:500],
                suggested_fix=str(fr.get("suggested_fix", "Review and revise."))[:400],
                is_missing_protection=bool(fr.get("is_missing_protection", False)),
            )
        else:
            is_flagged = False; flagged_risk = None
        return Action(clause_id=clause_id, is_flagged=is_flagged, flagged_risk=flagged_risk, confidence=min(1.0, max(0.0, confidence)))
    except Exception as e:
        print(f"  [PARSE ERROR] {e} | RAW: {raw[:100]}")
        return Action(clause_id=clause_id, is_flagged=False, confidence=0.3)

import time

def agent_fn(obs: Observation) -> Action:
    clause_id = obs.current_clause["clause_id"]
    if clause_id == "DONE":
        return Action(clause_id="DONE", is_flagged=False, confidence=1.0)
        
    max_retries = 10
    base_sleep = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(obs)}],
                temperature=0.0, max_tokens=600
            )
            return parse_action(response.choices[0].message.content, clause_id)
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                sleep_time = base_sleep * (2 ** attempt)
                print(f"  [RATE LIMIT] Exceeded TPM. Sleeping for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                raise e
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [API ERROR] {e}. Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"  [API ERROR FINAL] {e}")
                return Action(clause_id=clause_id, is_flagged=False, confidence=0.3)

def main():
    sep = "=" * 60
    print(f"\n{sep}\n  ContractReviewEnv Baseline [{MODEL}]\n{sep}\n")
    scores = []
    for TaskClass in ALL_TASKS:
        task = TaskClass()
        print(f">> {task.task_id} [{task.difficulty}]")
        result = task.run(agent_fn)
        scores.append(result["score"])
        print(f"  Score: {result['score']:.3f}")
        for k, v in result.items():
            if k not in ("task_id", "score"): print(f"  {k}: {v}")
        print()
    avg = sum(scores) / len(scores)
    print(f"{sep}\n  AVERAGE: {avg:.3f} | SCORES: {' | '.join(f'{s:.3f}' for s in scores)}\n{sep}\n")

if __name__ == "__main__":
    main()
