---
title: ContractReviewEnv
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# ContractReviewEnv: OpenEnv Legal Contract Analysis

ContractReviewEnv is a high-fidelity agent environment designed for testing the reasoning, legal comprehension, and attention-to-detail of frontier AI agents in a high-stakes context. It simulates a genuine workflow where an agent must review a contract clause-by-clause, identifying hidden risks, missing protections, and adversarial language.

## 🌟 Motivation & Real-World Utility
Legal document review is a highly analytical, fatigue-inducing task performed daily by lawyers and paralegals. An oversight in a single clause (e.g., an un-capped liability or a one-sided indemnification) can result in millions of dollars in damages. 

Most existing RL environments focus on toy games, web navigation, or coding tasks. **ContractReviewEnv fills a critical gap** by forcing agents to deeply comprehend nuanced, adversarial domain expertise rather than just navigating structural UI or syntax. 

## 🛠 Action and Observation Space

The environment implements the full OpenEnv interface using typed Pydantic models.

### Observation Space
At each step, the agent receives an `Observation` containing the entire contract metadata and the specific clause to review:
```json
{
  "contract_title": "Executive Employment Agreement",
  "contract_type": "Employment",
  "parties": {"employer": "Nexus", "employee": "Jordan"},
  "current_clause": {
    "clause_id": "C003-07",
    "clause_type": "Garden Leave",
    "text": "..."
  },
  "flags_so_far": [...],
  "running_f1": 0.85
}
```

### Action Space
The agent responds with an `Action` object determining if the clause contains a risk:
```json
{
  "clause_id": "C003-07",
  "is_flagged": true,
  "confidence": 0.95,
  "flagged_risk": {
    "clause_id": "C003-07",
    "risk_level": "high",
    "risk_type": "non_compete",
    "explanation": "This clause contains an unreasonable non-compete...",
    "suggested_fix": "Limit the restricted period to 6 months...",
    "is_missing_protection": false
  }
}
```

### Reward Function
The reward provides continuous, meaningful signal over the trajectory (range: `-0.30` to `+1.05` per step):
- **True Positives & True Negatives:** Rewarded (e.g., `+0.40` TP, `+0.10` TN).
- **Asymmetric Penalties:** Missing a risk (FN `-0.30`) is punished worse than an over-eager flag (FP `-0.20`).
- **Granular Bonuses:** Partial credit is awarded for guessing the correct severity/type (`+0.25`, `+0.20`), and providing high-quality explanations.
- **Buried Clause Bonus:** Catching adversarial legalese hiding within unrelated paragraphs grants massive bonuses.

## 🎯 Task Descriptions & Graders
We implement 3 tasks evaluated against expert attorney-annotated "Golden Sets". Graders are entirely programmatic, deterministic, and return scores strictly between `0.0` and `1.0`.

1. **Task 1: Mutual NDA (Easy)**
   - **Goal:** Catch 3 overt HIGH-risk clauses (e.g., $1k liability caps).
   - **Grader metric:** Direct Recall. Does the agent reliably spot obvious red flags?
2. **Task 2: SaaS Agreement (Medium)**
   - **Goal:** Full document review of standard commercial software terms.
   - **Grader metric:** Running F1-Score with a bonus mechanism. Requires the agent to spot missing standard protections (e.g., reciprocal IP indemnification) rather than just bad text.
3. **Task 3: Executive Employment (Hard)**
   - **Goal:** Uncovering an adversarial "buried clause" designed to exploit the agent's context window. An aggressive 24-month non-solicitation obligation is hidden deep within cross-referencing definitions.
   - **Grader metric:** Composite score (0.0 to 1.0) derived by multiplying the F1-score with a massive "buried multiplier" and factoring in severity-accuracy penalties. Genuine challenge for frontier models.

## 🚀 Setup & Usage Instructions

### Prerequisites
- Python 3.11+
- OpenAI API Key (or Groq API Key)

### Installation
The evaluation environment strictly requires `OPENAI_API_KEY` to be passed as an environment variable to pass automated hackathon validations. However, you can configure it to point to Groq for free local testing:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_groq_api_key_here
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export MODEL=llama-3.1-8b-instant
```

### Running the Baseline
The `inference.py` script uses the official `openai` Python client. It defaults to `gpt-4o-mini` if no `MODEL` is provided via the environment (to pass Hackathon validation), but will use Groq if configured with a BASE_URL.
```bash
python inference.py
```

### Docker Execution
```bash
docker build -t contract-review-env .
docker run -e OPENAI_API_KEY=your_key contract-review-env
```

## 📊 Baseline Scores
The following reproducible baseline scores were achieved using the `llama-3.1-8b-instant` default model via the automated grader over the 3 tasks:

- **Task 1 (Easy):** `0.667`
- **Task 2 (Medium):** `1.000`
- **Task 3 (Hard):** `0.988`
- **AVERAGE SCORE:** `0.885`

## 🛡 OpenEnv Compliance
This environment faithfully implements the OpenEnv standard:
- `openenv.yaml` metadata included.
- Typed Pydantic `Action` and `Observation` models.
- Standard `step()`, `reset()`, and `state()` API.
