import requests
from typing import Dict, Any
from models import Observation, StepResult, EnvState, Action

class ContractReviewEnvClient:
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url

    def reset(self) -> Observation:
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        data = response.json()
        return Observation(**data["observation"])

    def step(self, action: Action) -> StepResult:
        response = requests.post(f"{self.base_url}/step", json=action.model_dump())
        response.raise_for_status()
        data = response.json()
        return StepResult(
            observation=Observation(**data["observation"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            info=data["info"]
        )

    def state(self) -> EnvState:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return EnvState(**response.json())
