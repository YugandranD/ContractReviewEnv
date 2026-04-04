import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import uvicorn
from fastapi import FastAPI, HTTPException
from models import Action
sys.path.insert(0, os.path.dirname(__file__))
from env import ContractReviewEnv

app = FastAPI(title="ContractReviewEnv OpenEnv Wrapper")
environment = ContractReviewEnv()

@app.get("/")
def health_check():
    return {"status": "ok", "environment": "ContractReviewEnv"}

@app.post("/reset")
def reset():
    obs = environment.reset()
    return {
        "observation": obs.model_dump(),
        "reward": None,
        "done": False
    }

@app.post("/step")
def step(action: Action):
    try:
        step_result = environment.step(action)
        return {
            "observation": step_result.observation.model_dump(),
            "reward": step_result.reward,
            "done": step_result.done
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    """Entry point for the OpenEnv server."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()

@app.get("/state")
def state():
    return environment.state().model_dump()
