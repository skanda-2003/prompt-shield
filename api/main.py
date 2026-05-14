from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict


# --- request / response models ---

class ClassifyRequest(BaseModel):
    # Field() lets us add validation constraints alongside the type hint
    # min_length=1 rejects empty strings; max_length=4096 prevents abuse
    prompt: str = Field(..., min_length=1, max_length=4096)


class ClassifyResponse(BaseModel):
    is_safe: bool
    confidence: float
    # attack_type is always present - None when safe, a string when unsafe
    # this means API consumers never have to handle a missing key
    attack_type: str | None


# --- startup ---

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # run a dummy prediction at startup to trigger model loading
    # without this, the first real request would take ~2s to load the model
    predict("warmup")
    yield  # everything after yield runs on shutdown (nothing to do here)


app = FastAPI(
    title="PromptShield",
    description="Detects prompt injection and jailbreak attempts before they reach an LLM.",
    lifespan=lifespan,
)


# --- endpoint ---

@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    result = predict(request.prompt)
    return ClassifyResponse(
        is_safe=result["is_safe"],
        confidence=result["confidence"],
        attack_type=result["attack_type"],
    )
