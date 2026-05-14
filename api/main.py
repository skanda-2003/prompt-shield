from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict


# --- request / response models ---

# single reusable type so both request models share the same prompt constraints
_Prompt = Annotated[str, Field(min_length=1, max_length=4096)]


class ClassifyRequest(BaseModel):
    prompt: _Prompt


class BatchClassifyRequest(BaseModel):
    # min_length=1 rejects an empty list; max_length=100 caps batch size
    prompts: list[_Prompt] = Field(..., min_length=1, max_length=100)


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --- endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    result = predict(request.prompt)
    return ClassifyResponse(**result)


@app.post("/classify/batch", response_model=list[ClassifyResponse])
def classify_batch(request: BatchClassifyRequest) -> list[ClassifyResponse]:
    return [ClassifyResponse(**predict(p)) for p in request.prompts]
