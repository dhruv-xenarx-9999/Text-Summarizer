import os
import re
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv


app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"
load_dotenv(BASE_DIR.parent / ".env")
DEFAULT_HF_MODEL_ID = "xenarx/text-summarizer-t5"
templates = Jinja2Templates(directory=str(BASE_DIR))


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


@lru_cache(maxsize=1)
def load_local_model():
    try:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local summarization requires torch and transformers."
        ) from exc

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()
    return model, tokenizer, device


async def summarize_with_huggingface(dialogue: str) -> str:
    api_url = os.getenv("HF_INFERENCE_API_URL")
    model_id = os.getenv("HF_MODEL_ID") or DEFAULT_HF_MODEL_ID

    if not api_url:
        api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"

    headers = {"Content-Type": "application/json"}
    token = os.getenv("HF_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json={
                    "inputs": dialogue,
                    "parameters": {
                        "max_length": 150,
                        "num_beams": 4,
                        "early_stopping": True,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Hugging Face inference request failed: {exc}"
        ) from exc

    if isinstance(payload, list) and payload:
        summary = payload[0].get("summary_text") or payload[0].get("generated_text")
    elif isinstance(payload, dict):
        summary = payload.get("summary_text") or payload.get("generated_text")
    else:
        summary = None

    if not summary:
        raise RuntimeError("Hugging Face returned no summary.")
    return summary


async def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    if not dialogue:
        raise HTTPException(status_code=400, detail="Dialogue cannot be empty.")

    if (
        os.getenv("HF_INFERENCE_API_URL")
        or os.getenv("HF_MODEL_ID")
        or not MODEL_DIR.exists()
    ):
        return await summarize_with_huggingface(dialogue)

    import torch

    model, tokenizer, device = load_local_model()
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(targets[0], skip_special_tokens=True)


@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    try:
        summary = await summarize_dialogue(dialogue_input.dialogue)
        return {"summary": summary}
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
