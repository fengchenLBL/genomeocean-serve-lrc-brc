import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import json

app = FastAPI()
logger = logging.getLogger(__name__)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    logger.info("Starting tokenizer loading...")
    tokenizer = AutoTokenizer.from_pretrained(
        "pGenomeOcean/GenomeOcean-4B",
        trust_remote_code=True,
        padding_side="left",
    )
    logger.info("Tokenizer loaded successfully.")
    logger.info("Starting model loading...")
    model = AutoModelForCausalLM.from_pretrained(
        "pGenomeOcean/GenomeOcean-4B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    logger.info("Model loaded successfully.")

# Load model on startup (sync)
@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    if model is not None and tokenizer is not None:
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "loading"})

@app.post("/generate")
async def generate_sequences(request: Request):
    body_bytes = await request.body()
    body = json.loads(body_bytes)
    instances = body.get("instances", [])
    if not instances:
        raise HTTPException(status_code=400, detail="No instances provided in request")
    inner_request = instances[0]  # Assume single instance for POC

    # Manual validation and parameter extraction
    prompt = inner_request.get("prompt")
    num = inner_request.get("num", 10)
    length = inner_request.get("length", 100)  # Target length as a guide
    temperature = inner_request.get("temperature", 1.0)
    top_k = inner_request.get("top_k", 50)
    top_p = inner_request.get("top_p", 0.9)
    min_new_tokens = inner_request.get("min_new_tokens", 10)
    max_new_tokens = inner_request.get("max_new_tokens", 10)
    early_stopping = inner_request.get("early_stopping", False)
    repetition_penalty = inner_request.get("repetition_penalty", 1.0)
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"]
    input_ids = input_ids[:, :-1].to("cuda")

    # Use min_new_tokens and max_new_tokens directly, guided by length
    effective_max_length = len(prompt) + max_new_tokens  # Max length as prompt + new tokens
    model_output = model.generate(
        input_ids=input_ids,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=True,  # Hardcoded to True
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_beams=1,  # Hardcoded to 1 (no beam search)
        early_stopping=early_stopping,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num,
        max_length=effective_max_length,  # Safety cap
    )

    generated_sequences = []
    for i in range(model_output.shape[0]):
        generated = tokenizer.decode(model_output[i]).replace(" ", "")[5 + len(prompt):]
        generated_sequences.append(generated)

    if not all(len(seq) >= 10 for seq in generated_sequences):
        raise HTTPException(status_code=500, detail="One or more generated sequence lengths are too short")

    return {"predictions": [{"sequences": generated_sequences}]}

@app.post("/embed")
async def embed_sequences(request: Request):
    body_bytes = await request.body()
    body = json.loads(body_bytes)
    instances = body.get("instances", [])
    if not instances:
        raise HTTPException(status_code=400, detail="No instances provided in request")
    inner_request = instances[0]  # Assume single instance for POC

    # Manual validation
    sequences = inner_request.get("seqs", [])
    if not sequences or not isinstance(sequences, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'seqs' field: must be a list of strings")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Batch encode sequences
    output = tokenizer.batch_encode_plus(
        sequences,
        max_length=10240,
        return_tensors="pt",
        padding="longest",
        truncation=True
    )
    input_ids = output["input_ids"].cuda()
    attention_mask = output["attention_mask"].cuda()

    # Forward pass
    model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()

    # Average pooling for embeddings
    attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
    embeddings = torch.sum(model_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

    # Return embeddings as list of lists
    embeddings_list = embeddings.tolist()

    return {"predictions": [{"embeddings": embeddings_list}]}
