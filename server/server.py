import os
import torch
import tempfile, shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    pipeline, AutoTokenizer, AutoModelForCausalLM
)

# ---------- ENV & CONFIG ---------- #
os.environ['HF_HOME'] = '/home/guest/lv02/hf_cache'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------- FASTAPI SETUP ---------- #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------- WHISPER SETUP ---------- #
whisper_model_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

def transcribe_file(path: str) -> str:
    try:
        result = whisper_pipe(path, return_timestamps=True)
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/transcribe-batch")
async def transcribe_batch(files: List[UploadFile] = File(...)):
    temp_paths = []
    try:
        for file in files:
            suffix = os.path.splitext(file.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_paths.append(tmp.name)

        transcriptions = [""] * len(temp_paths)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(transcribe_file, path): i
                for i, path in enumerate(temp_paths)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    transcriptions[idx] = future.result()
                except Exception as e:
                    transcriptions[idx] = f"Error: {str(e)}"

        return JSONResponse(content={"transcriptions": transcriptions})
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except Exception:
                pass

# ---------- QWEN SANITIZER SETUP ---------- #
qwen_model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True
)

class TranscriptInput(BaseModel):
    transcript: str

@app.post("/sanitize")
async def sanitize(input: TranscriptInput):
    prompt = "You are a data privacy assistant.\n\nYour task is to sanitize transcripts. Apply the following redactions:\n\n1. For credit card numbers: Show the first 6 and last 4 digits. Replace the middle digits with X. Example: 123456XXXXXX1234.\n2. For Social Security Numbers (SSNs): Replace all but the last 4 digits with X. Example: XXX-XX-1234.\n\nKeep the rest of the transcript unchanged. Return only the sanitized text, no extra explanations."
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input.transcript}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(qwen_model.device)
    generated_ids = qwen_model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {"sanitized": response}

# ---------- HEALTH CHECK ---------- #
@app.get("/ping")
async def ping():
    return {"message": "Combined server is up"}
