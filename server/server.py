import os
import torch, logfire, config
from config import Settings
import tempfile, shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    pipeline, AutoTokenizer, AutoModelForCausalLM
)


# ---------- ENV & CONFIG ---------- #
os.environ['HF_HOME'] = '/home/guest/lv02/hf_cache' #configured cache for A100 GPU to store the models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32 #might need to change depending if GH200 works on f16

settings = Settings()

# -------------LOGFIRE SETUP------------ #
logfire.configure(token=settings.LOGFIRE_KEY)

# ---------- FASTAPI SETUP ---------- #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

#Validating whether file type and size match the requirements
ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 50 MB

def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large: {size} bytes")
    return ext

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

#Validate if GPU is being used
print("CUDA available:", torch.cuda.is_available())
print("Whisper model on device:", next(whisper_model.parameters()).device)


#Transcription pipeline which allows for 5 workers.
def transcribe_file(path: str,
                    chunk_length_s: float = 30.0,
                    stride_length_s: float = 5.0,
                    language: str = None,
                    task: str = "transcribe") -> dict:
    """
    Returns a dict with keys:
      - text (str)
    """
    pipe_kwargs = {
        "chunk_length_s": chunk_length_s,
        "stride_length_s": stride_length_s,
        "task": task
    }
    if language:
        pipe_kwargs["language"] = language

    result = whisper_pipe(path, **pipe_kwargs)
    return result




#Transcribe batch end-point, need to add more parameters and validation
@app.post("/transcribe-batch")
async def transcribe_batch(files: List[UploadFile] = File(...)):
    temp_paths = []
    responses = []
    try:
        # 1. Validate & save
        for file in files:
            ext = validate_file(file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_paths.append(tmp.name)

        # 2. Transcribe in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(
                    transcribe_file,
                    path,
                    chunk_length_s=30.0,
                    stride_length_s=5.0,
                    language="en",      # e.g. enforce English
                    task="transcribe"
                ): i
                for i, path in enumerate(temp_paths)
            }

            # prepare placeholder
            responses = [None] * len(temp_paths)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    responses[idx] = {"success": True, **res}
                except HTTPException as he:
                    responses[idx] = {"success": False, "error": he.detail}
                except Exception as e:
                    responses[idx] = {"success": False, "error": str(e)}

        # 3. Return structured JSON
        return JSONResponse(content={"results": responses})
    finally:
        # cleanup
        for path in temp_paths:
            try: os.remove(path)
            except: pass

# ---------- QWEN SANITIZER SETUP ---------- #
qwen_model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True
)

# ---------- Pydantic Schemas ----------
class TranscriptInput(BaseModel):
    transcript: str = Field(..., min_length=1)
    redact_cc: bool = Field(True, description="Redact credit card numbers")
    redact_ssn: bool = Field(True, description="Redact SSNs")
    max_new_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")

# ---------- Sanitization Logic ----------
def sanitize_text(
    text: str,
    redact_cc: bool = True,
    redact_ssn: bool = True,
    max_new_tokens: int = 512
) -> str:
    """
    Uses qwen_model to sanitize `text` according to the flags.
    Returns the sanitized string.
    """
    # 1. Build dynamic prompt
    prompt = '''You are a Data Privacy Assistant specialized in sanitizing text transcripts. Follow these rules precisely:

1. **Primary Redactions**  
   a. **Credit Card Numbers**  
      - Detect any sequence of digits that likely represents a credit/debit card number (13–19 digits, optionally separated by spaces or hyphens).  
      - Preserve the first six and last four digits. Replace all intervening digits with uppercase “X”.  
      - Maintain any original separators in their positions.  
      - Example:  
        - “My card 1234-5678-9012-3456” → “123456XXXXXX3456”  
        - “Card: 4111 1111 1111 1111” → “411111XXXXXX1111”  

   b. **Social Security Numbers (US SSNs)**  
      - Detect 9‑digit SSNs, with or without hyphens.  
      - Replace the first five digits with uppercase “X”, preserving hyphens.  
      - Example:  
        - “123-45-6789” → “XXX-XX-6789”  

2. **Formatting and Output**  
   - Return **only** the sanitized transcript—no additional commentary, notes, or JSON wrappers.  
   - Preserve all original punctuation, capitalization, and spacing, except where digits are replaced.  
   - Do **not** modify names, dates, addresses, phone numbers, or other PII unless explicitly listed above.  

3. **Error Handling and Guardrails**  
   - If you are uncertain whether a number is a credit card or SSN, leave it unchanged.  
   - Do **not** attempt to redact other types of sensitive data (e.g., phone numbers, email addresses) unless instructed.  
   - Do **not** hallucinate or invent any redactions. Only replace sequences that clearly match the patterns above.  
   - If the transcript contains no redaction targets, return it verbatim.  

4. **Examples**  
   - Input: “Contact me at 4111 1111 1111 1111 or SSN 123-45-6789.”  
     Output: “Contact me at 411111XXXXXX1111 or SSN XXX-XX-6789.”  

Begin sanitizing now.
'''

    # 2. Tokenize with chat template
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([formatted], return_tensors="pt").to(qwen_model.device)

    # 3. Generate and strip prompt tokens
    gen_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = gen_ids[0][ inputs["input_ids"].shape[1] : ]
    sanitized = tokenizer.decode(trimmed, skip_special_tokens=True)

    return sanitized

# ---------- SANITIZE ENDPOINT ----------
@app.post("/sanitize")
async def sanitize_endpoint(payload: TranscriptInput):
    """
    Sanitizes a single transcript according to user flags.
    """
    try:
        output = sanitize_text(
            text=payload.transcript,
            redact_cc=payload.redact_cc,
            redact_ssn=payload.redact_ssn,
            max_new_tokens=payload.max_new_tokens
        )
        return {"success": True, "sanitized": output}

    except ValueError as ve:
        # Parameter validation errors
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # Unexpected failures
        logfire.error("Sanitization failed", error=str(e))
        raise HTTPException(status_code=500, detail="Sanitization failed")

# ---------- HEALTH CHECK ---------- #
@app.get("/ping")
async def ping():
    return {"message": "Combined server is up"}

#might need to add more diagnostic endpoints


