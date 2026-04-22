from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid

from pipeline import answer_query, analyze_document

app = FastAPI()

# ── CORS (so React doesn't throw a tantrum) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ──
class QueryRequest(BaseModel):
    question: str


# ─────────────────────────────────────────────
# Query endpoint
# ─────────────────────────────────────────────
@app.post("/query")
def query_endpoint(req: QueryRequest):
    result = answer_query(req.question)

    return {
        "answer": result.response,
        "warnings": result.warnings,
        "confidence": result.verification_data.get("confidence", {}),
    }


# ─────────────────────────────────────────────
# File upload endpoint
# ─────────────────────────────────────────────
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run your pipeline
        report = analyze_document(file_path)

        # Optional: delete file after processing
        os.remove(file_path)

        return report

    except Exception as e:
        return {
            "error": str(e)
        }