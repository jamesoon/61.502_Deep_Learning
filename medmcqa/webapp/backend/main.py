"""
FastAPI backend for the MedMCQA exam platform.

Endpoints:
  GET    /api/questions          — list all questions
  POST   /api/questions          — add a question (admin)
  PUT    /api/questions/{id}     — update a question (admin)
  DELETE /api/questions/{id}     — delete a question (admin)
  POST   /api/export             — download questions as CSV
  POST   /api/grade              — AI grade + explanation for a submission
  POST   /api/auth               — validate admin password → returns token

Run locally:
  cd webapp/backend
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

Configure model (env vars):
  MODEL_ID       — HuggingFace model ID (default Qwen/Qwen3-30B-A3B)
  ADAPTER_PATH   — path to fine-tuned adapter (optional)
  USE_4BIT       — "1" to load in 4-bit NF4 (default 1)
  ADMIN_PASSWORD — admin password (default: admin123 — change in production)
  MOCK_MODEL     — "1" to skip model loading (for UI development)
"""

import os
import csv
import io
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="MedMCQA Exam Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────────

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_TOKEN = "admin-token-secret"   # stateless for simplicity; replace with JWT in prod
MOCK_MODEL = os.getenv("MOCK_MODEL", "0") == "1"

# ── In-memory question store (backed by CSV on disk) ─────────────────────────

QUESTIONS_CSV = os.getenv("QUESTIONS_CSV", "questions.csv")
_questions: dict[str, dict] = {}   # id → question dict


def load_questions_from_csv():
    global _questions
    _questions = {}
    if not os.path.exists(QUESTIONS_CSV):
        return
    with open(QUESTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            _questions[row["id"]] = row


def save_questions_to_csv():
    fieldnames = ["id", "question", "option_a", "option_b", "option_c", "option_d",
                  "correct_answer", "subject", "topic", "explanation"]
    with open(QUESTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_questions.values())


load_questions_from_csv()

# ── Model loading (lazy, on first /grade call) ───────────────────────────────

_model_engine = None


def get_model():
    global _model_engine
    if _model_engine is not None:
        return _model_engine
    if MOCK_MODEL:
        return None

    # Import here so the server starts even without GPU/PyTorch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from training.inference import MedQAInferenceEngine

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-30B-A3B")
    adapter_path = os.getenv("ADAPTER_PATH", None)
    use_4bit = os.getenv("USE_4BIT", "1") == "1"

    print(f"[model] Loading {model_id} (adapter={adapter_path}, 4bit={use_4bit})...")
    _model_engine = MedQAInferenceEngine(model_id, adapter_path, use_4bit)
    return _model_engine


# ── Auth ─────────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    password: str


@app.post("/api/auth")
def authenticate(req: AuthRequest):
    if req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"token": ADMIN_TOKEN, "role": "admin"}


def require_admin(x_admin_token: Optional[str] = Header(None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin access required")


# ── Question CRUD ─────────────────────────────────────────────────────────────

class QuestionCreate(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str   # "A" | "B" | "C" | "D"
    subject: str = ""
    topic: str = ""
    explanation: str = ""


@app.get("/api/questions")
def list_questions():
    return list(_questions.values())


@app.post("/api/questions", status_code=201)
def create_question(q: QuestionCreate, _: None = Depends(require_admin)):
    qid = str(uuid.uuid4())
    record = {"id": qid, **q.model_dump()}
    _questions[qid] = record
    save_questions_to_csv()
    return record


@app.put("/api/questions/{qid}")
def update_question(qid: str, q: QuestionCreate, _: None = Depends(require_admin)):
    if qid not in _questions:
        raise HTTPException(status_code=404, detail="Question not found")
    _questions[qid] = {"id": qid, **q.model_dump()}
    save_questions_to_csv()
    return _questions[qid]


@app.delete("/api/questions/{qid}")
def delete_question(qid: str, _: None = Depends(require_admin)):
    if qid not in _questions:
        raise HTTPException(status_code=404, detail="Question not found")
    del _questions[qid]
    save_questions_to_csv()
    return {"deleted": qid}


@app.post("/api/export")
def export_csv(_: None = Depends(require_admin)):
    """Stream the questions CSV file as a download."""
    fieldnames = ["id", "question", "option_a", "option_b", "option_c", "option_d",
                  "correct_answer", "subject", "topic", "explanation"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_questions.values())
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=questions.csv"},
    )


# ── Grading ───────────────────────────────────────────────────────────────────

class GradeRequest(BaseModel):
    question_id: str
    student_answer: str    # "A" | "B" | "C" | "D"


class GradeResponse(BaseModel):
    question_id: str
    student_answer: str
    correct_answer: str
    is_correct: bool
    model_explanation: str
    model_predicted_answer: Optional[str] = None


@app.post("/api/grade", response_model=GradeResponse)
def grade_question(req: GradeRequest):
    q = _questions.get(req.question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    correct_answer = q["correct_answer"].upper()
    is_correct = req.student_answer.upper() == correct_answer

    if MOCK_MODEL:
        # Development mode: return canned response
        explanation = (
            q.get("explanation") or
            f"The correct answer is {correct_answer}. "
            f"[Model not loaded — set MOCK_MODEL=0 and configure MODEL_ID to enable AI explanations.]"
        )
        return GradeResponse(
            question_id=req.question_id,
            student_answer=req.student_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
            model_explanation=explanation,
            model_predicted_answer=correct_answer,
        )

    engine = get_model()
    result = engine.answer(
        question=q["question"],
        options={
            "A": q["option_a"],
            "B": q["option_b"],
            "C": q["option_c"],
            "D": q["option_d"],
        },
    )

    return GradeResponse(
        question_id=req.question_id,
        student_answer=req.student_answer,
        correct_answer=correct_answer,
        is_correct=is_correct,
        model_explanation=result["explanation"],
        model_predicted_answer=result["answer"],
    )


# ── Batch grading (full exam submission) ─────────────────────────────────────

class ExamSubmission(BaseModel):
    student_name: str
    answers: dict[str, str]   # {question_id: answer_letter}


@app.post("/api/grade/exam")
def grade_exam(submission: ExamSubmission):
    """Grade a full exam submission and return per-question results + score."""
    results = []
    for qid, student_ans in submission.answers.items():
        try:
            result = grade_question(GradeRequest(question_id=qid, student_answer=student_ans))
            results.append(result.model_dump())
        except HTTPException:
            results.append({
                "question_id": qid,
                "student_answer": student_ans,
                "error": "Question not found",
            })

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    score_pct = round(100 * correct / total, 1) if total else 0

    return {
        "student_name": submission.student_name,
        "total_questions": total,
        "correct": correct,
        "score_percent": score_pct,
        "results": results,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "mock_model": MOCK_MODEL}
