from io import BytesIO
from xml.sax.saxutils import escape
from fastapi.responses import StreamingResponse

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import time
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from passlib.context import CryptContext
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
import fitz

from groq import Groq


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "openai/gpt-oss-20b"


# =========================================================
# APP
# =========================================================

app = FastAPI(title="StudyX API")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MONGODB
# =========================================================

client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["studyx"]

users_collection = db["users"]
materials_collection = db["materials"]
material_chunks_collection = db["material_chunks"]
study_plans_collection = db["study_plans"]
progress_collection = db["progress"]


# =========================================================
# PASSWORD HASHING
# =========================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# =========================================================
# LAZY EMBEDDING MODEL
# =========================================================

embedding_model = None

def get_embedding_model():
    global embedding_model

    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded successfully!")

    return embedding_model


# =========================================================
# MODELS
# =========================================================

class RegisterUser(BaseModel):
    name: str
    email: str
    password: str


class LoginUser(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str
    email: str = ""


class SearchRequest(BaseModel):
    question: str
    email: str
    top_k: int = 3


class QuizRequest(BaseModel):
    email: str
    material_id: str = ""
    number_of_questions: int = 5
    difficulty: str = "medium"


class QuestionPaperRequest(BaseModel):
    email: str
    material_id: str
    number_of_questions: int = 10
    difficulty: str = "medium"
    question_type: str = "mixed"
    total_marks: int = 30
    duration: int = 60


class StudyPlanRequest(BaseModel):
    email: str
    goal: str
    subject: str
    exam_date: str
    daily_hours: int = 2
    level: str = "intermediate"
    topics: str


class StudyTaskCompletionRequest(BaseModel):
    email: str
    day_index: int
    task_index: int
    completed: bool


class QuizProgressRequest(BaseModel):
    email: str
    filename: str = "Unknown"
    total_questions: int
    correct_answers: int
    score: float
    percentage: float
    difficulty: str = "medium"



# =========================================================
# AI VIVA MODE MODELS
# =========================================================

class VivaQuestionRequest(BaseModel):
    email: str
    material_id: str = ""
    difficulty: str = "medium"
    question_number: int = 1
    previous_questions: list[str] = []


class VivaEvaluateRequest(BaseModel):
    email: str
    material_id: str = ""
    question: str
    answer: str
    difficulty: str = "medium"
    question_number: int = 1
    previous_questions: list[str] = []


# =========================================================
# GROQ AI FUNCTION
# =========================================================

def ask_groq(
    prompt: str,
    json_mode: bool = False,
    max_completion_tokens: int | None = None
):
    """Generate an AI response using Groq.

    max_completion_tokens can be increased for structured outputs such as
    question papers. Existing callers keep their previous behavior.
    """
    try:
        request_data = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
        }

        if max_completion_tokens is not None:
            request_data["max_completion_tokens"] = max_completion_tokens

        if json_mode:
            request_data["response_format"] = {"type": "json_object"}

        response = groq_client.chat.completions.create(**request_data)
        content = response.choices[0].message.content

        if not content:
            raise Exception("Groq returned an empty response.")

        return content.strip()

    except Exception as error:
        error_text = str(error)
        print("Groq Error:", error_text)
        raise HTTPException(
            status_code=500,
            detail="Groq AI could not generate a response. Please try again."
        )


# Backward-compatible name so the existing endpoints keep working.
def ask_gemini(
    prompt: str,
    json_mode: bool = False,
    max_completion_tokens: int | None = None
):
    return ask_groq(
        prompt,
        json_mode=json_mode,
        max_completion_tokens=max_completion_tokens
    )


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
):

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        if end < len(text):

            last_space = text.rfind(
                " ",
                start,
                end
            )

            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# =========================================================
# VECTOR SEARCH FUNCTION
# =========================================================

def find_relevant_chunks(
    question: str,
    email: str,
    top_k: int = 3
):

    question_embedding = (
        get_embedding_model()
        .encode(question)
        .tolist()
    )

    chunks = material_chunks_collection.find({
        "email": email,
        "embedding": {
            "$exists": True
        }
    })

    scored_chunks = []

    for chunk in chunks:

        stored_embedding = chunk.get(
            "embedding"
        )

        if not stored_embedding:
            continue

        similarity = cosine_similarity(
            [question_embedding],
            [stored_embedding]
        )[0][0]

        scored_chunks.append({
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "similarity": float(similarity)
        })

    scored_chunks.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return scored_chunks[:top_k]


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "StudyX is running 🚀",
        "database": "MongoDB connected",
        "embedding": "Enabled",
        "ai_model": GROQ_MODEL
    }


# =========================================================
# REGISTER
# =========================================================

@app.post("/register")
def register(user: RegisterUser):

    email = user.email.strip().lower()

    existing_user = users_collection.find_one({
        "email": email
    })

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    hashed_password = pwd_context.hash(
        user.password
    )

    users_collection.insert_one({
        "name": user.name.strip(),
        "email": email,
        "password": hashed_password
    })

    return {
        "message": "Account created successfully!"
    }


# =========================================================
# LOGIN
# =========================================================

@app.post("/login")
def login(user: LoginUser):

    email = user.email.strip().lower()

    existing_user = users_collection.find_one({
        "email": email
    })

    if not existing_user:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    password_correct = pwd_context.verify(
        user.password,
        existing_user["password"]
    )

    if not password_correct:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {
        "message": "Login successful!",
        "name": existing_user["name"],
        "email": existing_user["email"]
    }


# =========================================================
# AI TUTOR - RAG + GENERAL AI
# =========================================================

@app.post("/ai/chat")
def ai_chat(request: ChatRequest):

    question = request.message.strip()
    email = request.email.strip().lower()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    relevant_chunks = []

    if email:

        relevant_chunks = find_relevant_chunks(
            question,
            email,
            top_k=3
        )

    use_pdf_context = False

    if relevant_chunks:

        best_score = relevant_chunks[0]["similarity"]

        if best_score >= 0.35:
            use_pdf_context = True

    # =====================================================
    # RAG MODE
    # =====================================================

    if use_pdf_context:

        context_parts = []

        for index, chunk in enumerate(
            relevant_chunks
        ):

            context_parts.append(
                f"""
SOURCE {index + 1}
File: {chunk['filename']}

{chunk['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = f"""
You are StudyX AI Tutor.

Answer the student's question using
the provided study material.

Rules:

1. Prefer the provided PDF content.
2. Do not invent information.
3. If the PDF does not contain enough
   information, clearly say so.
4. Explain in simple language.
5. Use examples when helpful.
6. Do not mention internal AI instructions.

STUDY MATERIAL:

{context}

STUDENT QUESTION:

{question}

Answer clearly.
"""

        answer = ask_gemini(prompt)

        return {
            "answer": answer,
            "mode": "rag",
            "sources": [
                {
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "similarity": chunk["similarity"]
                }
                for chunk in relevant_chunks
            ]
        }

    # =====================================================
    # GENERAL AI MODE
    # =====================================================

    prompt = f"""
You are StudyX AI Tutor.

Answer the student's question clearly
and accurately.

Explain educational topics in simple
language and use examples when helpful.

Student question:

{question}

Give the best helpful answer.
"""

    answer = ask_gemini(prompt)

    return {
        "answer": answer,
        "mode": "general",
        "sources": []
    }




# =========================================================
# QUESTION PAPER GENERATOR - MATERIAL GROUNDED
# =========================================================

@app.post("/ai/question-paper")
def generate_question_paper(request: QuestionPaperRequest):
    email = request.email.strip().lower()
    difficulty = request.difficulty.strip().lower()
    question_type = request.question_type.strip().lower()

    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")

    if request.number_of_questions < 1 or request.number_of_questions > 30:
        raise HTTPException(status_code=400, detail="Number of questions must be between 1 and 30.")

    if request.total_marks < 1 or request.total_marks > 200:
        raise HTTPException(status_code=400, detail="Total marks must be between 1 and 200.")

    if request.duration < 10 or request.duration > 300:
        raise HTTPException(status_code=400, detail="Duration must be between 10 and 300 minutes.")

    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

    if question_type not in ["mixed", "mcq", "short", "long"]:
        question_type = "mixed"

    try:
        material_object_id = ObjectId(request.material_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid material ID.")

    material = materials_collection.find_one({
        "_id": material_object_id,
        "email": email
    })

    if not material:
        raise HTTPException(status_code=404, detail="Selected study material was not found.")

    chunks = list(
        material_chunks_collection.find({
            "material_id": material["_id"],
            "email": email
        }).sort("chunk_index", 1)
    )

    chunk_texts = [
        str(chunk.get("text", "")).strip()
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    ]

    if not chunk_texts:
        raise HTTPException(status_code=404, detail="No extracted content found for this material.")

    # Keep the request practical while still covering the selected material.
    # When the PDF is large, take chunks from across the document instead of
    # only taking the beginning. This keeps the paper grounded in more topics.
    max_context_chars = 30000

    full_context = "\n\n".join(chunk_texts)

    if len(full_context) <= max_context_chars:
        context = full_context
    else:
        selected_parts = []
        used_chars = 0

        # Sample evenly across the material so later chapters/topics are
        # represented as well.
        target_chunks = min(
            len(chunk_texts),
            max(12, max_context_chars // 1000)
        )

        if target_chunks == 1:
            indexes = [0]
        else:
            indexes = [
                round(i * (len(chunk_texts) - 1) / (target_chunks - 1))
                for i in range(target_chunks)
            ]

        for index in indexes:
            part = chunk_texts[index]
            addition = (
                f"\n\n[SOURCE CHUNK {index + 1}]\n{part}"
            )

            if used_chars + len(addition) > max_context_chars:
                remaining = max_context_chars - used_chars
                if remaining > 300:
                    selected_parts.append(addition[:remaining])
                break

            selected_parts.append(addition)
            used_chars += len(addition)

        context = "".join(selected_parts)
        print(
            f"Question paper source capped at {max_context_chars} characters "
            f"using {len(selected_parts)} evenly distributed chunks."
        )

    if question_type == "mcq":
        distribution = "Create all questions as MCQs. Each MCQ must have exactly 4 options."
    elif question_type == "short":
        distribution = "Create all questions as short-answer questions."
    elif question_type == "long":
        distribution = "Create all questions as long/descriptive questions."
    else:
        distribution = """Create a balanced mixed paper:
- approximately 40% MCQ
- approximately 35% short-answer
- approximately 25% long-answer
Adjust the nearest whole numbers while keeping the exact requested question count."""

    prompt = f"""
You are StudyX Question Paper Generator.

Create exactly {request.number_of_questions} exam questions using ONLY the study material below.

Difficulty: {difficulty}
Format: {question_type}
Total marks: {request.total_marks}
Duration: {request.duration} minutes

{distribution}

STRICT RULES:
1. Use only the supplied study material. No outside knowledge.
2. Do not invent facts, topics, terminology, examples, or answers.
3. Questions must be clear, relevant, and non-repetitive.
4. Return ONLY valid JSON. No markdown.
5. The JSON must contain exactly one key: "sections".
6. The total number of question objects across all sections must be exactly {request.number_of_questions}.
7. MCQs must have exactly 4 different options.
8. Short/long questions must NOT contain an "options" key.
9. Do NOT include answers or explanations.
10. Do not include a "number" field; StudyX will number questions automatically.

Use this exact shape:
{{
  "sections": [
    {{
      "title": "SECTION A — MCQ",
      "questions": [
        {{
          "question": "Question text",
          "options": ["A", "B", "C", "D"],
          "type": "mcq"
        }}
      ]
    }}
  ]
}}

STUDY MATERIAL:
{context}
"""

    # Question papers can contain many structured objects. Give the JSON
    # response enough completion space so Groq does not stop halfway through.
    raw_response = ask_groq(
        prompt,
        json_mode=True,
        max_completion_tokens=8192
    )

    cleaned = raw_response.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first == -1 or last <= first:
            raise HTTPException(status_code=500, detail="AI returned an invalid question paper format.")
        try:
            parsed = json.loads(cleaned[first:last + 1])
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI returned an invalid question paper format.")

    sections = parsed.get("sections", []) if isinstance(parsed, dict) else []
    if not isinstance(sections, list):
        raise HTTPException(status_code=500, detail="AI returned an invalid section format.")

    clean_sections = []
    question_number = 1

    for section in sections:
        if not isinstance(section, dict):
            continue

        title = str(section.get("title", "SECTION")).strip()
        raw_questions = section.get("questions", [])

        if not isinstance(raw_questions, list):
            continue

        clean_questions = []

        for item in raw_questions:
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            qtype = str(item.get("type", "short")).strip().lower()

            if not question:
                continue

            cleaned_item = {
                "number": question_number,
                "question": question,
                "type": qtype
            }

            options = item.get("options")
            if qtype == "mcq":
                if not isinstance(options, list) or len(options) != 4:
                    continue
                options = [str(x).strip() for x in options]
                if len(set(options)) != 4 or any(not x for x in options):
                    continue
                cleaned_item["options"] = options

            clean_questions.append(cleaned_item)
            question_number += 1

            if question_number > request.number_of_questions:
                break

        if clean_questions:
            clean_sections.append({
                "title": title,
                "questions": clean_questions
            })

        if question_number > request.number_of_questions:
            break

    total_questions = sum(len(s["questions"]) for s in clean_sections)

    if total_questions != request.number_of_questions:
        raise HTTPException(
            status_code=500,
            detail=f"AI generated {total_questions} valid questions instead of {request.number_of_questions}. Please try again."
        )

    return {
        "message": "Question paper generated successfully!",
        "filename": material["filename"],
        "difficulty": difficulty,
        "question_type": question_type,
        "total_marks": request.total_marks,
        "duration": request.duration,
        "sections": clean_sections
    }


# =========================================================
# QUIZ GENERATOR
# =========================================================

@app.post("/ai/quiz")
def generate_quiz(request: QuizRequest):

    email = request.email.strip().lower()
    number_of_questions = request.number_of_questions
    difficulty = request.difficulty.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    if number_of_questions < 1 or number_of_questions > 20:

        raise HTTPException(
            status_code=400,
            detail="Number of questions must be between 1 and 20."
        )

    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

    # =====================================================
    # GET USER MATERIAL
    # =====================================================

    if request.material_id:

        try:
            material_object_id = ObjectId(
                request.material_id
            )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Invalid material ID."
            )

        material = materials_collection.find_one({
            "_id": material_object_id,
            "email": email
        })

    else:

        material = materials_collection.find_one({
            "email": email
        })

    if not material:

        raise HTTPException(
            status_code=404,
            detail="No study material found."
        )

    # =====================================================
    # GET MATERIAL CHUNKS
    # =====================================================

    chunks = material_chunks_collection.find({
        "material_id": material["_id"],
        "email": email
    }).sort(
        "chunk_index",
        1
    )

    chunk_texts = [
        chunk.get("text", "")
        for chunk in chunks
    ]

    chunk_texts = [
        text.strip()
        for text in chunk_texts
        if text.strip()
    ]

    if not chunk_texts:

        raise HTTPException(
            status_code=404,
            detail="No chunks found for this material."
        )

    context = "\n\n".join(
        chunk_texts[:3]
    )

    # =====================================================
    # QUIZ PROMPT
    # =====================================================

    prompt = f"""
You are StudyX Quiz Generator.

Create exactly {number_of_questions} multiple-choice questions using ONLY the study material below.

Difficulty: {difficulty}

STRICT RULES:

1. Return a JSON OBJECT, not markdown and not plain text.
2. The object must contain exactly one key: "questions".
3. "questions" must be an array containing exactly {number_of_questions} objects.
4. Every question object must contain exactly these keys:
   "question", "options", "correct_answer"
5. "options" must contain exactly 4 different strings.
6. "correct_answer" must be exactly one of the 4 option strings.
7. Base every question on the study material.
8. Do not add explanations, notes, prefixes, suffixes, or code fences.

OUTPUT STRUCTURE:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correct_answer": "Option A"
    }}
  ]
}}

STUDY MATERIAL:

{context}
"""

    raw_response = ask_gemini(
        prompt,
        json_mode=False
    )

    # =====================================================
    # JSON EXTRACTION
    # =====================================================

    cleaned_response = raw_response.strip()

    cleaned_response = (
        cleaned_response
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        parsed = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError:

        first = cleaned_response.find("{")
        last = cleaned_response.rfind("}")

        if first == -1 or last <= first:

            print(
                "Quiz JSON Error:",
                raw_response
            )

            raise HTTPException(
                status_code=500,
                detail="AI returned an invalid quiz format."
            )

        try:

            parsed = json.loads(
                cleaned_response[
                    first:last + 1
                ]
            )

        except json.JSONDecodeError:

            print(
                "Quiz JSON Error:",
                raw_response
            )

            raise HTTPException(
                status_code=500,
                detail="AI returned an invalid quiz format."
            )

    # =====================================================
    # ACCEPT OBJECT OR ARRAY
    # =====================================================

    if isinstance(parsed, dict):

        quiz = parsed.get(
            "questions",
            []
        )

    elif isinstance(parsed, list):

        quiz = parsed

    else:

        quiz = []

    # =====================================================
    # VALIDATE QUESTIONS
    # =====================================================

    valid_questions = []

    for item in quiz:

        if not isinstance(item, dict):
            continue

        question = item.get("question")
        options = item.get("options")
        correct_answer = item.get("correct_answer")

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            continue

        if (
            not isinstance(options, list)
            or len(options) != 4
        ):
            continue

        options = [
            str(option).strip()
            for option in options
        ]

        if any(
            not option
            for option in options
        ):
            continue

        if len(set(options)) != 4:
            continue

        if not isinstance(
            correct_answer,
            str
        ):
            continue

        if correct_answer not in options:
            continue

        valid_questions.append({
            "question": question.strip(),
            "options": options,
            "correct_answer": correct_answer.strip()
        })

    # =====================================================
    # FINAL CHECK
    # =====================================================

    if len(valid_questions) < number_of_questions:

        print(
            "Invalid/insufficient quiz response:",
            raw_response
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"AI generated only "
                f"{len(valid_questions)} valid questions "
                f"out of {number_of_questions}. "
                f"Please try again."
            )
        )

    valid_questions = valid_questions[
        :number_of_questions
    ]

    return {
        "message": "Quiz generated successfully!",
        "filename": material["filename"],
        "difficulty": difficulty,
        "questions": valid_questions
    }


# =========================================================
# PDF UPLOAD + AUTOMATIC EMBEDDINGS
# =========================================================

@app.post("/materials/upload")
async def upload_material(request: Request):

    try:

        form = await request.form()

        file = form.get("file")

        email = (
            request.query_params.get("email")
            or str(form.get("email") or "")
        )

        if not email:

            raise HTTPException(
                status_code=400,
                detail="User email is required."
            )

        if file is None:

            raise HTTPException(
                status_code=400,
                detail="PDF file is required."
            )

        if not file.filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported."
            )

        file_data = await file.read()

        pdf = fitz.open(
            stream=file_data,
            filetype="pdf"
        )

        extracted_text = ""

        for page in pdf:

            extracted_text += page.get_text()

        page_count = len(pdf)

        pdf.close()

        # =================================================
        # CREATE CHUNKS
        # =================================================

        chunks = create_chunks(
            extracted_text,
            chunk_size=1000,
            overlap=150
        )

        # =================================================
        # SAVE MATERIAL
        # =================================================

        material = {
            "filename": file.filename,
            "email": email.strip().lower(),
            "pages": page_count,
            "text": extracted_text,
            "text_length": len(extracted_text),
            "chunk_count": len(chunks)
        }

        result = materials_collection.insert_one(
            material
        )

        material_id = result.inserted_id

        # =================================================
        # CREATE EMBEDDINGS
        # =================================================

        print(
            f"Generating embeddings for "
            f"{len(chunks)} chunks..."
        )

        chunk_documents = []

        for index, chunk in enumerate(chunks):

            embedding = (
                get_embedding_model()
                .encode(chunk)
                .tolist()
            )

            chunk_documents.append({

                "material_id": material_id,

                "email":
                    email.strip().lower(),

                "filename":
                    file.filename,

                "chunk_index":
                    index,

                "text":
                    chunk,

                "text_length":
                    len(chunk),

                "embedding":
                    embedding,

                "embedding_model":
                    "all-MiniLM-L6-v2"
            })

        if chunk_documents:

            material_chunks_collection.insert_many(
                chunk_documents
            )

        print(
            "Embeddings generated successfully!"
        )

        return {

            "message":
                "PDF uploaded, chunked and embedded successfully!",

            "material_id":
                str(material_id),

            "filename":
                file.filename,

            "pages":
                page_count,

            "text_length":
                len(extracted_text),

            "chunk_count":
                len(chunks),

            "embeddings":
                "Generated automatically"
        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "Upload Error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process PDF."
        )


# =========================================================
# GET USER MATERIALS
# =========================================================

@app.get("/materials")
def get_materials(email: str):

    email = email.strip().lower()

    materials = materials_collection.find(
        {
            "email": email
        },
        {
            "text": 0
        }
    ).sort(
        "_id",
        -1
    )

    result = []

    for material in materials:

        result.append({

            "id":
                str(material["_id"]),

            "filename":
                material["filename"],

            "pages":
                material["pages"],

            "text_length":
                material["text_length"],

            "chunk_count":
                material.get(
                    "chunk_count",
                    0
                )
        })

    return {
        "materials": result
    }


# =========================================================
# GET MATERIAL CHUNKS
# =========================================================

@app.get("/materials/{material_id}/chunks")
def get_material_chunks(material_id: str):

    try:

        object_id = ObjectId(
            material_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid material ID."
        )

    chunks = (
        material_chunks_collection
        .find({
            "material_id":
                object_id
        })
        .sort(
            "chunk_index",
            1
        )
    )

    result = []

    for chunk in chunks:

        result.append({

            "id":
                str(chunk["_id"]),

            "chunk_index":
                chunk["chunk_index"],

            "text":
                chunk["text"]
        })

    return {

        "material_id":
            material_id,

        "chunks":
            result
    }


# =========================================================
# VECTOR SEARCH
# =========================================================

@app.post("/ai/search")
def vector_search(request: SearchRequest):

    question = request.question.strip()
    email = request.email.strip().lower()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    results = find_relevant_chunks(
        question,
        email,
        request.top_k
    )

    return {

        "question":
            question,

        "results":
            results
    }


# =========================================================
# DELETE MATERIAL
# =========================================================

@app.delete("/materials/{material_id}")
def delete_material(material_id: str):

    try:

        object_id = ObjectId(
            material_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid material ID."
        )

    result = (
        materials_collection
        .delete_one({
            "_id":
                object_id
        })
    )

    if result.deleted_count == 0:

        raise HTTPException(
            status_code=404,
            detail="Material not found."
        )

    material_chunks_collection.delete_many({

        "material_id":
            object_id

    })

    return {

        "message":
            "Material and its chunks deleted successfully!"
    }

# =========================================================
# AI REVISION NOTES PDF GENERATOR
# =========================================================

class NotesPDFRequest(BaseModel):
    email: str
    material_id: str


def clean_revision_json(raw_response: str):

    cleaned = raw_response.strip()

    cleaned = (
        cleaned
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        first = cleaned.find("{")
        last = cleaned.rfind("}")

        if first == -1 or last <= first:
            raise ValueError(
                "Invalid JSON returned by Groq."
            )

        return json.loads(
            cleaned[first:last + 1]
        )


def create_revision_pdf(notes_data, filename):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RevisionTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "RevisionSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "RevisionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=7
    )

    topic_style = ParagraphStyle(
        "RevisionTopic",
        parent=styles["Heading3"],
        fontSize=11,
        leading=15,
        spaceBefore=7,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "RevisionBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=5
    )

    story = []

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    title = notes_data.get(
        "title",
        "AI Revision Notes"
    )

    story.append(
        Paragraph(
            escape(str(title)),
            title_style
        )
    )

    story.append(
        Paragraph(
            "Generated by StudyX AI",
            subtitle_style
        )
    )

    # -----------------------------------------------------
    # IMPORTANT TOPICS
    # -----------------------------------------------------

    topics = notes_data.get(
        "important_topics",
        []
    )

    if topics:

        story.append(
            Paragraph(
                "Important Topics",
                heading_style
            )
        )

        for index, topic in enumerate(
            topics,
            start=1
        ):

            if not isinstance(topic, dict):
                continue

            topic_name = topic.get(
                "topic",
                f"Topic {index}"
            )

            explanation = topic.get(
                "explanation",
                ""
            )

            key_points = topic.get(
                "key_points",
                []
            )

            story.append(
                Paragraph(
                    f"{index}. "
                    f"{escape(str(topic_name))}",
                    topic_style
                )
            )

            if explanation:

                story.append(
                    Paragraph(
                        "<b>Explanation:</b> "
                        + escape(
                            str(explanation)
                        ),
                        body_style
                    )
                )

            if isinstance(
                key_points,
                list
            ):

                for point in key_points:

                    story.append(
                        Paragraph(
                            "• "
                            + escape(
                                str(point)
                            ),
                            body_style
                        )
                    )

    # -----------------------------------------------------
    # DEFINITIONS
    # -----------------------------------------------------

    definitions = notes_data.get(
        "definitions",
        []
    )

    if definitions:

        story.append(
            Paragraph(
                "Key Definitions",
                heading_style
            )
        )

        for item in definitions:

            if not isinstance(
                item,
                dict
            ):
                continue

            term = item.get(
                "term",
                ""
            )

            definition = item.get(
                "definition",
                ""
            )

            story.append(
                Paragraph(
                    f"<b>{escape(str(term))}:</b> "
                    f"{escape(str(definition))}",
                    body_style
                )
            )

    # -----------------------------------------------------
    # EXAMPLES
    # -----------------------------------------------------

    examples = notes_data.get(
        "examples",
        []
    )

    if examples:

        story.append(
            Paragraph(
                "Important Examples",
                heading_style
            )
        )

        for example in examples:

            if isinstance(
                example,
                dict
            ):

                topic = example.get(
                    "topic",
                    ""
                )

                example_text = example.get(
                    "example",
                    ""
                )

                story.append(
                    Paragraph(
                        f"<b>{escape(str(topic))}:</b> "
                        f"{escape(str(example_text))}",
                        body_style
                    )
                )

            else:

                story.append(
                    Paragraph(
                        "• "
                        + escape(
                            str(example)
                        ),
                        body_style
                    )
                )

    # -----------------------------------------------------
    # EXAM QUESTIONS
    # -----------------------------------------------------

    questions = notes_data.get(
        "important_exam_questions",
        []
    )

    if questions:

        story.append(
            Paragraph(
                "Important Exam Questions",
                heading_style
            )
        )

        for index, question in enumerate(
            questions,
            start=1
        ):

            story.append(
                Paragraph(
                    f"{index}. "
                    + escape(
                        str(question)
                    ),
                    body_style
                )
            )

    # -----------------------------------------------------
    # QUICK REVISION
    # -----------------------------------------------------

    quick_revision = notes_data.get(
        "quick_revision",
        []
    )

    if quick_revision:

        story.append(
            Paragraph(
                "Quick Revision",
                heading_style
            )
        )

        for item in quick_revision:

            story.append(
                Paragraph(
                    "• "
                    + escape(
                        str(item)
                    ),
                    body_style
                )
            )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    def add_footer(
        canvas,
        doc
    ):

        canvas.saveState()

        canvas.setFont(
            "Helvetica",
            8
        )

        canvas.drawCentredString(
            A4[0] / 2,
            10 * mm,
            f"StudyX AI Revision Notes | Page {doc.page}"
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_footer,
        onLaterPages=add_footer
    )

    buffer.seek(0)

    return buffer


# =========================================================
# GENERATE REVISION PDF
# =========================================================

@app.post("/ai/notes-pdf")
def generate_revision_notes_pdf(
    request: NotesPDFRequest
):

    email = request.email.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    # -----------------------------------------------------
    # FIND SELECTED PDF
    # -----------------------------------------------------

    try:

        material_object_id = ObjectId(
            request.material_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid material ID."
        )

    material = materials_collection.find_one({
        "_id": material_object_id,
        "email": email
    })

    if not material:

        raise HTTPException(
            status_code=404,
            detail="Study material not found."
        )

    # -----------------------------------------------------
    # GET EXISTING PDF CHUNKS
    # -----------------------------------------------------

    chunks_cursor = (
        material_chunks_collection
        .find({
            "material_id": material_object_id,
            "email": email
        })
        .sort(
            "chunk_index",
            1
        )
    )

    chunks = []

    for chunk in chunks_cursor:

        text = chunk.get(
            "text",
            ""
        ).strip()

        if text:

            chunks.append(text)

    if not chunks:

        raise HTTPException(
            status_code=404,
            detail="No text chunks found for this PDF."
        )

    print(
        "========================================"
    )

    print(
        "AI REVISION NOTES GENERATION STARTED"
    )

    print(
        "File:",
        material["filename"]
    )

    print(
        "Total chunks:",
        len(chunks)
    )

    print(
        "========================================"
    )

    # -----------------------------------------------------
    # SINGLE-CALL AI ANALYSIS
    # -----------------------------------------------------
    # Keep Groq usage to ONE request per PDF.
    # The existing MongoDB chunks are reused; the original PDF is not
    # uploaded again. A generous source limit keeps the prompt practical
    # while avoiding multiple Groq calls and quota exhaustion.

    source_text = "\n\n".join(chunks)
    max_source_chars = 80000

    if len(source_text) > max_source_chars:
        source_text = source_text[:max_source_chars]
        print(
            f"Source text capped at {max_source_chars} characters "
            "to keep the single Groq request manageable."
        )

    print("Analyzing complete PDF with ONE Groq request...")

    revision_prompt = f"""
You are StudyX AI Revision Notes Generator.

Create a complete, concise, exam-friendly revision guide from the
following student's study material.

IMPORTANT RULES:
1. Use ONLY information present in the supplied study material.
2. Do not invent facts, topics, definitions, examples, formulas, or questions.
3. Remove repetition, filler, and irrelevant content.
4. Keep explanations simple and student-friendly.
5. Preserve important technical terminology exactly where useful.
6. Identify the most important topics and concepts for revision.
7. Include definitions, examples, formulas only when present in the source.
8. Include likely exam questions based ONLY on the source material.
9. Include short quick-revision points.
10. Organize the result clearly for a printable PDF.
11. Keep the response reasonably concise so it can be converted into a PDF.
12. Return ONLY valid JSON. Do not use Markdown fences.

Return EXACTLY this JSON structure:

{{
  "title": "AI Revision Notes",
  "important_topics": [
    {{
      "topic": "Topic Name",
      "explanation": "Concise and simple explanation",
      "key_points": [
        "Key point 1",
        "Key point 2",
        "Key point 3"
      ]
    }}
  ],
  "definitions": [
    {{
      "term": "Term",
      "definition": "Concise definition"
    }}
  ],
  "examples": [
    {{
      "topic": "Topic",
      "example": "Concise example"
    }}
  ],
  "important_exam_questions": [
    "Question 1",
    "Question 2",
    "Question 3"
  ],
  "quick_revision": [
    "Quick revision point 1",
    "Quick revision point 2",
    "Quick revision point 3"
  ]
}}

STUDY MATERIAL:

{source_text}
"""

    try:

        final_response = ask_gemini(
            revision_prompt,
            json_mode=False
        )

        notes_data = clean_revision_json(
            final_response
        )

    except Exception as error:

        print(
            "Revision generation error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AI could not create the revision notes. "
                "Please check your Groq quota and try again."
            )
        )

    if not isinstance(
        notes_data,
        dict
    ):

        raise HTTPException(
            status_code=500,
            detail="Invalid revision notes format."
        )

    # -----------------------------------------------------
    # PDF FILE NAME
    # -----------------------------------------------------

    original_filename = material.get(
        "filename",
        "Study_Material.pdf"
    )

    base_name = os.path.splitext(
        original_filename
    )[0]

    safe_name = "".join(
        character
        for character in base_name
        if character.isalnum()
        or character in (
            " ",
            "_",
            "-"
        )
    ).strip()

    if not safe_name:

        safe_name = "Study_Material"

    output_filename = (
        f"{safe_name}_AI_Revision_Notes.pdf"
    )

    # -----------------------------------------------------
    # CREATE PDF
    # -----------------------------------------------------

    pdf_buffer = create_revision_pdf(
        notes_data,
        output_filename
    )

    print(
        "Revision PDF created:",
        output_filename
    )

    print(
        "========================================"
    )

    # -----------------------------------------------------
    # DOWNLOAD PDF
    # -----------------------------------------------------

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="{output_filename}"'
        }
    )

# =========================================================
# AI STUDY PLANNER
# =========================================================

@app.post("/ai/study-plan")
def generate_study_plan(
    request: StudyPlanRequest
):

    email = request.email.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    if not request.goal.strip():

        raise HTTPException(
            status_code=400,
            detail="Study goal is required."
        )

    if not request.subject.strip():

        raise HTTPException(
            status_code=400,
            detail="Subject is required."
        )

    if not request.exam_date.strip():

        raise HTTPException(
            status_code=400,
            detail="Exam date is required."
        )

    if not request.topics.strip():

        raise HTTPException(
            status_code=400,
            detail="Topics are required."
        )

    if request.daily_hours < 1 or request.daily_hours > 10:

        raise HTTPException(
            status_code=400,
            detail="Daily study hours must be between 1 and 10."
        )

    # =====================================================
    # AI PROMPT
    # =====================================================

    prompt = f"""
You are StudyX, an AI study planning assistant.

Create a realistic and practical study plan for a student.

Student Goal:
{request.goal}

Subject:
{request.subject}

Exam / Target Date:
{request.exam_date}

Daily Study Time:
{request.daily_hours} hours

Current Level:
{request.level}

Topics / Syllabus:
{request.topics}

IMPORTANT:

Return ONLY valid JSON.

Do not use markdown.
Do not add any explanation outside JSON.

Use exactly this structure:

{{
    "title": "Personalized Study Plan",
    "summary": "Short summary of the study plan",
    "days": [
        {{
            "date": "Day 1",
            "tasks": [
                {{
                    "topic": "Topic name",
                    "activity": "What the student should study or practice",
                    "hours": 1
                }}
            ]
        }}
    ]
}}

Rules:

1. Divide the syllabus across realistic study days.
2. Do not overload a single day.
3. Respect the daily study time.
4. Include learning, practice and revision.
5. Start with easier topics.
6. Gradually move toward harder topics.
7. Include final revision before the exam date.
8. Keep activities specific and useful.
9. "hours" must always be a number.
10. Return only valid JSON.
"""

    raw_response = ask_gemini(
        prompt,
        json_mode=True
    )

    cleaned_response = raw_response.strip()

    # =====================================================
    # REMOVE MARKDOWN
    # =====================================================

    if cleaned_response.startswith("```json"):

        cleaned_response = (
            cleaned_response
            .replace("```json", "", 1)
            .replace("```", "")
            .strip()
        )

    elif cleaned_response.startswith("```"):

        cleaned_response = (
            cleaned_response
            .replace("```", "")
            .strip()
        )

    # =====================================================
    # PARSE JSON
    # =====================================================

    try:

        plan = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError:

        print(
            "Study Plan JSON Error:",
            raw_response
        )

        raise HTTPException(
            status_code=500,
            detail="AI returned an invalid study plan format."
        )

    # =====================================================
    # VALIDATE PLAN
    # =====================================================

    if not isinstance(plan, dict):

        raise HTTPException(
            status_code=500,
            detail="Invalid study plan format."
        )

    if "days" not in plan:

        raise HTTPException(
            status_code=500,
            detail="Study plan does not contain days."
        )

    if not isinstance(
        plan["days"],
        list
    ):

        raise HTTPException(
            status_code=500,
            detail="Invalid study plan days format."
        )

    # =====================================================
    # SAVE STUDY PLAN TO MONGODB
    # =====================================================

    study_plan_document = {

        "email":
            email,

        "goal":
            request.goal.strip(),

        "subject":
            request.subject.strip(),

        "exam_date":
            request.exam_date.strip(),

        "daily_hours":
            request.daily_hours,

        "level":
            request.level.strip().lower(),

        "topics":
            request.topics.strip(),

        "plan":
            plan,

        "completed_tasks":
            [],

        "created_at":
            datetime.now(timezone.utc)
    }

    result = study_plans_collection.insert_one(
        study_plan_document
    )

    # =====================================================
    # RETURN PLAN
    # =====================================================

    return {

        "message":
            "Study plan generated and saved successfully!",

        "email":
            email,

        "plan_id":
            str(result.inserted_id),

        "plan":
            plan,

        "completed_tasks":
            []
    }


# =========================================================
# GET SAVED STUDY PLANS
# =========================================================

@app.get("/study-plans")
def get_study_plans(email: str):

    email = email.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    plans = (
        study_plans_collection
        .find({
            "email":
                email
        })
        .sort(
            "created_at",
            -1
        )
    )

    result = []

    for plan in plans:

        result.append({

            "id":
                str(plan["_id"]),

            "goal":
                plan.get(
                    "goal",
                    ""
                ),

            "subject":
                plan.get(
                    "subject",
                    ""
                ),

            "exam_date":
                plan.get(
                    "exam_date",
                    ""
                ),

            "daily_hours":
                plan.get(
                    "daily_hours",
                    0
                ),

            "level":
                plan.get(
                    "level",
                    "intermediate"
                ),

            "topics":
                plan.get(
                    "topics",
                    ""
                ),

            "plan":
                plan.get(
                    "plan",
                    {}
                ),

            "completed_tasks":
                plan.get(
                    "completed_tasks",
                    []
                ),

            "created_at":
                plan.get(
                    "created_at"
                )
        })

    return {

        "email":
            email,

        "plans":
            result
    }


# =========================================================
# STUDY PLAN TASK COMPLETION
# =========================================================

@app.post("/study-plans/{plan_id}/task")
def update_study_task(
    plan_id: str,
    request: StudyTaskCompletionRequest
):

    email = request.email.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    if request.day_index < 0:

        raise HTTPException(
            status_code=400,
            detail="Invalid day index."
        )

    if request.task_index < 0:

        raise HTTPException(
            status_code=400,
            detail="Invalid task index."
        )

    try:

        plan_object_id = ObjectId(
            plan_id
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid study plan ID."
        )

    study_plan = study_plans_collection.find_one({
        "_id":
            plan_object_id,

        "email":
            email
    })

    if not study_plan:

        raise HTTPException(
            status_code=404,
            detail="Study plan not found."
        )

    plan = study_plan.get(
        "plan",
        {}
    )

    days = plan.get(
        "days",
        []
    )

    if (
        request.day_index >= len(days)
        or not isinstance(
            days[request.day_index],
            dict
        )
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid study day."
        )

    tasks = days[
        request.day_index
    ].get(
        "tasks",
        []
    )

    if request.task_index >= len(tasks):

        raise HTTPException(
            status_code=400,
            detail="Invalid study task."
        )

    task_key = (
        f"{request.day_index}-"
        f"{request.task_index}"
    )

    completed_tasks = study_plan.get(
        "completed_tasks",
        []
    )

    if not isinstance(
        completed_tasks,
        list
    ):

        completed_tasks = []

    if request.completed:

        if task_key not in completed_tasks:

            completed_tasks.append(
                task_key
            )

    else:

        completed_tasks = [
            key
            for key in completed_tasks
            if key != task_key
        ]

    study_plans_collection.update_one(
        {
            "_id":
                plan_object_id,

            "email":
                email
        },
        {
            "$set": {
                "completed_tasks":
                    completed_tasks
            }
        }
    )

    return {

        "message":
            "Task completion updated successfully!",

        "plan_id":
            plan_id,

        "task":
            task_key,

        "completed":
            request.completed,

        "completed_tasks":
            completed_tasks
    }


# =========================================================
# SAVE QUIZ RESULT
# =========================================================

@app.post("/progress/quiz")
def save_quiz_progress(
    request: QuizProgressRequest
):

    email = request.email.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    if request.total_questions < 1:

        raise HTTPException(
            status_code=400,
            detail="Total questions must be at least 1."
        )

    if request.correct_answers < 0:

        raise HTTPException(
            status_code=400,
            detail="Invalid correct answers."
        )

    if request.correct_answers > request.total_questions:

        raise HTTPException(
            status_code=400,
            detail="Correct answers cannot exceed total questions."
        )

    progress = {

        "email":
            email,

        "type":
            "quiz",

        "filename":
            request.filename,

        "total_questions":
            request.total_questions,

        "correct_answers":
            request.correct_answers,

        "score":
            request.score,

        "percentage":
            request.percentage,

        "difficulty":
            request.difficulty,

        "created_at":
            datetime.now(timezone.utc)
    }

    result = progress_collection.insert_one(
        progress
    )

    return {

        "message":
            "Quiz progress saved successfully!",

        "progress_id":
            str(result.inserted_id)
    }


# =========================================================
# GET USER PROGRESS
# =========================================================

@app.get("/progress")
def get_progress(email: str):

    email = email.strip().lower()

    if not email:

        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )

    records = (
        progress_collection
        .find({
            "email":
                email
        })
        .sort(
            "created_at",
            -1
        )
    )

    result = []

    for record in records:

        result.append({

            "id":
                str(record["_id"]),

            "type":
                record.get(
                    "type",
                    "quiz"
                ),

            "filename":
                record.get(
                    "filename",
                    "Unknown"
                ),

            "total_questions":
                record.get(
                    "total_questions",
                    0
                ),

            "correct_answers":
                record.get(
                    "correct_answers",
                    0
                ),

            "score":
                record.get(
                    "score",
                    0
                ),

            "percentage":
                record.get(
                    "percentage",
                    0
                ),

            "difficulty":
                record.get(
                    "difficulty",
                    "medium"
                ),

            "created_at":
                record.get(
                    "created_at"
                )
        })

    return {

        "email":
            email,

        "progress":
            result
    }


# =========================================================
# AI VIVA MODE - VOICE-FIRST VIVA
# =========================================================

def _get_viva_material(email: str, material_id: str = ""):
    email = email.strip().lower()

    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")

    if material_id:
        try:
            object_id = ObjectId(material_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid material ID.")

        material = materials_collection.find_one({
            "_id": object_id,
            "email": email
        })
    else:
        material = materials_collection.find_one(
            {"email": email},
            sort=[("_id", -1)]
        )

    if not material:
        raise HTTPException(
            status_code=404,
            detail="Study material not found. Upload a PDF first."
        )

    chunks = list(
        material_chunks_collection.find({
            "material_id": material["_id"],
            "email": email
        }).sort("chunk_index", 1)
    )

    texts = [
        str(chunk.get("text", "")).strip()
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    ]

    if not texts:
        raise HTTPException(
            status_code=404,
            detail="No extracted content found for this material."
        )

    # Keep the prompt practical while sampling across the PDF.
    full_text = "\n\n".join(texts)
    max_chars = 24000

    if len(full_text) <= max_chars:
        context = full_text
    else:
        target = min(len(texts), max(12, max_chars // 1000))
        if target <= 1:
            indexes = [0]
        else:
            indexes = [
                round(i * (len(texts) - 1) / (target - 1))
                for i in range(target)
            ]

        parts = []
        used = 0
        for idx in indexes:
            addition = f"\n\n[SOURCE CHUNK {idx + 1}]\n{texts[idx]}"
            if used + len(addition) > max_chars:
                remaining = max_chars - used
                if remaining > 300:
                    parts.append(addition[:remaining])
                break
            parts.append(addition)
            used += len(addition)

        context = "".join(parts)

    return material, context


@app.post("/ai/viva/question")
def viva_question(request: VivaQuestionRequest):
    difficulty = request.difficulty.strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

    if request.question_number < 1 or request.question_number > 20:
        raise HTTPException(
            status_code=400,
            detail="Question number must be between 1 and 20."
        )

    material, context = _get_viva_material(
        request.email,
        request.material_id
    )

    previous = [
        str(q).strip()
        for q in request.previous_questions[-8:]
        if str(q).strip()
    ]

    previous_text = "\n".join(
        f"- {q}" for q in previous
    ) or "None"

    prompt = f"""
You are the voice examiner for StudyX AI Viva Mode.

Create ONE viva question from ONLY the supplied study material.

Material: {material["filename"]}
Difficulty: {difficulty}
Question number: {request.question_number}

Previous questions already asked:
{previous_text}

STRICT RULES:
1. Ask exactly ONE clear oral-exam question.
2. Use ONLY information present in the study material.
3. Do not invent facts or topics outside the material.
4. Avoid repeating previous questions.
5. Prefer conceptual questions suitable for a student viva.
6. The question should be natural when spoken aloud.
7. Return ONLY valid JSON with exactly:
{{"question":"...","topic":"..."}}
8. No markdown or extra text.

STUDY MATERIAL:
{context}
"""

    raw = ask_groq(
        prompt,
        json_mode=True,
        max_completion_tokens=800
    )

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            data = json.loads(cleaned[first:last + 1])
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="AI returned an invalid viva question."
            )

    question = str(data.get("question", "")).strip()
    topic = str(data.get("topic", "General")).strip()

    if not question:
        raise HTTPException(
            status_code=500,
            detail="AI could not create a viva question."
        )

    return {
        "message": "Viva question generated successfully!",
        "filename": material["filename"],
        "question_number": request.question_number,
        "difficulty": difficulty,
        "question": question,
        "topic": topic
    }


@app.post("/ai/viva/evaluate")
def viva_evaluate(request: VivaEvaluateRequest):
    question = request.question.strip()
    answer = request.answer.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if not answer:
        raise HTTPException(status_code=400, detail="Answer is required.")

    if len(answer) > 6000:
        answer = answer[:6000]

    difficulty = request.difficulty.strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

    material, context = _get_viva_material(
        request.email,
        request.material_id
    )

    prompt = f"""
You are a strict but helpful university viva examiner for StudyX.

Evaluate the student's oral answer using ONLY the supplied study material.

Study material: {material["filename"]}
Difficulty: {difficulty}

QUESTION:
{question}

STUDENT ANSWER:
{answer}

STUDY MATERIAL:
{context}

Evaluate factual correctness and concept understanding, not accent, grammar,
or speaking style.

Return ONLY valid JSON with exactly these keys:
{{
  "score": 0,
  "verdict": "Correct",
  "feedback": "Short spoken-friendly feedback.",
  "strength": "What the student got right.",
  "missing_points": ["Important point missed"],
  "follow_up_needed": true
}}

RULES:
1. score is an integer from 0 to 10.
2. Do not give credit for unsupported claims.
3. If the answer is partially correct, reflect that in the score.
4. Keep feedback concise because it will be spoken aloud.
5. missing_points must be [] when nothing important is missing.
6. follow_up_needed should be true when clarification or deeper testing is useful.
7. Do not judge pronunciation, accent, personality, or intelligence.
8. Do not invent information outside the material.
"""

    raw = ask_groq(
        prompt,
        json_mode=True,
        max_completion_tokens=1000
    )

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            data = json.loads(cleaned[first:last + 1])
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="AI returned an invalid viva evaluation."
            )

    try:
        score = max(0, min(10, int(data.get("score", 0))))
    except Exception:
        score = 0

    verdict = str(data.get("verdict", "Needs Improvement")).strip()
    feedback = str(data.get("feedback", "")).strip()
    strength = str(data.get("strength", "")).strip()
    missing_points = data.get("missing_points", [])

    if not isinstance(missing_points, list):
        missing_points = []

    missing_points = [
        str(item).strip()
        for item in missing_points
        if str(item).strip()
    ][:5]

    return {
        "message": "Viva answer evaluated successfully!",
        "score": score,
        "verdict": verdict,
        "feedback": feedback,
        "strength": strength,
        "missing_points": missing_points,
        "follow_up_needed": bool(data.get("follow_up_needed", True))
    }


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
