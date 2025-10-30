from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import uuid
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Load environment ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "cvs")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
KEYWORD_BOOST = float(os.getenv("KEYWORD_BOOST", 0.05))

# Section weights (importance for scoring)
SECTION_WEIGHTS = {
    "skills": 0.4,
    "experience": 0.4,
    "education": 0.2
}

# --- Initialize app and dependencies ---
app = FastAPI(title="RAG CV Scoring API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(url=QDRANT_URL)

# Ensure collection exists
if COLLECTION not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

# --- Helper Function ---
def score_cv(job_emb, job_text, cv_data, model):
    total_score = 0
    details = {}

    for section, text in cv_data.items():
        if not text.strip():
            continue

        section_emb = model.encode(text, normalize_embeddings=True)
        sim = cosine_similarity([job_emb], [section_emb])[0][0]

        # Keyword boost
        keywords = re.findall(r'\b\w+\b', job_text.lower())
        match_count = sum(kw in text.lower() for kw in keywords)
        boost = KEYWORD_BOOST * min(match_count / 10, 1.0)

        weighted = (sim + boost) * SECTION_WEIGHTS.get(section, 0)
        total_score += weighted
        details[section] = round(weighted, 3)

    return round(total_score, 3), details


# ============================
# 🔹 ENDPOINT 1: Embed CVs
# ============================
@app.post("/embed")
def embed_cvs(data: dict = Body(...)):
    """
    Store CVs with structured fields.
    Example:
    {
      "cvs": [
        {
          "id": "1",
          "skills": "React, Node.js, MongoDB",
          "experience": "Built scalable MERN applications",
          "education": "B.Sc. Computer Science"
        }
      ]
    }
    """
    cvs = data.get("cvs", [])
    points = []

    for cv in cvs:
        cv_id = str(cv.get("id", uuid.uuid4()))
        sections = {
            "skills": cv.get("skills", ""),
            "experience": cv.get("experience", ""),
            "education": cv.get("education", "")
        }

        # Combine all text for vector representation
        combined_text = " ".join(sections.values())
        cv_emb = model.encode(combined_text, normalize_embeddings=True).tolist()

        points.append(models.PointStruct(
            id=cv_id,
            vector=cv_emb,
            payload={
                "id": cv_id,
                "sections": sections
            }
        ))

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    return {"status": "success", "count": len(points)}


# ============================
# 🔹 ENDPOINT 2: Score CVs
# ============================
@app.post("/score")
def score_job_description(data: dict = Body(...)):
    """
    Input:
    {
      "job_description": "Looking for a full stack developer skilled in React and Node."
    }
    """
    job_desc = data.get("job_description", "").strip()
    if not job_desc:
        return {"error": "job_description is required"}

    job_emb = model.encode(job_desc, normalize_embeddings=True).tolist()

    # Retrieve top candidates from Qdrant
    search_results = client.search(
        collection_name=COLLECTION,
        query_vector=job_emb,
        limit=10
    )

    ranked = []
    for res in search_results:
        cv_payload = res.payload
        score, details = score_cv(
            job_emb=model.encode(job_desc, normalize_embeddings=True),
            job_text=job_desc,
            cv_data=cv_payload.get("sections", {}),
            model=model
        )

        ranked.append({
            "id": cv_payload["id"],
            "score": score,
            "details": details,
            "similarity": round(res.score, 3)
        })

    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
    return {"job_description": job_desc[:80] + "...", "results": ranked}
