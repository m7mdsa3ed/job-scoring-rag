import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def extract_sections(cv_text: str) -> dict:
    """
    Basic heuristic section extractor — replace with smarter parser later.
    """
    cv_text = cv_text.lower()
    sections = {"skills": "", "experience": "", "education": ""}
    if "skills" in cv_text:
        parts = re.split(r"(skills|experience|education)", cv_text)
        for i in range(1, len(parts)-1, 2):
            sections[parts[i].strip()] = parts[i+1].strip()
    else:
        sections["experience"] = cv_text
    return sections

def score_cv(job_emb, job_text, cv_data, model, weights, keyword_boost):
    total_score = 0
    details = {}

    for section, text in cv_data.items():
        if not text.strip():
            continue
        section_emb = model.encode(text, normalize_embeddings=True)
        sim = cosine_similarity([job_emb], [section_emb])[0][0]

        keywords = re.findall(r'\b\w+\b', job_text.lower())
        match_count = sum(kw in text.lower() for kw in keywords)
        boost = keyword_boost * min(match_count / 10, 1.0)

        weighted = (sim + boost) * weights.get(section, 0)
        total_score += weighted
        details[section] = round(weighted, 3)

    return round(total_score, 3), details
