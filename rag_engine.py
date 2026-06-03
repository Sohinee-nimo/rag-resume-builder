# rag_engine.py
# All the backend logic — chunking, embedding, retrieval, generation

import os
import json
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Groq client (LLM) ────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── Local Sentence Transformers (Embeddings) ─────────────────
@st.cache_resource
def load_embedding_model():
    """
    Loads and caches the embedding model locally inside the cloud container.
    This bypasses external HTTP requests entirely, resolving DNS errors.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    """
    Generates text embeddings locally on the server.
    """
    model = load_embedding_model()
    # model.encode automatically handles pooling and returns a clean vector
    vector = model.encode(text)
    return vector.tolist()

# ── Pure Python vector store ─────────────────────────────────
# ... [Keep your remaining VectorStore, chunking, and Groq generation functions as they are] ...



# ── Pure Python vector store ─────────────────────────────────
class VectorStore:
    """
    Stores embeddings in plain Python lists.
    Searches using cosine similarity with numpy.
    No external dependencies — works everywhere.
    """
    def __init__(self):
        self.embeddings = []
        self.documents  = []
        self.metadatas  = []

    def clear(self):
        self.embeddings = []
        self.documents  = []
        self.metadatas  = []

    def add(self, embedding, document, metadata):
        self.embeddings.append(np.array(embedding))
        self.documents.append(document)
        self.metadatas.append(metadata)

    def query(self, query_embedding, top_k=5):
        if not self.embeddings:
            return []
        q = np.array(query_embedding)
        scores = [
            np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-10)
            for emb in self.embeddings
        ]
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        return [
            {"text": self.documents[i], "section": self.metadatas[i]["section"]}
            for i in top_indices
        ]

    def count(self):
        return len(self.documents)


# single global store — persists for the session
vector_store = VectorStore()


# ── Chunking ─────────────────────────────────────────────────
def build_chunks(profile):
    chunks = []

    chunks.append({
        "text": f"Professional summary: {profile['summary'].strip()}",
        "section": "summary"
    })

    for job in profile["experience"]:
        chunks.append({
            "text": f"Role: {job['role']} at {job['company']} ({job['duration']})\n{job['details'].strip()}",
            "section": "experience"
        })

    skills_text = "Technical skills:\n"
    for category, items in profile["skills"].items():
        skills_text += f"  {category}: {', '.join(items)}\n"
    chunks.append({
        "text": skills_text.strip(),
        "section": "skills"
    })

    for proj in profile["projects"]:
        chunks.append({
            "text": f"Project: {proj['name']}\n{proj['details']}",
            "section": "projects"
        })

    chunks.append({
        "text": f"Education: {profile['education']}\nCertifications: {', '.join(profile['certifications'])}",
        "section": "education"
    })

    return chunks


# ── Ingest ───────────────────────────────────────────────────
def ingest_profile(profile):
    vector_store.clear()
    for chunk in build_chunks(profile):
        embedding = get_embedding(chunk["text"])
        vector_store.add(
            embedding=embedding,
            document=chunk["text"],
            metadata={"section": chunk["section"]}
        )
    return vector_store.count()


# ── Retrieve ─────────────────────────────────────────────────
def retrieve_for_jd(job_description, top_k=5):
    jd_embedding = get_embedding(job_description)
    return vector_store.query(jd_embedding, top_k=top_k)


def build_context(chunks):
    return "\n\n---\n\n".join([
        f"[{c['section'].upper()}]\n{c['text']}" for c in chunks
    ])


# ── Generate CV ──────────────────────────────────────────────
def generate_cv(job_description, candidate_name):
    context = build_context(retrieve_for_jd(job_description))
    prompt = f"""You are an expert CV writer. Write a tailored CV for {candidate_name}.
RULES: Only use facts from CANDIDATE BACKGROUND. Do not invent anything.
Use strong action verbs. Output valid JSON only — no markdown, no explanation.

CANDIDATE BACKGROUND:
{context}

JOB DESCRIPTION:
{job_description}

Return exactly this JSON structure:
{{
  "name": "...",
  "tagline": "...",
  "summary": "...",
  "experience": [
    {{
      "role": "...",
      "company": "...",
      "duration": "...",
      "bullets": ["...", "..."]
    }}
  ],
  "skills": {{
    "technical": ["...", "..."],
    "soft": ["...", "..."]
  }},
  "projects": [
    {{
      "name": "...",
      "description": "..."
    }}
  ],
  "education": "...",
  "certifications": ["..."]
}}"""

    raw = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.3
    ).choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


# ── Skill gap analysis ───────────────────────────────────────
def analyse_gap(job_description, profile):
    your_skills = set()
    for items in profile["skills"].values():
        your_skills.update([s.lower() for s in items])

    # Step 1: extract skills from JD
    jd_prompt = f"""Extract all skills from this job description.
Return ONLY this JSON, no markdown, no explanation:
{{"required": ["skill1", "skill2"], "preferred": ["skill3"], "soft_skills": ["skill4"]}}

JOB DESCRIPTION:
{job_description}"""

    raw = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": jd_prompt}],
        max_tokens=400,
        temperature=0.1
    ).choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    jd_skills = json.loads(raw)

    # Step 2: find gap
    matched          = [s for s in jd_skills["required"] if s.lower() in your_skills]
    missing_required = [s for s in jd_skills["required"] if s.lower() not in your_skills]
    missing_preferred= [s for s in jd_skills["preferred"] if s.lower() not in your_skills]

    # Step 3: generate recommendations for gaps
    recs = []
    if missing_required or missing_preferred:
        rec_prompt = f"""For each missing skill below, give a practical learning recommendation.
Return ONLY a JSON array, no markdown, no explanation:
[
  {{
    "skill": "...",
    "priority": "high or medium",
    "what_to_learn": "specific subtopics to focus on",
    "best_resource": "one specific free resource",
    "time_estimate": "e.g. 2 weeks"
  }}
]

Missing required skills : {missing_required}
Missing preferred skills : {missing_preferred}
Job context: {job_description[:300]}"""

        raw2 = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": rec_prompt}],
            max_tokens=800,
            temperature=0.3
        ).choices[0].message.content.strip()

        if raw2.startswith("```"):
            raw2 = raw2.split("```")[1]
            if raw2.startswith("json"):
                raw2 = raw2[4:]
        recs = json.loads(raw2)

    return {
        "matched":           matched,
        "missing_required":  missing_required,
        "missing_preferred": missing_preferred,
        "recommendations":   recs
    }