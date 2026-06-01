# rag_engine.py
# All the backend logic — chunking, embedding, retrieval, generation

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import json
import os

embed_model = SentenceTransformer(...)
groq_client = Groq(...)
# groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chroma_client = chromadb.Client()
collection = None


def build_chunks(profile):
    chunks = []
    chunks.append({"id": "summary", "text": f"Professional summary: {profile['summary'].strip()}", "section": "summary"})
    for i, job in enumerate(profile["experience"]):
        chunks.append({"id": f"experience_{i}", "text": f"Role: {job['role']} at {job['company']} ({job['duration']})\n{job['details'].strip()}", "section": "experience"})
    skills_text = "Technical skills:\n"
    for category, items in profile["skills"].items():
        skills_text += f"  {category}: {', '.join(items)}\n"
    chunks.append({"id": "skills", "text": skills_text.strip(), "section": "skills"})
    for i, proj in enumerate(profile["projects"]):
        chunks.append({"id": f"project_{i}", "text": f"Project: {proj['name']}\n{proj['details']}", "section": "projects"})
    chunks.append({"id": "education", "text": f"Education: {profile['education']}\nCertifications: {', '.join(profile['certifications'])}", "section": "education"})
    return chunks


def ingest_profile(profile):
    global collection
    try:
        chroma_client.delete_collection("resume_profile")
    except:
        pass
    collection = chroma_client.create_collection("resume_profile", metadata={"hnsw:space": "cosine"})
    for chunk in build_chunks(profile):
        embedding = embed_model.encode(chunk["text"]).tolist()
        collection.add(ids=[chunk["id"]], embeddings=[embedding], documents=[chunk["text"]], metadatas=[{"section": chunk["section"]}])
    return collection.count()


def retrieve_for_jd(job_description, top_k=5):
    jd_embedding = embed_model.encode(job_description).tolist()
    results = collection.query(query_embeddings=[jd_embedding], n_results=top_k)
    return [{"section": m["section"], "text": d} for d, m in zip(results["documents"][0], results["metadatas"][0])]


def build_context(chunks):
    return "\n\n---\n\n".join([f"[{c['section'].upper()}]\n{c['text']}" for c in chunks])


def generate_cv(job_description, candidate_name):
    context = build_context(retrieve_for_jd(job_description))
    prompt = f"""You are an expert CV writer. Write a tailored CV for {candidate_name}.
RULES: Only use facts from CANDIDATE BACKGROUND. Do not invent anything.
Use strong action verbs. Output valid JSON only — no markdown, no explanation.

CANDIDATE BACKGROUND:
{context}

JOB DESCRIPTION:
{job_description}

Return exactly this JSON:
{{
  "name": "...", "tagline": "...", "summary": "...",
  "experience": [{{"role":"...","company":"...","duration":"...","bullets":["..."]}}],
  "skills": {{"technical":["..."],"soft":["..."]}},
  "projects": [{{"name":"...","description":"..."}}],
  "education": "...", "certifications": ["..."]
}}"""
    raw = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500, temperature=0.3
    ).choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw)


def analyse_gap(job_description, profile):
    your_skills = set()
    for items in profile["skills"].values():
        your_skills.update([s.lower() for s in items])

    jd_prompt = f"""Extract skills from this JD. Return ONLY JSON, no markdown:
{{"required":["..."],"preferred":["..."],"soft_skills":["..."]}}
JD: {job_description}"""
    raw = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": jd_prompt}],
        max_tokens=400, temperature=0.1
    ).choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    jd_skills = json.loads(raw)

    matched = [s for s in jd_skills["required"] if s.lower() in your_skills]
    missing_required = [s for s in jd_skills["required"] if s.lower() not in your_skills]
    missing_preferred = [s for s in jd_skills["preferred"] if s.lower() not in your_skills]

    recs = []
    if missing_required or missing_preferred:
        rec_prompt = f"""For each missing skill give learning advice. Return ONLY a JSON array, no markdown:
[{{"skill":"...","priority":"high/medium","what_to_learn":"...","best_resource":"...","time_estimate":"..."}}]
Missing required: {missing_required}
Missing preferred: {missing_preferred}
Job context: {job_description[:300]}"""
        raw2 = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": rec_prompt}],
            max_tokens=800, temperature=0.3
        ).choices[0].message.content.strip()
        if raw2.startswith("```"):
            raw2 = raw2.split("```")[1]
            if raw2.startswith("json"): raw2 = raw2[4:]
        recs = json.loads(raw2)

    return {"matched": matched, "missing_required": missing_required, "missing_preferred": missing_preferred, "recommendations": recs}
