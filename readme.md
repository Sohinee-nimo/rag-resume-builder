# RAG Resume Builder

> Paste any job description. Get a CV tailored to it in seconds — grounded in your real experience, never hallucinated.

**Live demo →** [Streamlit](https://rag-resume-builder-hwvuqmj76jeajww4tqtbl4.streamlit.app/))

---

## What this project does

Most AI CV tools write generic content or make things up. This one doesn't.

It uses **Retrieval-Augmented Generation (RAG)** — a technique where the AI is only allowed to use *your actual experience* when writing. You enter your profile once. For every job you apply to, you paste the job description and the system:

1. Finds the most relevant parts of your background for that specific role
2. Feeds only those parts to the LLM as context
3. Generates a CV that is tailored, specific, and factually grounded

It also analyses the skill gap between you and the job, and recommends exactly what to learn to become a stronger candidate.

---

## How RAG works — the architecture

```
YOUR PROFILE (one time)
        │
        ▼
  ┌─────────────┐
  │   Chunking  │  Split profile into sections
  │             │  (one job role = one chunk,
  │             │   skills = one chunk, etc.)
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  Embedding model │  HuggingFace Inference API
  │  all-MiniLM-L6   │  converts each chunk into
  │                  │  384 numbers (a vector)
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │   VectorStore    │  Custom in-memory store —
  │   (pure Python)  │  3 synced lists holding
  │                  │  vectors, text, metadata
  └──────────────────┘

JOB DESCRIPTION (every time)
        │
        ▼
  ┌─────────────┐
  │  Embed JD   │  Same model embeds the JD
  └──────┬──────┘  into the same vector space
         │
         ▼
  ┌──────────────────┐
  │ Cosine similarity│  numpy dot product finds
  │     search       │  top-k most relevant chunks
  │    (top-k=5)     │  from your profile
  └──────┬───────────┘
         │
         ▼
  ┌─────────────┐
  │   Prompt    │  Retrieved chunks + JD +
  │ engineering │  strict rules assembled
  │             │  into a grounded prompt
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  LLaMA 3.3 70B   │  Groq API generates CV
  │  via Groq API    │  as structured JSON —
  │                  │  only from retrieved context
  └──────────────────┘
```

### Why RAG instead of just prompting?

A naive approach would be: dump your entire CV into the prompt and say "tailor this." That breaks in two ways — the LLM loses focus with too much context, and it starts hallucinating impressive-sounding details you don't actually have.

RAG fixes this by retrieving only the relevant chunks before generation. The LLM gets a short, targeted context — just the experience that actually matches the job. This produces sharper, more honest output and prevents fabrication.

### Why a custom vector store instead of ChromaDB?

ChromaDB pulled in `opentelemetry`, `protobuf`, and `grpcio` as dependencies. These three libraries had cascading version conflicts with NumPy 2.0 and Python 3.11 on every free deployment platform (Streamlit Cloud, Render, Vercel). The app crashed at import before a single line of our code ran.

The solution: a 20-line pure Python `VectorStore` class using three synced lists and NumPy cosine similarity. Zero extra dependencies, identical functionality for our use case, deploys everywhere cleanly.

```python
class VectorStore:
    def __init__(self):
        self.embeddings = []   # numpy arrays
        self.documents  = []   # original text
        self.metadatas  = []   # section labels

    def query(self, query_embedding, top_k=5):
        q = np.array(query_embedding)
        scores = [np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-10)
                  for e in self.embeddings]
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"text": self.documents[i], "section": self.metadatas[i]["section"]}
                for i in top]
```

This is the same maths ChromaDB uses internally — just without the packaging overhead.

---

## Features

- One-time profile setup — fill a form once, reuse for every application
- Semantic search — finds relevant experience even when exact words differ ("AWS" matches "cloud infrastructure")
- Tailored CV generation — rewrites bullet points to match JD language and priorities
- Skill gap analysis — compares JD requirements against your profile, scores your match percentage
- Learning roadmap — recommends specific topics, resources, and time estimates for missing skills
- Download — exports CV and gap report as `.txt` files

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Embeddings | HuggingFace Inference API (all-MiniLM-L6-v2) | Free, no PyTorch install, runs on HF servers |
| Vector store | Custom `VectorStore` class | Zero dependencies, no deployment conflicts |
| LLM | Groq + LLaMA 3.3 70B | Free tier, ~500 tokens/sec on custom LPU hardware |
| UI | Streamlit | Fast to build, one-command deploy |
| Similarity | NumPy cosine similarity | No extra library needed, same maths as any vector DB |
| Language | Python 3.11 | — |
| Hosting | Streamlit Cloud | Free, auto-deploys from GitHub on push |

**Total cost: Rs. 0.** Every layer uses a free tier or open-source library.

---

## Project structure

```
rag-resume-builder/
│
├── app.py              # Streamlit UI — profile form, CV tab, gap analysis tab
├── rag_engine.py       # All RAG logic — chunking, embedding, retrieval, generation
├── requirements.txt    # Four dependencies only: streamlit, groq, requests, numpy
├── runtime.txt         # Pins Python 3.11 for Streamlit Cloud
├── .env                # GROQ_API_KEY + HF_TOKEN (local only, never committed)
├── .gitignore
└── README.md
```

The separation between `app.py` and `rag_engine.py` is intentional — the RAG pipeline is completely UI-agnostic. You could swap Streamlit for a FastAPI backend or a CLI without touching `rag_engine.py`.

---

## Run it locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/rag-resume-builder.git
cd rag-resume-builder

# Install (only 4 packages — installs in seconds)
pip install -r requirements.txt

# Add API keys
echo "GROQ_API_KEY=your_groq_key" > .env
echo "HF_TOKEN=your_hf_token" >> .env

# Run
streamlit run app.py
```

Get your free keys:
- Groq API key → [console.groq.com](https://console.groq.com) (no credit card)
- HuggingFace token → [huggingface.co](https://huggingface.co) → Settings → Access Tokens

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → New app
3. Pick your repo, branch `main`, main file `app.py`
4. Under Advanced settings → Secrets, add:
   ```
   GROQ_API_KEY = "your_groq_key"
   HF_TOKEN     = "your_hf_token"
   ```
5. Click Deploy — live in ~2 minutes at `your-app.streamlit.app`

---

## Key design decisions

**Chunking strategy** — each chunk maps to one logical unit of experience (one job role, one skill cluster, one project). Smaller chunks = more precise retrieval. If the whole CV were one chunk, every query would retrieve everything and nothing would be targeted.

**Cosine similarity over keyword search** — traditional keyword search would miss "built scalable APIs" when the JD says "REST API development." Cosine similarity on embeddings captures semantic meaning, so linguistically different but conceptually similar text scores highly.

**HuggingFace API over local sentence-transformers** — `sentence-transformers` requires PyTorch (~800MB). Every free deployment platform has a build memory cap around 512MB–1GB, so the build crashed before the app started. The HuggingFace Inference API runs the same model on their servers — identical output, ~50MB install size.

**temperature=0.3 for generation** — low temperature keeps the LLM factual and consistent. Higher values produce more creative output but also more hallucinations — the opposite of what you want in a CV.

**Structured JSON output** — the LLM is prompted to return strict JSON rather than free text. This makes output programmatically parseable and lets the UI render sections cleanly without regex hacks. The schema is defined explicitly in the prompt.

**Anti-hallucination rule in prompt** — the single most important line: "Only use facts from CANDIDATE BACKGROUND. Do not invent anything." Without this, the LLM invents impressive-sounding experience the candidate doesn't have. This grounds every generated bullet to retrieved reality.

**Skill gap uses set intersection, not AI** — comparing your skills to JD requirements is a plain Python set operation, not an LLM call. The LLM is only used where it genuinely adds value: extracting unstructured skills from raw JD text, and generating learning advice. Everything else is deterministic code.

---

## What I learned building this

- How embedding models represent semantic meaning as vectors — why "AWS" and "cloud infrastructure" get similar embeddings despite sharing no words
- How cosine similarity works geometrically — measuring the angle between vectors rather than their magnitude
- How chunking strategy directly affects retrieval quality — too large loses precision, too small loses context
- How to write grounded LLM prompts that prevent hallucination by constraining the model to retrieved context only
- How to structure a RAG pipeline so retrieval logic is decoupled from generation logic
- How to debug dependency conflicts across Python packages — NumPy 2.0 breaking changes, protobuf version pinning, build memory limits on free deployment tiers
- Why building a simple custom solution (20-line VectorStore) can be better engineering than reaching for a complex library (ChromaDB) when the use case doesn't need the extra features

---

## Deployment challenges solved

This project hit three real-world deployment bugs that are worth documenting:

**Bug 1 — sentence-transformers crashes build**
PyTorch is ~800MB. Streamlit Cloud's build limit is ~1GB. The build ran out of memory before the app started.
Fix: replaced with HuggingFace Inference API — same model, no local install.

**Bug 2 — ChromaDB breaks on Python 3.14**
ChromaDB's telemetry module imported `opentelemetry` which used deprecated protobuf descriptors removed in newer protobuf releases.
Fix: pinned `chromadb==0.4.24` and `numpy==1.26.4`, added `runtime.txt` to force Python 3.11.

**Bug 3 — ChromaDB breaks on NumPy 2.0**
`chromadb==0.5.x` used `np.float_` which was removed in NumPy 2.0. Streamlit Cloud installed NumPy 2.4.6 by default.
Fix: replaced ChromaDB entirely with a custom `VectorStore` class. Zero dependencies, guaranteed to deploy.

---

## Potential extensions

- [ ] PDF export using `reportlab` or `weasyprint`
- [ ] Persistent vector store — save with `numpy.save()` so profile survives session restarts
- [ ] Cover letter generator using the same RAG pipeline
- [ ] Match score history — track how your fit improves as you learn new skills
- [ ] Multi-user support with profile stored per user in a database
- [ ] Interview question generator based on JD + your profile gaps

---

## Author

Built by **[Sohinee]** as a hands-on project to learn RAG from scratch.

[LinkedIn](www.linkedin.com/in/sohinee-mondal) · [GitHub](https://github.com/Sohinee-nimo)
