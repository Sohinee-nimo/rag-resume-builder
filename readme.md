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
  ┌─────────────┐
  │  Embedding  │  sentence-transformers converts
  │   Model     │  each chunk into a vector
  │             │  (384 numbers capturing meaning)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  ChromaDB   │  Vectors stored in a local
  │  Vector DB  │  vector database, indexed
  │             │  for fast similarity search
  └─────────────┘

JOB DESCRIPTION (every time)
        │
        ▼
  ┌─────────────┐
  │  Embed JD   │  Same model embeds the JD
  └──────┬──────┘  into the same vector space
         │
         ▼
  ┌─────────────┐
  │  Similarity │  Cosine similarity finds the
  │   Search    │  top-k most relevant chunks
  │   (top-k)   │  from your profile
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Prompt    │  Retrieved chunks + JD +
  │ Engineering │  instruction assembled into
  │             │  a grounded LLM prompt
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  LLaMA 3.3  │  LLM generates tailored CV
  │  70B (Groq) │  as structured JSON —
  │             │  only from retrieved context
  └─────────────┘
```

### Why RAG instead of just prompting?

A naive approach would be: dump your entire CV into the prompt and say "tailor this." That breaks in two ways — the LLM loses focus with too much context, and it starts hallucinating impressive-sounding details you don't actually have.

RAG fixes this by **retrieving only the relevant chunks** before generation. The LLM gets a short, targeted context — just the experience that actually matches the job. This produces sharper, more honest output and prevents fabrication.

---

## Features

- **One-time profile setup** — fill a form once, reuse for every application
- **Semantic search** — finds relevant experience even when exact words differ ("AWS" matches "cloud infrastructure")
- **Tailored CV generation** — rewrites bullet points to match JD language and priorities
- **Skill gap analysis** — compares JD requirements against your profile, scores your match percentage
- **Learning roadmap** — recommends specific topics, resources, and time estimates for missing skills
- **Download** — exports CV and gap report as `.txt` files

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, runs locally, no API cost |
| Vector database | ChromaDB | Simple, in-memory, zero config |
| LLM | Groq + LLaMA 3.3 70B | Free API tier, ~500 tokens/sec |
| UI | Streamlit | Fast to build, easy to deploy |
| Language | Python 3.10+ | — |

**Total API cost: $0.** Embeddings run locally on CPU. LLM calls go through Groq's free tier.

---

## Project structure

```
rag-resume-builder/
│
├── app.py              # Streamlit UI — profile form, CV tab, gap analysis tab
├── rag_engine.py       # All RAG logic — chunking, embedding, retrieval, generation
├── requirements.txt    # Python dependencies
├── .env                # GROQ_API_KEY (local only, never committed)
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

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# Run
streamlit run app.py
```

---

## Key design decisions

**Chunking strategy** — each chunk maps to one logical unit of experience (one job role, one skill cluster, one project). Smaller chunks = more precise retrieval. If the whole CV were one chunk, every query would retrieve everything and nothing would be targeted.

**Cosine similarity over keyword search** — traditional search would miss "built scalable APIs" when the JD says "REST API development." Cosine similarity on embeddings captures semantic meaning, so linguistically different but conceptually similar text scores highly.

**Temperature 0.3 for generation** — low temperature keeps the LLM factual and consistent. Higher values produce more creative output but also more hallucinations — the opposite of what you want in a CV.

**Structured JSON output** — the LLM is prompted to return strict JSON rather than free text. This makes the output programmatically parseable and lets the UI render it cleanly without regex hacks.

**Skill gap uses set intersection, not AI** — comparing your skills to JD requirements is a plain Python set operation, not an LLM call. The LLM is only used where it genuinely adds value: extracting unstructured skills from JD text and generating learning advice.

---

## What I learned building this

- How embedding models represent semantic meaning as vectors and why two different phrases can have near-identical embeddings
- How ChromaDB indexes vectors using HNSW (Hierarchical Navigable Small World) for fast approximate nearest-neighbour search
- How chunking strategy directly affects retrieval quality — too large loses precision, too small loses context
- How to write grounded LLM prompts that prevent hallucination by constraining the model to retrieved context only
- How to structure a RAG pipeline so the retrieval logic is decoupled from the generation logic

---

## Potential extensions

- [ ] PDF export using `reportlab` or `weasyprint`
- [ ] Persistent ChromaDB so profile survives session restarts
- [ ] Support multiple profiles (one per user)
- [ ] Cover letter generator using the same RAG pipeline
- [ ] Match score history — track how your fit improves as you learn new skills

---

## Author

Built by **Sohinee Mondal** as a project to learn RAG from scratch by building something real.

[LinkedIn](www.linkedin.com/in/sohinee-mondal) · [GitHub](https://github.com/Sohinee-nimo)
