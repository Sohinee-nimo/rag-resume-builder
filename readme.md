# RAG Resume Builder
---
title: RAG Resume Builder
emoji: 📄
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

> Paste any job description. Get a CV tailored to it in seconds — grounded in your real experience, never hallucinated.

**Live demo →** [Streamlit](https://rag-resume-builder-wbt7wk7gczr8h7v7mhuhea.streamlit.app/)

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
│  Embedding model │  Local SentenceTransformer
│  all-MiniLM-L6   │  runs inside the app container
│   (via PyTorch)  │  to convert text to vectors
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
│   Embed JD  │  Same model embeds the JD
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
│   LLaMA 3.3 70B  │  Groq API generates CV
│   via Groq API   │  as structured JSON —
│                  │  only from retrieved context
└──────────────────┘

```

### Why RAG instead of just prompting?

A naive approach would be: dump your entire CV into the prompt and say "tailor this." That breaks in two ways — the LLM loses focus with too much context, and it starts hallucinating impressive-sounding details you don't actually have.

RAG fixes this by retrieving only the relevant chunks before generation. The LLM gets a short, targeted context — just the experience that actually matches the job. This produces sharper, more honest output and prevents fabrication.

### Why a custom vector store instead of ChromaDB?

ChromaDB pulled in `opentelemetry`, `protobuf`, and `grpcio` as dependencies. These libraries had cascading version conflicts with NumPy and Python execution environments on free deployment platforms (Streamlit Cloud, Render, Vercel). The app crashed at import before a single line of our code ran.

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

* **One-time profile setup** — fill a form once, reuse for every application
* **Semantic search** — finds relevant experience even when exact words differ ("AWS" matches "cloud infrastructure")
* **Tailored CV generation** — rewrites bullet points to match JD language and priorities
* **Skill gap analysis** — compares JD requirements against your profile, scores your match percentage
* **Learning roadmap** — recommends specific topics, resources, and time estimates for missing skills
* **Local Vectors** — fully self-contained text embedding without reliance on unstable external inference keys
* **Download** — exports CV and gap report as `.txt` files

---

## Tech stack

| Layer | Tool | Why |
| --- | --- | --- |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, internal execution container. Immune to external API network drops or DNS resolution failures. |
| **Vector store** | Custom `VectorStore` class | Zero external DB dependencies, zero deployment conflicts |
| **LLM** | Groq + LLaMA 3.3 70B | Free tier, ultra-fast generation on dedicated LPU hardware |
| **UI** | Streamlit | Fast to build, clean web-form architecture |
| **Similarity** | NumPy cosine similarity | No extra heavy software needed, standard dot-product math |
| **Language** | Python 3.11 | Optimized environment |
| **Hosting** | Streamlit Cloud | Free, auto-deploys from GitHub on push |

**Total running cost: Rs. 0.** Every layer uses a free tier or an open-source library.

---

## Project structure

```
rag-resume-builder/
│
├── app.py              # Streamlit UI — profile form, CV tab, gap analysis tab
├── rag_engine.py       # All RAG logic — chunking, local embedding, retrieval, generation
├── requirements.txt    # Four clean dependencies: streamlit, groq, numpy, sentence-transformers
├── runtime.txt         # Pins Python 3.11 for Streamlit Cloud
├── .env                # GROQ_API_KEY (local only, never committed to git)
├── .gitignore
└── README.md

```

---

## Run it locally

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/rag-resume-builder.git](https://github.com/YOUR_USERNAME/rag-resume-builder.git)
cd rag-resume-builder

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_groq_key" > .env

# Run the app
streamlit run app.py

```

Get your free key:

* Groq API key → [console.groq.com](https://console.groq.com) (No credit card required)

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**.
3. Select your repository, branch `main`, and main file paths as `app.py`.
4. Click **Advanced settings** → **Secrets**, and add your model engine credential keys:
```toml
GROQ_API_KEY = "your_groq_key"

```


5. Click **Deploy**. Your app will be live in about 2 minutes!

---

## Key design decisions

**Local sentence-transformers over Web API** — While hosted API inference tools save space, free hosting platforms frequently suffer from cluster DNS bugs (`NameResolutionError`). Processing the **`all-MiniLM-L6-v2`** model internally within your container space makes the RAG logic faster and immune to third-party web routing timeouts.

**Streamlit Caching for Heavy Resources** — Reloading neural network model files on every interaction consumes system memory. Wrapping the model load in `@st.cache_resource` ensures it parses into server memory exactly once on initial startup, allowing instantaneous sub-second user queries.

**Chunking strategy** — Each chunk maps to one logical unit of experience (one job role, one skill cluster, one project). Smaller chunks mean more precise retrieval. If the whole CV were one chunk, every query would retrieve everything and nothing would be targeted.

**Cosine similarity over keyword search** — Traditional keyword search would miss "built scalable APIs" if the JD says "REST API development." Cosine similarity on embeddings captures semantic meaning, so linguistically different but conceptually similar text scores highly.

**temperature=0.3 for generation** — A low temperature keeps the LLM factual and consistent. Higher values produce more creative output but also more hallucinations — the opposite of what you want in a professional CV.

**Structured JSON output** — The LLM is prompted to return strict JSON rather than free text. This makes output programmatically parseable and lets the UI render sections cleanly without fragile regex parser hacks.

**Anti-hallucination rule in prompt** — The single most important line: *"Only use facts from CANDIDATE BACKGROUND. Do not invent anything."* This forces every generated bullet point to ground itself in retrieved reality.

**Skill gap uses set intersection, not AI** — Comparing your skills to JD requirements is a plain Python set operation, not an LLM call. The LLM is only used where it genuinely adds value: extracting unstructured skills from raw JD text, and generating learning advice. Everything else is deterministic code.

---

## Deployment challenges solved

**Bug 1 — External API NameResolutionError Overcome**
Using raw web requests to external endpoints frequently caused the application to crash with a `gaierror: [Errno -2]` when cloud infrastructure dropped external host links.
*Fix:* Migrated the pipeline to a local execution layer with `sentence-transformers`. By pairing this with the modern `uv` pip package manager on Streamlit Cloud, the runtime environment downloads and spins up the weights natively without breaking server memory caps.

**Bug 2 — ChromaDB breaks on modern python environments**
ChromaDB's telemetry module imported `opentelemetry` which had deprecated protobuf descriptors that crashed on startup.
*Fix:* Eliminated ChromaDB entirely and replaced it with our custom `VectorStore` class.

**Bug 3 — ChromaDB breaks on NumPy 2.0**
Older vector databases used legacy types like `np.float_` which were removed in modern NumPy releases.
*Fix:* Writing a clean vector class utilizing fundamental matrix vector operations ensures the code is completely independent of dependency deprecation cycles.

---

## Potential extensions

* [ ] PDF export using `reportlab` or `weasyprint`
* [ ] Persistent vector store — save with `numpy.save()` so profile survives session restarts
* [ ] Cover letter generator using the same RAG pipeline
* [ ] Match score history — track how your fit improves as you learn new skills
* [ ] Multi-user support with profile stored per user in a database
* [ ] Interview question generator based on JD + your profile gaps

---


## Author

Built by **[Sohinee]** as a hands-on project to learn RAG from scratch.

[LinkedIn](www.linkedin.com/in/sohinee-mondal) · [GitHub](https://github.com/Sohinee-nimo)
