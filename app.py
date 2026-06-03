# app.py
# The entire web interface — run with: streamlit run app.py

import streamlit as st
import rag_engine as rag

st.set_page_config(page_title="RAG Resume Builder", page_icon="📄", layout="wide")
st.title("RAG Resume Builder")
st.caption("Enter your profile once. Generate a tailored CV for any job.")

# ── session state ─────────────────────────────────────────────
if "profile" not in st.session_state:
    st.session_state.profile = None
if "ingested" not in st.session_state:
    st.session_state.ingested = False

# ── sidebar: profile form ─────────────────────────────────────
with st.sidebar:
    st.header("Your Profile")
    st.caption("Fill once — reused for every CV.")

    name    = st.text_input("Full name")
    summary = st.text_area("Professional summary", height=100)

    st.subheader("Experience")
    num_jobs = st.number_input("Number of roles", 1, 5, 2)
    experience = []
    for i in range(num_jobs):
        with st.expander(f"Role {i+1}", expanded=(i==0)):
            experience.append({
                "role":     st.text_input("Job title",   key=f"role_{i}"),
                "company":  st.text_input("Company",     key=f"company_{i}"),
                "duration": st.text_input("Duration",    key=f"duration_{i}"),
                "details":  st.text_area("Details",      key=f"details_{i}", height=80)
            })

    st.subheader("Skills")
    langs  = st.text_input("Languages",   "Python, JavaScript")
    frames = st.text_input("Frameworks",  "Django, React")
    tools  = st.text_input("Tools",       "Docker, AWS, Git")
    soft   = st.text_input("Soft skills", "Team leadership, Agile")

    st.subheader("Projects")
    num_proj = st.number_input("Number of projects", 1, 4, 2)
    projects = []
    for i in range(num_proj):
        with st.expander(f"Project {i+1}", expanded=(i==0)):
            projects.append({
                "name":    st.text_input("Project name", key=f"pname_{i}"),
                "details": st.text_area("Details",       key=f"pdet_{i}", height=60)
            })

    edu   = st.text_input("Education")
    certs = st.text_input("Certifications (comma separated)")

    if st.button("Save & Ingest Profile", type="primary"):
        if not name:
            st.error("Name is required.")
        else:
            st.session_state.profile = {
                "name": name, "summary": summary,
                "experience": [e for e in experience if e["role"]],
                "skills": {
                    "languages":   [s.strip() for s in langs.split(",") if s.strip()],
                    "frameworks":  [s.strip() for s in frames.split(",") if s.strip()],
                    "tools":       [s.strip() for s in tools.split(",") if s.strip()],
                    "soft_skills": [s.strip() for s in soft.split(",") if s.strip()]
                },
                "projects": [p for p in projects if p["name"]],
                "education": edu,
                "certifications": [s.strip() for s in certs.split(",") if s.strip()]
            }
            with st.spinner("Embedding your profile..."):
                count = rag.ingest_profile(st.session_state.profile)
            st.session_state.ingested = True
            st.success(f"Profile saved — {count} chunks stored.")

# ── main area: two tabs ───────────────────────────────────────
tab1, tab2 = st.tabs(["Generate CV", "Skill Gap Analysis"])

with tab1:
    st.header("Generate a Tailored CV")
    if not st.session_state.ingested:
        st.info("Fill your profile in the sidebar and click Save first.")
    else:
        jd = st.text_area("Paste the job description", height=200)
        if st.button("Generate CV", type="primary"):
            if not jd.strip():
                st.warning("Paste a job description first.")
            else:
                with st.spinner("Retrieving relevant experience and generating CV..."):
                    cv = rag.generate_cv(jd, st.session_state.profile["name"])

                st.success("CV generated!")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader(cv["name"])
                    st.caption(cv["tagline"])
                    st.markdown("**Summary**")
                    st.write(cv["summary"])
                    st.markdown("**Experience**")
                    for job in cv["experience"]:
                        st.markdown(f"**{job['role']}** — {job['company']} ({job['duration']})")
                        for b in job["bullets"]:
                            st.markdown(f"- {b}")
                    st.markdown("**Projects**")
                    for p in cv["projects"]:
                        st.markdown(f"- **{p['name']}**: {p['description']}")
                    st.markdown("**Education**")
                    st.write(cv["education"])
                with col2:
                    st.markdown("**Skills**")
                    st.write("Technical: " + ", ".join(cv["skills"]["technical"]))
                    st.write("Soft: " + ", ".join(cv["skills"]["soft"]))

                # download button
                cv_text = f"{cv['name']}\n{cv['tagline']}\n\n{cv['summary']}\n\nEXPERIENCE\n"
                for job in cv["experience"]:
                    cv_text += f"\n{job['role']} | {job['company']} | {job['duration']}\n"
                    for b in job["bullets"]: cv_text += f"  • {b}\n"
                st.download_button("Download CV (.txt)", cv_text, file_name="tailored_cv.txt")

with tab2:
    st.header("Skill Gap Analysis")
    if not st.session_state.ingested:
        st.info("Fill your profile in the sidebar and click Save first.")
    else:
        jd_gap = st.text_area("Paste the job description to analyse", height=200, key="jd_gap")
        if st.button("Analyse Gap", type="primary"):
            if not jd_gap.strip():
                st.warning("Paste a job description first.")
            else:
                with st.spinner("Analysing skill gap..."):
                    result = rag.analyse_gap(jd_gap, st.session_state.profile)

                total = len(result["matched"]) + len(result["missing_required"])
                pct = round(len(result["matched"]) / total * 100) if total else 0

                st.metric("Match score", f"{pct}%", f"{len(result['matched'])} of {total} required skills covered")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"You have ({len(result['matched'])})")
                    for s in result["matched"]: st.write(f"✓ {s}")
                with col2:
                    st.error(f"Missing required ({len(result['missing_required'])})")
                    for s in result["missing_required"]: st.write(f"✗ {s}")
                with col3:
                    st.warning(f"Missing preferred ({len(result['missing_preferred'])})")
                    for s in result["missing_preferred"]: st.write(f"○ {s}")

                if result["recommendations"]:
                    st.subheader("Your Learning Roadmap")
                    for rec in result["recommendations"]:
                        with st.expander(f"{'🔴' if rec['priority']=='high' else '🟡'} {rec['skill']} — {rec['time_estimate']}"):
                            st.write(f"**What to learn:** {rec['what_to_learn']}")
                            st.write(f"**Best resource:** {rec['best_resource']}")