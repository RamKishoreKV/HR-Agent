

# HR Hiring Copilot

An AI-first hiring assistant that turns a short request like “hire a founding engineer and a GenAI intern” into:
- LinkedIn-style job descriptions
- A structured, week-by-week hiring plan (steps, owners, timeline)
- An interview kit (screening questions, evaluation rubric, scorecard)
- Useful tools (search, email drafts, LinkedIn boolean query, job board shortcuts)

Built with LangGraph and Streamlit. Runs on OpenAI or local Ollama.

[![Demo Video](https://img.shields.io/badge/Demo-Video-green?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1SuRrD0AJhNF_aBqM14lQZuACt7Lc_yon/view?usp=sharing)


## 🚀 Features

- Hiring Wizard: Single, streamlined flow 
- JD Generation: Startup tone; comp note with sensible defaults
- Hiring Plan: Steps, owners, timeline in JSON and Markdown
- Interview Kit: Screening questions, evaluation guide, rubric, scorecard
- Tools: Search simulator, Email writer, LinkedIn search, ATS checker, Job board shortcuts
- Session Memory & Analytics: File-based sessions and usage stats
- LLM Providers: OpenAI or local Ollama (toggle in UI)

## 🏗️ Architecture

```
┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI │    │     LangGraph    │    │   LLM Provider  │
│  (Wizard+UI)  │◄──►│  clarify → … →   │◄──►│  OpenAI / Ollama │
└───────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
  File Sessions           Deterministic             External Tools
   + Analytics            Markdown Assembly        (Search/Email/etc.)
```

### Workflow Nodes

1) clarify: optional clarifying prompts (LLM)
2) extract: slot extraction (roles, timeline_weeks, budget_range, location, etc.)
3) jd: JD JSON per role with startup tone
4) plan: hiring plan JSON (steps, owners, timeline)
5) kit: interview kit (questions, rubric, scorecard)
6) assemble: final Markdown + JSON (deterministic)

## 📦 Setup

### Prerequisites
- Python 3.10+
- Git

### Install
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment Variables
Create a `.env` in the project root (examples):
```bash
# Choose one: openai or ollama
LLM_PROVIDER=ollama

# Ollama (local)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b

# OpenAI (cloud)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```
## ⚙️ Notes on LLM Providers

- **Ollama (default / tested):**  
  This project was primarily built and tested with [Ollama](https://ollama.com/) running locally.  
  Make sure Ollama is installed and running before starting the app:  

  ```bash
  ollama serve
  ollama pull llama3.1:8b

  The Streamlit wizard and workflow were validated end-to-end using Ollama.

OpenAI (optional):
The code also supports OpenAI models (via OPENAI_API_KEY), but since usage incurs cost, it was not tested extensively in this build.


## 🎯 Run Locally
```bash
streamlit run ui/streamlit_app.py
```
Open `http://localhost:8501`.

## Usage
1) Type a hiring request (e.g., “Help me hire a data analyst”).  
2) The Wizard opens. Fill timeline, budget, seniority, location, skills.  
3) Generate: JD(s) + plan + interview kit.  
4) Review: Results tabs (Cards, Markdown, JSON, Refine).  
5) Tools: Search, Email, LinkedIn search, ATS check, Job board shortcuts.

### Multi-role
If the input mentions multiple roles, the wizard shows separate role forms.

### Refinements
Use the “Refine” tab to request changes; outputs update accordingly.

## Tools
- Search Simulator: Google CSE → DuckDuckGo → LLM → curated
- Email Writer: Outreach/scheduling drafts using JD context
- LinkedIn Candidate Search: Boolean query + URL
- ATS Resume Checker: Heuristic JD match score
- Job Board Posting Shortcuts: Copy-ready JD, .txt/.html download, LinkedIn/Indeed links

## Sessions & Analytics
- Sessions saved under `data/sessions/`; reload via sidebar.  
- Analytics shown in the sidebar; raw log in `data/analytics.jsonl`.

## 🧪 Testing
```bash
python -m pytest tests/ -v
# quick
python tests/test_graph.py
# provider connectivity
python test_connection.py
```

## ☁️ Deploy (Streamlit Cloud)
- Push to GitHub.  
- Create an app from the repo.  
- Set env vars (`LLM_PROVIDER`, `OPENAI_API_KEY`/`OLLAMA_*`, optional Google keys).  
- Adjust resources for larger models if needed.

## 🛠 Troubleshooting
- No LLM response: check `.env` and run `python test_connection.py`.
- - Ollama: Make sure Ollama is installed and running. Start with `ollama serve` and ensure the model is pulled with `ollama pull llama3.1:8b`. If Ollama isn’t running, you’ll see connection errors in Streamlit.
- Google CSE empty: verify `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`.
- UI glitches: use sidebar “Reset Session”.


## 🧰 Tech Stack Used
- Streamlit: UI and session state
- LangGraph: Multi-step workflow orchestration
- OpenAI/Ollama: LLM providers (configurable in `.env`)
- Pydantic: Typed data models
- Requests: HTTP calls (Ollama/OpenAI, search)
- Python-dotenv: Environment configuration
- Pytest: Tests

## 🧭 Design Decisions
- LLM-first, deterministic assembly: Use LLMs for flexible generation; assemble final markdown/JSON deterministically to avoid hallucinated salary.
- Single-path UX: Removed alternate chat mode; hiring intent routes directly to the Wizard for speed and clarity.
- Robust JSON parsing: Tolerant parsing to support local models; strict structure enforced post-parse.
- File-based state: Sessions and analytics stored as JSON for simplicity, portability, and easy debugging.
- Tooling fallbacks: Search tries Google CSE → DuckDuckGo → LLM synthesis → curated links for reliability without keys.

## 🔄 What I’d Improve With More Time
- RAG knowledge base for hiring best practices/templates
- ATS integrations (Greenhouse/Lever exports, webhooks)
- Calendar integration for interview loops
- Salary benchmarking by role/location (bands + geo modifiers)
- Multi-user persistence (DB) and auth for team use
- Social share generator (LinkedIn post, X thread)

---
For a deeper code walkthrough, see `docs/ARCHITECTURE.txt`.
