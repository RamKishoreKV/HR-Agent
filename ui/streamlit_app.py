"""Streamlit UI for the HR Hiring Copilot."""

import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.agent import run_agent_langgraph, get_agent_status, validate_overrides
from app.comp import compose_comp_note
from app.llm import test_llm_connection, ask_llm
from app.memory import get_session_id, save_state, clear_session, get_analytics_summary, get_prefs, save_prefs, list_sessions
from app.schema import SessionState, SlotExtraction
from app.tools import simulate_search, email_writer, checklist_export


# ------------------------------ Helpers ------------------------------
def _is_hiring_request(user_input: str) -> bool:
    hiring_keywords = [
        "hire", "hiring", "recruit", "job", "position", "role",
        "candidate", "employee", "talent", "staff"
    ]
    return any(k in user_input.lower() for k in hiring_keywords)


def _clean_role_title(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r"^[^A-Za-z0-9]+", "", s)
    s = s.lower()
    # Remove leading helper phrases, articles, and conjunctions
    s = re.sub(r"^(help\s*me\s*hire|help\s*hire|hire|hiring|looking\s*for|need\s*to\s*hire|and|&|,|for)\s+", "", s)
    s = re.sub(r"^(a|an|the)\s+", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    # Fix common abbreviations/typos
    s = s.replace("gen ai", "generative ai").replace("genai", "generative ai")
    # Title case
    s = s.title()
    # Fix AI capitalization
    s = re.sub(r"\bAi\b", "AI", s)
    return s


def _detect_roles_from_text(text: str) -> list:
    try:
        if not text:
            return []
        # Find phrases ending with common role nouns
        matches = re.findall(r"([A-Za-z][A-Za-z \-/]+?(?:engineer|analyst|manager|scientist|designer|developer|intern))\b", text, flags=re.I)
        roles = [" ".join(m.strip().split()) for m in matches]
        # Clean up noise like 'help me hire' and 'and'
        roles = [_clean_role_title(r) for r in roles]
        # Deduplicate preserving order
        seen = set(); out = []
        for r in roles:
            if r not in seen:
                seen.add(r); out.append(r)
        return out
    except Exception:
        return []


    


def _apply_local_refinements(result, overrides: dict) -> None:
    import re as _re
    if not (overrides and result and result.jd_drafts):
        return
    tweak_text = overrides.get("tweak")
    if not tweak_text:
        return
    years_match = _re.search(r"(\d+)\s*\+?\s*years", tweak_text, flags=_re.I)
    replace_years = years_match.group(1) if years_match else None
    remove_tokens, add_tokens = [], []
    for m in _re.finditer(r"remove\s+([a-z0-9 \-/\+\.#]+)", tweak_text, flags=_re.I):
        token = m.group(1).strip()
        if token:
            remove_tokens.append(token)
    for m in _re.finditer(r"add\s+([a-z0-9 \-/\+\.#]+)", tweak_text, flags=_re.I):
        token = m.group(1).strip()
        if token:
            add_tokens.append(token)

    for jd in result.jd_drafts:
        if replace_years and jd.requirements:
            jd.requirements = [_re.sub(r"\b\d+\s*\+?\s*years\b", f"{replace_years}+ years", rq, flags=_re.I) for rq in jd.requirements]
        if remove_tokens:
            jd.responsibilities = [r for r in jd.responsibilities if not any(t.lower() in r.lower() for t in remove_tokens)]
            jd.requirements = [r for r in jd.requirements if not any(t.lower() in r.lower() for t in remove_tokens)]
        for t in add_tokens:
            if t.lower() not in [rq.lower() for rq in jd.requirements]:
                jd.requirements.append(t)
    # Exact salary override
    sal = overrides.get("salary_range") or overrides.get("budget_range")
    if sal and "-" not in sal.replace("–", "-").replace("—", "-"):
        exact = sal.strip()
        for jd in result.jd_drafts:
            jd.compensation_note = compose_comp_note(jd.title, overrides.get("company_stage") or "startup", exact)
    # Sync json
    if result.raw_json and result.raw_json.get("job_descriptions"):
        for i, jd_json in enumerate(result.raw_json["job_descriptions"]):
            if i < len(result.jd_drafts):
                jd_live = result.jd_drafts[i]
                jd_json["responsibilities"] = jd_live.responsibilities
                jd_json["requirements"] = jd_live.requirements
                jd_json["compensation_note"] = jd_live.compensation_note
def _is_refinement_request(user_input: str) -> bool:
    """Detect if the message is asking to modify/refine the current plan/JD."""
    if not user_input:
        return False
    refine_keywords = [
        "refine", "edit", "change", "update", "tweak", "improve", "expand",
        "shorten", "add ", "remove ", "replace", "make it more", "make it less"
    ]
    text = user_input.lower()
    return any(k in text for k in refine_keywords)


def _extract_salary_value(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        from app.prompts import SALARY_EXTRACT_PROMPT
        prompt = SALARY_EXTRACT_PROMPT.format(text=text)
        resp = ask_llm(prompt)
        cleaned = resp.replace("“", '"').replace("”", '"').replace("’", "'")
        start = cleaned.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        obj = json.loads(cleaned[start:i+1])
                        single = (obj.get("salary_single") or "").strip()
                        smin = (obj.get("salary_min") or "").strip()
                        smax = (obj.get("salary_max") or "").strip()
                        # Prefer single if present, else synthesize from range
                        if single:
                            return single
                        if smin and smax:
                            return f"{smin}-{smax}"
        # fallback regex
        m = re.search(r"\$?\s*(\d{2,3}(?:,\d{3})?)(k|K)?\b", text)
        if not m:
            return None
        num = m.group(1)
        has_k = bool(m.group(2))
        if has_k:
            return f"${num}k" if not num.startswith("$") else f"{num}k"
        return f"${num}"
    except Exception:
        m = re.search(r"\$?\s*(\d{2,3}(?:,\d{3})?)(k|K)?\b", text)
        if not m:
            return None
        num = m.group(1)
        has_k = bool(m.group(2))
        if has_k:
            return f"${num}k" if not num.startswith("$") else f"{num}k"
        return f"${num}"


def handle_user_message(user_input: str):
    """Process a single user message coming from the bottom input box."""
    if not user_input:
        return

    # Append to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Hiring request (LLM intent) → open Hiring Wizard directly
    try:
        from app.agent import _classify_intent_llm
        intent = _classify_intent_llm(user_input)
    except Exception:
        intent = "hiring" if _is_hiring_request(user_input) else "small_talk"
    if intent == "hiring":
        # Clear any legacy chat-mode states
        st.session_state.pop("pending_clarification", None)
        st.session_state.pop("pending_multi_roles", None)
        st.session_state.pop("awaiting_mode_choice", None)
        st.session_state.pop("chat_mode", None)
        # Open wizard with this intent
        st.session_state["hiring_intent"] = user_input
        st.session_state.wizard_mode = True
        st.rerun()
        return

    # Refinement flow → apply locally without regenerating
    if (st.session_state.get("last_output") or st.session_state.get("role_outputs")) and _is_refinement_request(user_input):
        overrides = dict(st.session_state.get("last_overrides", {}))
        overrides["tweak"] = user_input
        sal_single = _extract_salary_value(user_input)
        if sal_single:
            overrides["salary_range"] = sal_single
        # Do NOT persist tweak into session overrides to avoid reapplying on future generations
        st.session_state["last_overrides"] = {k: v for k, v in overrides.items() if k != "tweak"}

        # Choose target result: active role if set, else last_output
        target_result = None
        if st.session_state.get("role_outputs") and st.session_state.get("active_role") in st.session_state.role_outputs:
            target_result = st.session_state.role_outputs[st.session_state.active_role]
        else:
            target_result = st.session_state.get("last_output")

        # Prefer LLM rewrite for all JDs; fall back to local refiner per JD
        try:
            from app.agent import refine_jd_with_llm, refine_kit_with_llm
            lo = target_result
            if lo and lo.raw_json and lo.raw_json.get("job_descriptions") and lo.jd_drafts:
                jd_list = lo.raw_json["job_descriptions"]
                for idx, jd_json in enumerate(jd_list):
                    new_jd = refine_jd_with_llm(jd_json, user_input)
                    if new_jd and all(k in new_jd for k in ["title", "summary", "responsibilities", "requirements", "compensation_note"]):
                        # Apply to JD at idx
                        if idx < len(lo.jd_drafts):
                            lo.jd_drafts[idx].title = new_jd["title"]
                            lo.jd_drafts[idx].summary = new_jd["summary"]
                            lo.jd_drafts[idx].responsibilities = new_jd.get("responsibilities", [])
                            lo.jd_drafts[idx].requirements = new_jd.get("requirements", [])
                            lo.jd_drafts[idx].compensation_note = new_jd.get("compensation_note", lo.jd_drafts[idx].compensation_note)
                        jd_list[idx] = new_jd
                    else:
                        # Fallback per JD
                        _apply_local_refinements(lo, overrides)
                # Also refine Interview Kit if present
                kit = lo.raw_json.get("interview_kit") if lo.raw_json else None
                if kit:
                    refined_kit = refine_kit_with_llm(kit, user_input)
                    if isinstance(refined_kit, dict):
                        existing = lo.raw_json.get("interview_kit", {})
                        existing.update({k: v for k, v in refined_kit.items() if v})
                        lo.raw_json["interview_kit"] = existing
            else:
                _apply_local_refinements(lo, overrides)
        except Exception:
            _apply_local_refinements(target_result or st.session_state.last_output, overrides)
        # Post a compact confirmation to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Applied your changes to the current plan ({st.session_state.get('active_role') or 'current role'}). Check the Results pane."
        })
        # Mark refinement timestamp for UI badges
        st.session_state["last_refined_at"] = datetime.now().isoformat()
        # Save updated state to disk
        try:
            session_state = SessionState(
                session_id=st.session_state.session_id,
                last_input=st.session_state.get("hiring_intent") or "refinement",
                last_output=st.session_state.last_output,
                extraction=None,
                timestamp=datetime.now().isoformat(),
            )
            save_state(st.session_state.session_id, session_state)
        except Exception:
            pass
        st.rerun()
        return

    # Otherwise treat as general/small talk → ensure wizard is closed

    st.session_state.wizard_mode = False
    st.session_state.pop("hiring_intent", None)
    process_user_input(user_input)


# ------------------------------ Main App ------------------------------

def main():
    st.set_page_config(
        page_title="HR Hiring Copilot",
        page_icon="🤝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Init session vars
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_session_id()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_output" not in st.session_state:
        st.session_state.last_output = None
    if "role_outputs" not in st.session_state:
        st.session_state.role_outputs = {}
    if "wizard_mode" not in st.session_state:
        st.session_state.wizard_mode = False
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = os.getenv("LLM_PROVIDER", "ollama")

    # Apply provider selection for this run
    if st.session_state.selected_provider:
        os.environ["LLM_PROVIDER"] = st.session_state.selected_provider

    st.title("🤝 HR Hiring Copilot")
    st.caption("AI-powered hiring assistant for startups and growing companies")

    render_sidebar()

    # Always show Chat + Results layout (no toggle)
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        render_chat_interface()      # history + wizard (NO input box here)
    with col2:
        # Use containers to keep results and tools organized
        results_container = st.container()
        tools_container = st.container()
        with results_container:
            if st.session_state.last_output:
                render_results_section()
        with tools_container:
            render_tools_panel()

    # Removed: results rendering moved to right pane above

    # --- Bottom chat input ---
    user_input = st.chat_input("Ask me to help hire someone or ask a general question...")
    if user_input:
        handle_user_message(user_input)


# ------------------------------ Sidebar ------------------------------

def render_sidebar():
    st.sidebar.header("Settings")

    status = get_agent_status()
    current_provider = status["provider"]

    st.sidebar.subheader("🤖 LLM Provider")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button(
            "🦙 Ollama",
            type="primary" if current_provider == "ollama" else "secondary",
            help="Use local Ollama (default)",
            use_container_width=True,
        ):
            st.session_state.selected_provider = "ollama"
            st.rerun()
    with c2:
        if st.button(
            "🤖 OpenAI",
            type="primary" if current_provider == "openai" else "secondary",
            help="Use OpenAI API",
            use_container_width=True,
        ):
            st.session_state.selected_provider = "openai"
            st.rerun()

    if current_provider == "openai":
        if status["openai_configured"]:
            st.sidebar.success("✅ OpenAI API Key configured")
        else:
            st.sidebar.error("❌ OpenAI API Key missing")
            st.sidebar.caption("Add OPENAI_API_KEY to .env file")
    elif current_provider == "ollama":
        st.sidebar.info(f"🦙 Ollama: {status['ollama_url']}")
        st.sidebar.caption("Make sure Ollama is running locally")
    else:
        st.sidebar.error(f"❌ Invalid provider: {current_provider}")
        st.sidebar.caption("Check your .env file configuration")

    if st.sidebar.button("🔍 Test Connections", help="Test both OpenAI and Ollama connections"):
        with st.sidebar:
            with st.spinner("Testing connections..."):
                test_results = test_llm_connection()
            st.subheader("Connection Test Results")
            if test_results["openai"]["available"]:
                st.success("✅ OpenAI: Working")
                st.caption(f"Response: {test_results['openai']['response'][:50]}...")
            else:
                st.error("❌ OpenAI: Failed")
                st.caption(f"Error: {test_results['openai']['error']}")
            if test_results["ollama"]["available"]:
                st.success("✅ Ollama: Working")
                st.caption(f"Response: {test_results['ollama']['response'][:50]}...")
            else:
                st.error("❌ Ollama: Failed")
                st.caption(f"Error: {test_results['ollama']['error']}")

    st.sidebar.divider()

    st.sidebar.subheader("Session")
    st.sidebar.write(f"**ID:** {st.session_state.session_id[:8]}...")
    if st.sidebar.button("🔄 Reset Session", type="secondary"):
        clear_session(st.session_state.session_id)
        st.session_state.session_id = get_session_id()
        st.session_state.chat_history = []
        st.session_state.last_output = None
        st.session_state.wizard_mode = False
        st.session_state.pop("hiring_intent", None)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Analytics")
    analytics = get_analytics_summary()
    if analytics["total_runs"] > 0:
        st.sidebar.metric("Total Runs", analytics["total_runs"])
        if analytics["common_roles"]:
            st.sidebar.write("**Top Roles:**")
            for role_info in analytics["common_roles"][:3]:
                st.sidebar.write(f"• {role_info['role']} ({role_info['count']})")
    else:
        st.sidebar.write("No analytics data yet")

    # Session loader
    st.sidebar.divider()
    st.sidebar.subheader("Load Previous Session")
    try:
        sessions = list_sessions()
        if sessions:
            options = [f"{s['session_id']} — {s['timestamp'][:16]}" for s in sessions]
            sel = st.sidebar.selectbox("Session", options, index=0)
            if st.sidebar.button("Load Session"):
                from app.memory import load_state
                sess_id = sessions[options.index(sel)]['session_id']
                loaded = load_state(sess_id)
                if loaded and loaded.last_output:
                    st.session_state.session_id = sess_id
                    st.session_state.last_output = loaded.last_output
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Loaded session {sess_id}."})
                    st.rerun()
                else:
                    st.sidebar.warning("No output in selected session")
        else:
            st.sidebar.caption("No saved sessions yet")
    except Exception:
        pass

    st.sidebar.divider()
    st.sidebar.subheader("Display")
    if "plan_view" not in st.session_state:
        st.session_state.plan_view = "Detailed"
    st.session_state.plan_view = st.sidebar.selectbox("Plan View", ["Detailed", "Concise"], index=0, help="Concise view shows shorter bullets")

    # (Chat options removed)

    st.sidebar.subheader("Plan Style (Startup)")
    if "plan_style" not in st.session_state:
        st.session_state.plan_style = "Startup"
    st.session_state.plan_style = st.sidebar.selectbox("Style", ["Startup", "Enterprise", "Consulting"], index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Preferences (Startup)")
    prefs = get_prefs()
    default_remote = st.sidebar.checkbox("Always Remote", value=prefs.get("always_remote", False))
    default_salary = st.sidebar.text_input("Default Salary Range", value=prefs.get("default_salary", "competitive"))
    default_stage = st.sidebar.selectbox("Default Stage", ["startup", "scale-up", "enterprise"], index=["startup", "scale-up", "enterprise"].index(prefs.get("default_stage", "startup")))
    if st.sidebar.button("Save Preferences"):
        save_prefs({
            "always_remote": default_remote,
            "default_salary": default_salary.strip(),
            "default_stage": default_stage
        })
        st.sidebar.success("Preferences saved")


# ------------------------------ Left Column (no input box) ------------------------------

def render_chat_interface():
    """Show chat history and (if active) the Hiring Wizard. No input box here."""
    st.subheader("💬 Chat")

    # Show chat history
    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).write(m["content"])

    # Hiring Wizard (single path)
    if st.session_state.get("wizard_mode", False):
        intent = st.session_state.get("hiring_intent") or "help me hire a software engineer"
        with st.chat_message("assistant"):
            st.write("I'd love to help you with hiring! Let me gather some details first.")
            render_wizard(intent, form_key="hiring_wizard")

    # Per-role selector if we have multiple role outputs
    if st.session_state.get("role_outputs"):
        roles = list(st.session_state.role_outputs.keys())
        st.write("\n")
        st.markdown("**Active Role Chat**")
        if "active_role" not in st.session_state:
            st.session_state.active_role = roles[0]
        st.session_state.active_role = st.selectbox("Select role to refine/chat", roles, index=roles.index(st.session_state.active_role) if st.session_state.active_role in roles else 0, key="active_role_select")
        st.caption(f"Messages and refinements will apply to: {st.session_state.active_role}")


def render_wizard(initial_input: str, form_key: str = "hiring_wizard"):
    """Render the hiring wizard form."""
    st.subheader("📋 Hiring Wizard")
    # Multi-role detection
    detected_roles = _detect_roles_from_text(initial_input)
    if len(detected_roles) > 1:
        st.info("Multiple roles detected. Configure each role below and generate plans separately.")
        for idx, role_preset in enumerate(detected_roles):
            st.markdown(f"#### {role_preset}")
            with st.form(f"{form_key}_{idx}"):
                col1, col2 = st.columns(2)
                with col1:
                    # role fixed for this form
                    role_title = st.text_input("Job Title", value=role_preset)
                    timeline_weeks = st.number_input("Timeline (weeks)", min_value=1, max_value=52, value=8, key=f"tw_{idx}")
                    seniority = st.selectbox("Seniority Level", ["junior", "mid", "senior", "lead", "principal"], index=1, key=f"sen_{idx}")
                    location = st.text_input("Location", value="remote", key=f"loc_{idx}")
                with col2:
                    salary_presets = ["competitive", "$60k-$85k", "$80k-$120k", "$110k-$160k", "$140k-$200k", "$180k-$250k"]
                    preset_choice = st.selectbox("Salary Range (preset)", options=salary_presets, index=0, help="Predefined by HR", key=f"sal_{idx}")
                    custom_salary = st.text_input("Or enter custom salary range", value="", help="e.g., $95k-$130k or $120k", key=f"csal_{idx}")
                    employment_type = st.selectbox("Employment Type", ["full-time", "part-time", "contract", "intern"], index=0, key=f"emp_{idx}")
                    company_stage = st.selectbox("Company Stage", ["startup", "scale-up", "enterprise"], index=0, key=f"stg_{idx}")
                    key_skills = st.text_input("Key Skills", help="Comma-separated (e.g., Python, SQL)", key=f"ks_{idx}")

                submitted = st.form_submit_button(f"Generate for {role_preset}", type="primary")
                if submitted:
                    selected_salary = (custom_salary.strip() or preset_choice or "competitive").strip()
                    overrides = {
                        "timeline_weeks": timeline_weeks,
                        "salary_range": selected_salary,
                        "budget_range": selected_salary,
                        "roles": [role_title.strip()] if role_title.strip() else [],
                        "seniority": seniority,
                        "employment_type": employment_type,
                        "location": location,
                        "key_skills": [s.strip() for s in key_skills.split(",") if s.strip()],
                        "company_stage": company_stage or "startup",
                        "plan_style": st.session_state.get("plan_style", "Startup"),
                    }
                    st.session_state["last_overrides"] = overrides
                    # do not close the wizard: allow generating multiple role outputs
                    # Narrow the intent to the specific role for clarity
                    process_role_input(f"help me hire {role_title}", overrides, role_title)
        return
    with st.form(form_key):
        col1, col2 = st.columns(2)
        with col1:
            # Apply saved preferences for defaults
            prefs = get_prefs()
            role_title = st.text_input("Job Title", value="", placeholder="e.g., Data Analyst, Software Engineer")
            additional_roles = st.text_input("Additional Roles (comma-separated)", value="", placeholder="e.g., Data Engineer, BI Analyst")
            timeline_weeks = st.number_input("Timeline (weeks)", min_value=1, max_value=52, value=8)
            seniority = st.selectbox("Seniority Level", ["junior", "mid", "senior", "lead", "principal"], index=1)
            location_default = "remote" if prefs.get("always_remote", False) else "remote"
            location = st.text_input("Location", value=location_default)
        with col2:
            # HR-defined salary presets
            salary_presets = [
                "competitive",
                "$60k-$85k",
                "$80k-$120k",
                "$110k-$160k",
                "$140k-$200k",
                "$180k-$250k",
            ]
            default_salary = prefs.get("default_salary", "competitive")
            idx = salary_presets.index(default_salary) if default_salary in salary_presets else 0
            preset_choice = st.selectbox("Salary Range (preset)", options=salary_presets, index=idx, help="Predefined by HR")
            custom_salary = st.text_input("Or enter custom salary range", value="", help="e.g., $95k-$130k or $120k")
            employment_type = st.selectbox("Employment Type", ["full-time", "part-time", "contract", "intern"], index=0)
            stage_options = ["startup", "scale-up", "enterprise"]
            company_stage = st.selectbox("Company Stage", stage_options, index=stage_options.index(prefs.get("default_stage", "startup")))
            key_skills = st.text_input("Key Skills", help="Comma-separated (e.g., Python, SQL)")

        with st.expander("Advanced details (optional)"):
            colA, colB = st.columns(2)
            with colA:
                department = st.text_input("Department/Team", value="")
                must_have = st.text_area("Must-have Skills", placeholder="Python, SQL, Communication")
                nice_to_have = st.text_area("Nice-to-have Skills", placeholder="DBT, Airflow")
                min_experience = st.number_input("Minimum Years Experience", min_value=0, max_value=40, value=0)
            with colB:
                work_auth = st.text_input("Work Authorization Requirements", value="")
                target_start = st.text_input("Target Start Date / Timeline", value="")
                urgency = st.selectbox("Urgency/Priority", ["normal", "high", "critical"], index=0)
                min_qualification = st.text_input("Minimum Qualification", value="")
            colC, colD = st.columns(2)
            with colC:
                ideal_background = st.text_area("Ideal Background", placeholder="Industries, companies, education, certifications")
                culture_values = st.text_area("Cultural Fit / Values", placeholder="Team-first, entrepreneurial, detail-oriented")
                deal_breakers = st.text_area("Deal-breakers", placeholder="No relocation, must have cloud experience, etc.")
            with colD:
                process_stages = st.text_area("Hiring Stages", placeholder="Screening, Technical, Manager, Final")
                decision_makers = st.text_area("Decision-Makers / Interviewers", placeholder="Hiring Manager, Tech Lead, HR")
                evaluation_focus = st.text_area("Preferred Evaluation Criteria", placeholder="Technical ability, leadership, problem-solving")
            colE, colF = st.columns(2)
            with colE:
                company_overview = st.text_area("Company Overview", placeholder="Size, stage, industry")
                usp = st.text_area("Unique Selling Points", placeholder="Mission, impact, perks, culture")
            with colF:
                di_goals = st.text_area("Diversity & Inclusion Goals", placeholder="e.g., broaden candidate pools, structured rubrics")

        skip_wizard = st.checkbox("Skip wizard and use defaults", value=False)

        col_submit, col_skip = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("Generate Hiring Plan", type="primary")
        with col_skip:
            if st.form_submit_button("Use Defaults"):
                skip_wizard = True
                submitted = True

        if submitted:
            overrides: Dict[str, Any] = {}
            if not skip_wizard:
                selected_salary = (custom_salary.strip() or preset_choice or "competitive").strip()
                overrides = {
                    "timeline_weeks": timeline_weeks,
                    # New canonical field consumed by backend for comp
                    "salary_range": selected_salary,
                    # Mirror to legacy budget_range to keep extraction/session views consistent
                    "budget_range": selected_salary,
                    "roles": [r.strip() for r in ([role_title] + additional_roles.split(",")) if r and r.strip()],
                    "seniority": seniority,
                    "employment_type": employment_type,
                    "location": location,
                    "key_skills": [s.strip() for s in key_skills.split(",") if s.strip()],
                    "company_stage": company_stage or "startup",
                    "plan_style": st.session_state.get("plan_style", "Startup"),
                    # Advanced
                    "department": department,
                    "must_have_skills": [s.strip() for s in must_have.split(",") if s.strip()],
                    "nice_to_have_skills": [s.strip() for s in nice_to_have.split(",") if s.strip()],
                    "min_experience": min_experience,
                    "work_authorization": work_auth,
                    "target_start": target_start,
                    "urgency": urgency,
                    "min_qualification": min_qualification,
                    "ideal_background": ideal_background,
                    "culture_values": culture_values,
                    "deal_breakers": [s.strip() for s in deal_breakers.split(",") if s.strip()],
                    "process_stages": [s.strip() for s in process_stages.split(",") if s.strip()],
                    "decision_makers": [s.strip() for s in decision_makers.split(",") if s.strip()],
                    "evaluation_focus": [s.strip() for s in evaluation_focus.split(",") if s.strip()],
                    "company_overview": company_overview,
                    "unique_selling_points": [s.strip() for s in usp.split(",") if s.strip()],
                    "diversity_inclusion_goals": di_goals,
                }
                st.session_state["last_overrides"] = overrides

            # Close wizard BEFORE rerun happens in process_user_input
            st.session_state.wizard_mode = False
            st.session_state.pop("hiring_intent", None)

            process_user_input(initial_input, overrides)


# ------------------------------ Processing / Tools / Results ------------------------------

def process_user_input(user_input: str, overrides: Optional[Dict[str, Any]] = None):
    try:
        # If we're in chat mode (wizard not active) and user supplies constraints, merge into overrides
        if not overrides:
            overrides = {}
        # Lightweight extraction of common constraints from free text via existing helpers
        # Salary via LLM extractor
        sal_single = _extract_salary_value(user_input)
        if sal_single:
            overrides["salary_range"] = sal_single
        # Seniority
        lower = user_input.lower()
        for sen in ["junior", "mid", "senior", "lead", "principal"]:
            if sen in lower:
                overrides["seniority"] = sen
                break
        # Employment type
        for et in ["full-time", "part-time", "contract", "intern"]:
            if et in lower:
                overrides["employment_type"] = et
                break
        # Location heuristic (keep simple; user can refine)
        if "remote" in lower:
            overrides["location"] = "remote"
        # Merge any last_overrides carried from clarifications
        if st.session_state.get("last_overrides"):
            overrides = {**st.session_state.get("last_overrides", {}), **overrides}

        if overrides:
            overrides = validate_overrides(overrides)

        with st.spinner("Generating hiring plan..."):
            result = run_agent_langgraph(user_input, overrides)

        # Apply local refinements (years/add/remove/salary) and sync JSON
        _apply_local_refinements(result, overrides or {})

        st.session_state.last_output = result

        # Organized chat summary via LLM (fallback to deterministic on failure)
        try:
            roles = [jd.title for jd in (result.jd_drafts or [])]
            if not roles and result.raw_json.get("extraction"):
                roles = result.raw_json.get("extraction", {}).get("roles", [])
            timeline = None
            if result.hiring_plan:
                timeline = result.hiring_plan.timeline_weeks
            elif result.raw_json.get("extraction"):
                timeline = result.raw_json.get("extraction", {}).get("timeline_weeks", None)

            main_jd = result.jd_drafts[0].model_dump() if (result.jd_drafts and hasattr(result.jd_drafts[0], 'model_dump')) else (
                result.raw_json.get("job_descriptions", [{}])[0] if result.raw_json.get("job_descriptions") else {}
            )
            kit = result.raw_json.get("interview_kit") if result.raw_json else None
            from app.prompts import CHAT_SUMMARY_PROMPT
            prompt = CHAT_SUMMARY_PROMPT.format(
                roles=", ".join(roles) if roles else "the role",
                timeline_weeks=timeline or 8,
                primary_jd=json.dumps(main_jd, ensure_ascii=False),
                interview_kit=json.dumps(kit or {}, ensure_ascii=False)
            )
            msg = ask_llm(prompt)
            if msg and isinstance(msg, str) and len(msg.strip()) > 0:
                st.session_state.chat_history.append({"role": "assistant", "content": msg.strip()})
            else:
                raise ValueError("Empty summary from LLM")
        except Exception:
            try:
                roles = [jd.title for jd in (result.jd_drafts or [])]
                timeline = result.hiring_plan.timeline_weeks if result.hiring_plan else None
                main_jd = result.jd_drafts[0] if result.jd_drafts else None
                jd_title = main_jd.title if main_jd else (roles[0] if roles else "the role")
                jd_summary = main_jd.summary if main_jd else ""
                top_resps = (main_jd.responsibilities[:3] if main_jd and main_jd.responsibilities else [])
                top_reqs = (main_jd.requirements[:3] if main_jd and main_jd.requirements else [])
                comp = main_jd.compensation_note if main_jd else ""
                parts = []
                header = "Here’s your organized hiring summary"
                if roles:
                    header += f" for {', '.join(roles)}"
                if timeline:
                    header += f" (timeline: {timeline} weeks)"
                parts.append(header + ":")
                if jd_title:
                    parts.append(f"\nJD: {jd_title}")
                if jd_summary:
                    parts.append(f"Summary: {jd_summary}")
                if top_resps:
                    parts.append("Top Responsibilities:")
                    parts.extend([f"- {r}" for r in top_resps])
                if top_reqs:
                    parts.append("Top Requirements:")
                    parts.extend([f"- {rq}" for rq in top_reqs])
                if comp:
                    parts.append(f"Compensation: {comp}")
                parts.append("\nSee the Interview Kit on the right for full details.")
                chat_msg = "\n".join(parts)
                st.session_state.chat_history.append({"role": "assistant", "content": chat_msg})
            except Exception:
                if result.raw_markdown:
                    st.session_state.chat_history.append({"role": "assistant", "content": result.raw_markdown})

        # Save session state (only when meaningful results)
        extraction = None
        if overrides and (result.jd_drafts or result.hiring_plan):
            extraction = SlotExtraction(
                timeline_weeks=overrides.get("timeline_weeks", 8),
                budget_range=overrides.get("salary_range") or overrides.get("budget_range", "competitive"),
                seniority=overrides.get("seniority", "mid"),
                employment_type=overrides.get("employment_type", "full-time"),
                location=overrides.get("location", "remote"),
                key_skills=overrides.get("key_skills", []),
                company_stage=overrides.get("company_stage", "startup"),
                roles=[jd.title for jd in result.jd_drafts] if result.jd_drafts else [],
            )

        session_state = SessionState(
            session_id=st.session_state.session_id,
            last_input=user_input,
            last_output=result,
            extraction=extraction,
            timestamp=datetime.now().isoformat(),
        )
        save_state(st.session_state.session_id, session_state)

        st.rerun()  # Single rerun from here
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
        })


def render_tools_panel():
    st.subheader("🛠️ Tools")

    # Determine selected role/result for tools
    selected_result = st.session_state.get("last_output")
    if st.session_state.get("role_outputs"):
        roles = list(st.session_state.role_outputs.keys())
        # Default to active_role if set, else first role
        if "active_role" not in st.session_state or st.session_state.active_role not in roles:
            st.session_state.active_role = roles[0]
        st.write("**Context**")
        st.session_state.active_role = st.selectbox(
            "Role for tools",
            roles,
            index=roles.index(st.session_state.active_role),
            key="tools_role_select",
        )
        selected_result = st.session_state.role_outputs.get(st.session_state.active_role)

    if not selected_result:
        st.info("Generate a hiring plan first to use tools")
        return

    # Search simulation
    st.write("**Search Simulator**")
    search_query = st.text_input("Search query", placeholder="data analyst interview questions")
    if st.button("🔍 Simulate Search"):
        if search_query:
            results = simulate_search(search_query)
            st.write("**Search Results:**")
            for i, r in enumerate(results, 1):
                title = r.get('title') or r.get('url') or 'Result'
                url = r.get('url') or ''
                snippet = r.get('snippet') or ''
                st.markdown(f"{i}. [**{title}**]({url})")
                if snippet:
                    st.write(f"   {snippet}")
        else:
            st.warning("Please enter a search query")

    st.divider()

    # Google Search URL builder (based on selected role context)
    st.write("**Google Search**")
    default_terms = []
    if selected_result and selected_result.jd_drafts:
        default_terms.append(selected_result.jd_drafts[0].title)
        default_terms.extend((st.session_state.get("last_overrides", {}).get("key_skills") or [])[:3])
        loc = st.session_state.get("last_overrides", {}).get("location")
        if loc:
            default_terms.append(loc)
    google_default = " ".join(default_terms) or "hiring plan interview questions"
    g_query = st.text_input("Google query", value=google_default, key="google_query_input")
    if st.button("🔗 Build Google Search URL"):
        from app.tools import google_search_url
        url = google_search_url(g_query)
        st.code(url)
        colg1, colg2 = st.columns(2)
        with colg1:
            st.markdown(f"[Open in Google]({url})")
        with colg2:
            if st.button("Open in new tab", key=f"open_google_newtab_{st.session_state.get('active_role','single')}"):
                components.html(f"<script>window.open('{url}', '_blank')</script>", height=0)

    st.divider()

    # Job Board Posting Shortcuts
    st.write("**Job Board Posting Shortcuts**")
    if not selected_result or not (selected_result.jd_drafts and len(selected_result.jd_drafts) > 0):
        st.info("Generate a JD first to access posting shortcuts")
    else:
        jd0 = selected_result.jd_drafts[0]
        # Build ATS-friendly plain text JD
        title = jd0.title or "Job Title"
        summary = jd0.summary or ""
        responsibilities = jd0.responsibilities or []
        requirements = jd0.requirements or []
        comp = jd0.compensation_note or "Competitive"
        lines = []
        lines.append(title)
        if summary:
            lines.append("")
            lines.append(summary)
        if responsibilities:
            lines.append("")
            lines.append("Responsibilities:")
            for r in responsibilities:
                lines.append(f"- {r}")
        if requirements:
            lines.append("")
            lines.append("Requirements:")
            for rq in requirements:
                lines.append(f"- {rq}")
        if comp:
            lines.append("")
            lines.append(f"Compensation: {comp}")
        ats_text = "\n".join(lines)
        # Minimal HTML version
        html_parts = [
            f"<h1>{title}</h1>",
            f"<p>{summary}</p>" if summary else "",
        ]
        if responsibilities:
            html_parts.append("<h3>Responsibilities</h3><ul>" + "".join([f"<li>{r}</li>" for r in responsibilities]) + "</ul>")
        if requirements:
            html_parts.append("<h3>Requirements</h3><ul>" + "".join([f"<li>{rq}</li>" for rq in requirements]) + "</ul>")
        if comp:
            html_parts.append(f"<p><strong>Compensation:</strong> {comp}</p>")
        ats_html = "\n".join([p for p in html_parts if p])

        st.text_area("Copy-ready JD (plain text)", ats_text, height=260, key="ats_copy_ready")
        col_dl1, col_dl2 = st.columns(2)
        # Safe filename
        safe_name = re.sub(r"[^a-z0-9]+", "-", (title or "jd").lower()).strip("-") or "jd"
        with col_dl1:
            st.download_button(
                label="⬇️ Download JD (.txt)",
                data=ats_text,
                file_name=f"{safe_name}.txt",
                mime="text/plain",
                key="dl_jd_txt",
            )
        with col_dl2:
            st.download_button(
                label="⬇️ Download JD (.html)",
                data=ats_html,
                file_name=f"{safe_name}.html",
                mime="text/html",
                key="dl_jd_html",
            )

        st.write("")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            li_url = "https://www.linkedin.com/talent/post-a-job"
            st.markdown(f"[Open LinkedIn Job Posting]({li_url})")
            if st.button("Open LinkedIn in new tab", key="open_li_post"):
                components.html(f"<script>window.open('{li_url}', '_blank')</script>", height=0)
        with col_p2:
            indeed_url = "https://employers.indeed.com/"
            st.markdown(f"[Open Indeed Employer]({indeed_url})")
            if st.button("Open Indeed in new tab", key="open_indeed_post"):
                components.html(f"<script>window.open('{indeed_url}', '_blank')</script>", height=0)

    # Email writer (context-aware)
    st.write("**Email Writer**")
    email_to = st.text_input("To", placeholder="candidate@example.com")
    email_subject = st.text_input("Subject", placeholder="Exciting opportunity at our startup")
    email_outline = st.text_area("Body outline", placeholder="Brief reason for outreach, what you liked in profile, call to action. For JD-based emails, try: 'compose a mail to candidate based on this jd'.")
    if st.button("✉️ Draft Email"):
        if email_to and email_subject and email_outline:
            # Build context from last_output and last_overrides
            ctx = {}
            lo = selected_result
            ov = st.session_state.get("last_overrides", {})
            if lo and lo.jd_drafts:
                jd0 = lo.jd_drafts[0]
                ctx["role_title"] = jd0.title
                ctx["jd_summary"] = jd0.summary
                ctx["jd_responsibilities"] = jd0.responsibilities
                ctx["jd_compensation_note"] = jd0.compensation_note
            ctx.update({
                "seniority": ov.get("seniority"),
                "location": ov.get("location"),
                "company_stage": ov.get("company_stage"),
                "salary_range": ov.get("salary_range") or ov.get("budget_range"),
                "key_skills": ov.get("must_have_skills") or ov.get("key_skills"),
                "unique_selling_points": ov.get("unique_selling_points"),
                "timeline_weeks": ov.get("timeline_weeks"),
                "process_stages": ov.get("process_stages"),
                "company_name": os.getenv("COMPANY_NAME", "our team"),
                "recruiter_name": os.getenv("RECRUITER_NAME", "[Your Name]"),
            })
            # Prefer interview kit outreach if present
            if lo and lo.raw_json and lo.raw_json.get("interview_kit"):
                kit = lo.raw_json["interview_kit"]
                ctx["outreach_subject"] = kit.get("outreach_subject")
                ctx["outreach_body"] = kit.get("outreach_body")
                if ctx.get("outreach_subject") and not email_subject:
                    email_subject = ctx["outreach_subject"]

            draft = email_writer(email_to, email_subject, email_outline, context=ctx)
            st.text_area("Email Draft", draft, height=260)
        else:
            st.warning("Please fill in all email fields")

    st.divider()

    # LinkedIn candidate search
    st.divider()
    st.write("**LinkedIn Candidate Search**")
    lk_title = st.text_input("Role title for search", value=(selected_result.jd_drafts[0].title if selected_result and selected_result.jd_drafts else ""), placeholder="Software Engineer")
    lk_location = st.text_input("Location filter (optional)", value=(st.session_state.get("last_overrides", {}).get("location", "")), placeholder="Remote or City")
    # Prefer JD-derived skills; fallback to wizard
    jd_skills = []
    if selected_result:
        try:
            # Prefer JD JSON
            jd0 = None
            if selected_result.raw_json and selected_result.raw_json.get("job_descriptions"):
                jd0 = selected_result.raw_json["job_descriptions"][0]
            elif selected_result.jd_drafts and hasattr(selected_result.jd_drafts[0], 'model_dump'):
                jd0 = selected_result.jd_drafts[0].model_dump()
            if jd0:
                from app.prompts import JD_SKILLS_EXTRACT_PROMPT
                prompt = JD_SKILLS_EXTRACT_PROMPT.format(jd_json=json.dumps(jd0, ensure_ascii=False))
                resp = ask_llm(prompt)
                cleaned = resp.replace("“", '"').replace("”", '"').replace("’", "'")
                start = cleaned.find('{')
                top = []
                if start != -1:
                    depth = 0
                    for i in range(start, len(cleaned)):
                        if cleaned[i] == '{':
                            depth += 1
                        elif cleaned[i] == '}':
                            depth -= 1
                            if depth == 0:
                                obj = json.loads(cleaned[start:i+1])
                                top = obj.get("top_skills") or []
                                break
                jd_skills = top[:5]
        except Exception:
            # Silent fallback to empty; UI will use overrides/top_skills next
            jd_skills = []
    ov = st.session_state.get("last_overrides", {})
    top_skills = (ov.get("must_have_skills") or ov.get("key_skills") or [])[:5]
    lk_skills_default = ", ".join((jd_skills[:5] or top_skills))
    lk_skills = st.text_input("Skills (comma-separated)", value=lk_skills_default)
    if st.button("🔎 Build LinkedIn Search"):
        from app.tools import linkedin_candidate_search
        skills_list = [s.strip() for s in lk_skills.split(",") if s.strip()]
        res = linkedin_candidate_search(skills_list, lk_title or (selected_result.jd_drafts[0].title if selected_result and selected_result.jd_drafts else ""), lk_location, st.session_state.get("last_overrides", {}).get("seniority", ""))
        st.write("**Boolean Query:**")
        st.code(res["boolean_query"] or "(empty)")
        st.write("**LinkedIn Search URL:**")
        st.code(res["search_url"]) 
        st.write("**Sample Results:**")
        for r in res["results"]:
            st.write(f"- {r['name']} — {r['headline']} ({r['location']})")
            st.write(f"  {r['profile_url']}")

    st.divider()
    # Exports
    st.write("**Exports**")
    if st.button("📋 Export Checklist"):
        lo = selected_result
        if lo and lo.hiring_plan:
            plan_dict = lo.hiring_plan.model_dump()
            checklist_md = checklist_export(plan_dict)
            st.text_area("Hiring Checklist", checklist_md, height=300)
            st.download_button(
                label="📥 Download Checklist",
                data=checklist_md,
                file_name=f"hiring_checklist_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )
        else:
            st.warning("No hiring plan available to export")

    if st.button("📝 Download Full Report (Markdown)"):
        lo = selected_result
        if lo and lo.raw_markdown:
            st.download_button(
                label="Download Now",
                data=lo.raw_markdown,
                file_name=f"hiring_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )
        else:
            st.warning("No report available yet")

    st.divider()
    # ATS Resume Checker
    st.write("**ATS Resume Checker**")
    resume = st.text_area("Paste candidate resume (text)", height=200)
    if st.button("✅ Check Against Current JD"):
        lo = selected_result
        if not lo or not lo.raw_json.get("job_descriptions"):
            st.warning("Generate a JD first")
        elif not resume.strip():
            st.warning("Paste a resume to check")
        else:
            from app.tools import ats_resume_check
            jd0 = lo.raw_json.get("job_descriptions")[0]
            res = ats_resume_check(jd0, resume)
            st.metric("ATS Score", f"{res['score']} / 100")
            st.write("**Matched skills (top):**")
            st.write(", ".join(res.get("matched_skills", [])) or "None")
            st.write("**Missing skills (top):**")
            st.write(", ".join(res.get("missing_skills", [])) or "None")
            if res.get("notes"):
                st.write("**Notes:**")
                for n in res["notes"]:
                    st.write(f"- {n}")


def render_results_section():
    st.subheader("📊 Results")
    # If multiple role results exist, show per-role tabs (use cleaned names)
    if st.session_state.get("role_outputs"):
        roles = list(st.session_state.role_outputs.keys())
        rtabs = st.tabs(roles)
        for i, role in enumerate(roles):
            with rtabs[i]:
                result = st.session_state.role_outputs[role]
                sub1, sub2, sub3 = st.tabs(["📱 Cards", "📄 Markdown", "🔧 JSON"])
                with sub1:
                    render_cards_view(result, key_ns=f"role_{i}")
                with sub2:
                    if result.raw_markdown:
                        st.markdown(result.raw_markdown)
                    else:
                        st.info("No markdown output available")
                with sub3:
                    if result.raw_json:
                        st.json(result.raw_json)
                    else:
                        st.info("No JSON output available")
        return
    # Single result mode
    result = st.session_state.last_output
    tab1, tab2, tab3, tab4 = st.tabs(["📱 Cards", "📄 Raw Markdown", "🔧 Raw JSON", "✍️ Refine"])
    with tab1:
        render_cards_view(result, key_ns="single")
    with tab2:
        if result.raw_markdown:
            st.markdown(result.raw_markdown)
        else:
            st.info("No markdown output available")
    with tab3:
        if result.raw_json:
            st.json(result.raw_json)
        else:
            st.info("No JSON output available")
    with tab4:
        tweak = st.text_area("Refinement prompt", placeholder="e.g., Add cloud skills to the JD; shorten responsibilities; make tone more formal.")
        if st.button("Apply Refinement"):
            ov = st.session_state.get("last_overrides", {}).copy()
            ov["tweak"] = tweak
            process_user_input(st.session_state.get("hiring_intent") or "Refine hiring plan", ov)

def process_role_input(user_input: str, overrides: Optional[Dict[str, Any]], role_key: str):
    try:
        if overrides:
            overrides = validate_overrides(overrides)
        with st.spinner(f"Generating plan for {role_key}..."):
            result = run_agent_langgraph(user_input, overrides)
        # store per-role without overwriting others
        if "role_outputs" not in st.session_state:
            st.session_state.role_outputs = {}
        # Clean display key to remove leading articles or helper words
        role_key_clean = _clean_role_title(role_key)
        st.session_state.role_outputs[role_key_clean] = result
        st.session_state.last_output = result  # keep latest for tools
        # minimal analytics save
        session_state = SessionState(
            session_id=st.session_state.session_id,
            last_input=user_input,
            last_output=result,
            extraction=None,
            timestamp=datetime.now().isoformat(),
        )
        save_state(st.session_state.session_id, session_state)
        st.rerun()
    except Exception as e:
        st.error(f"Error generating for {role_key}: {str(e)}")


def render_cards_view(result, key_ns: str = "default"):
    if result.jd_drafts:
        st.write("### 💼 Job Descriptions")
        for i, jd in enumerate(result.jd_drafts):
            with st.expander(f"📋 {jd.title}", expanded=i == 0):
                # Badge by role family
                title_lower = jd.title.lower()
                badge = ""
                if "engineer" in title_lower:
                    badge = "🔧 Engineer"
                elif "analyst" in title_lower:
                    badge = "📊 Analyst"
                elif "manager" in title_lower:
                    badge = "🧭 Manager"
                if badge:
                    st.caption(badge)
                if st.session_state.get("last_refined_at"):
                    ts = st.session_state.get("last_refined_at")
                    st.caption(f"Refined • {ts[:16]}")
                st.write(f"**Summary:** {jd.summary}")
                st.write("**Key Responsibilities:**")
                for r in jd.responsibilities:
                    if st.session_state.get("plan_view") == "Concise":
                        st.write(f"• {r.split(' – ')[0].split(':')[0]}")
                    else:
                        st.write(f"• {r}")
                st.write("**Requirements:**")
                for rq in jd.requirements:
                    if st.session_state.get("plan_view") == "Concise":
                        st.write(f"• {rq.split(' – ')[0].split(':')[0]}")
                    else:
                        st.write(f"• {rq}")
                st.write(f"**Compensation:** {jd.compensation_note}")

    if result.hiring_plan:
        st.write("### 📅 Hiring Plan")
        with st.expander(f"📈 {result.hiring_plan.timeline_weeks}-Week Process", expanded=True):
            st.write(f"**Timeline:** {result.hiring_plan.timeline_weeks} weeks")
            st.write("**Process Steps:**")
            for i, (step, owner) in enumerate(zip(result.hiring_plan.steps, result.hiring_plan.owners), 1):
                st.write(f"{i}. {step}")
                st.caption(f"   *Owner: {owner}*")

    # Interview kit (if available in raw_json)
    if result.raw_json and result.raw_json.get("interview_kit"):
        kit = result.raw_json["interview_kit"]
        st.write("### 🧰 Interview Kit")
        # Quick copy block of top 5 questions
        if kit.get("screening_questions"):
            top5 = kit.get("screening_questions", [])[:5]
            st.text_area(
                "Copy top screening questions",
                "\n".join([f"- {q}" for q in top5]),
                height=120,
                key=f"copy_screen_top5_{key_ns}"
            )
        if kit.get("screening_guide"):
            with st.expander("🧪 Initial Screening Guide", expanded=True):
                for pair in kit.get("screening_guide", []):
                    q = pair.get("question")
                    w = pair.get("what_to_evaluate")
                    if q:
                        st.markdown(f"- **Q:** {q}")
                        if w:
                            st.caption(f"   Evaluate: {w}")
        if kit.get("screening_questions"):
            with st.expander("📄 Screening Questions", expanded=True):
                for q in kit.get("screening_questions", []):
                    st.write(f"- {q}")
        if kit.get("evaluation_rubric"):
            with st.expander("✅ Evaluation Rubric", expanded=False):
                for r in kit.get("evaluation_rubric", []):
                    st.write(f"- {r}")
        if kit.get("scorecard"):
            with st.expander("🗒️ Scorecard Fields", expanded=False):
                for s in kit.get("scorecard", []):
                    st.write(f"- {s}")
        if kit.get("outreach_subject") or kit.get("outreach_body"):
            with st.expander("✉️ Outreach Template", expanded=False):
                if kit.get("outreach_subject"):
                    st.write(f"**Subject:** {kit['outreach_subject']}")
                if kit.get("outreach_body"):
                    st.text_area("Email Body", kit["outreach_body"], height=180)


# ------------------------------ Entrypoint ------------------------------

if __name__ == "__main__":
    main()
