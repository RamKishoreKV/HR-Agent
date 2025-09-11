"""LangGraph workflow for the HR Hiring Copilot.

This module implements a resilient LangGraph pipeline with:
- Consistent dict-style state handling
- Robust JSON parsing for LLM outputs (works better with local models)
- Role canonicalization and default skill inference for common roles
- Deterministic final markdown assembly (no LLM hallucinations for salary)
"""

import json
import re
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from .llm import ask_llm
from .prompts import (
    SLOT_EXTRACT_PROMPT,
    JD_GENERATION_PROMPT,
    PLAN_JSON_PROMPT,
    INTERVIEW_KIT_PROMPT,
    CLARIFY_ROLE_PROMPT,
    ROLE_SKILL_NORMALIZE_PROMPT,
)
from .comp import compose_comp_note, get_default_budget_for_role
from .schema import SlotExtraction, JobDescription, HiringPlan, AgentOutput, InterviewKit
from pydantic import ValidationError


class HiringGraphState(TypedDict):
    user_input: str
    clarifications_asked: List[str]
    extraction: Optional[SlotExtraction]
    job_descriptions: List[JobDescription]
    hiring_plan: Optional[HiringPlan]
    interview_kit: Optional[InterviewKit]
    raw_markdown: str
    raw_json: Dict[str, Any]
    overrides: Dict[str, Any]
    needs_clarification: bool


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    fence_matches = re.findall(r"```(?:json|JSON)?\n([\s\S]*?)\n```", text)
    if fence_matches:
        return fence_matches[0].strip()
    return text


def _loose_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i+1]
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _canonicalize_role(role: str) -> str:
    original = role or ""
    role_lower = " ".join(original.strip().lower().split())
    if role_lower.startswith("a "):
        role_lower = role_lower[2:]
    if role_lower.startswith("an "):
        role_lower = role_lower[3:]
    if role_lower.startswith("the "):
        role_lower = role_lower[4:]
    if role_lower.startswith("n "):
        role_lower = role_lower[2:]
    role_lower = role_lower.replace(" engineer engineer", " engineer")
    if "gen ai" in role_lower or "genai" in role_lower:
        return "Generative AI Engineer"
    if "machine learning engineer" in role_lower or role_lower == "ml engineer":
        return "Machine Learning Engineer"
    if role_lower in ["i engineer", "iengineer"]:
        return "AI Engineer"
    if "ai engineer" in role_lower:
        return "AI Engineer"
    if "software engineer" in role_lower:
        return "Software Engineer"
    if "data analyst" in role_lower:
        return "Data Analyst"
    if "data engineer" in role_lower:
        return "Data Engineer"
    if "business analyst" in role_lower:
        return "Business Analyst"
    if "financial analyst" in role_lower:
        return "Financial Analyst"
    if "database developer" in role_lower:
        return "Database Developer"
    if "embedded ai engineer" in role_lower:
        return "Embedded AI Engineer"
    return " ".join(word.capitalize() for word in role_lower.split())


ROLE_DEFAULT_SKILLS: Dict[str, List[str]] = {
    "Generative AI Engineer": [
        "Python", "LLMs", "LangChain", "Transformers", "Vector Databases",
        "Prompt Engineering", "Embeddings", "Model Serving"
    ],
    "AI Engineer": [
        "Python", "PyTorch", "TensorFlow", "ML Ops", "Docker", "Kubernetes"
    ],
    "Machine Learning Engineer": [
        "Python", "Scikit-learn", "PyTorch", "Feature Engineering", "ML Pipelines"
    ],
    "Data Engineer": [
        "Python", "SQL", "ETL", "Airflow", "Spark", "AWS", "Docker"
    ],
    "Data Analyst": [
        "SQL", "Python", "Dashboards", "BI Tools", "A/B Testing"
    ],
    "Software Engineer": [
        "Python/Java/JS", "APIs", "Databases", "Docker", "CI/CD"
    ],
    "Business Analyst": [
        "Requirements Gathering", "SQL", "Dashboards", "Stakeholder Management", "Process Mapping"
    ],
    "Financial Analyst": [
        "Financial Modeling", "Excel", "Forecasting", "FP&A", "SQL"
    ],
    "Database Developer": [
        "SQL", "Database Design", "Normalization", "Indexes", "Performance Tuning"
    ],
    "Embedded AI Engineer": [
        "C/C++", "ARM", "Edge AI", "TensorRT", "RTOS"
    ],
}


def _infer_roles_from_text(text: str) -> List[str]:
    if not text:
        return []
    text_l = text.lower()
    m = re.search(r"hire\s+(?:a|an|the)?\s*([a-z0-9 \-/]+)", text_l)
    role = None
    if m:
        role = m.group(1)
        role = re.sub(r'^(me|us|our|my)\s+', '', role).strip()
        role = re.split(r"\bfor\b|\bin\b|\bwithin\b|\bby\b|\bon\b|\bwith\b|,|\.|!|\?", role)[0]
    else:
        keywords = ["engineer", "analyst", "manager", "designer", "scientist", "developer"]
        for kw in keywords:
            if kw in text_l:
                idx = text_l.find(kw)
                snippet = text_l[max(0, idx-25): idx+len(kw)]
                tokens = snippet.strip().split()
                tail = [tokens[-3], tokens[-2], tokens[-1]] if len(tokens) >= 3 else tokens
                role = " ".join(tail)
                break
    if not role:
        return []
    role = " ".join(role.split())
    canonical = _canonicalize_role(role)
    return [canonical]


def clarify_node(state: HiringGraphState) -> HiringGraphState:
    try:
        prompt = CLARIFY_ROLE_PROMPT.format(user_input=state["user_input"])
        resp = ask_llm(prompt)
        data = _loose_json_extract(resp) or {}
        need = bool(data.get("need_clarification", False))
        question = str(data.get("question", "")).strip()
        options = data.get("options") or []
        suggested_roles = data.get("suggested_roles") or []
        if need:
            msg = question if question else "Could you clarify the specific role?"
            if options:
                msg += " Options: " + ", ".join(options)
            state["clarifications_asked"].append(msg)
            state["needs_clarification"] = True
        if (not need) and suggested_roles:
          
            if not state.get("overrides"):
                state["overrides"] = {}
            if not state["overrides"].get("roles"):
                state["overrides"]["roles"] = suggested_roles
    except Exception:
        if not state["user_input"] or len(state["user_input"].strip()) < 10:
            state["clarifications_asked"].append("Could you provide more details about the role you'd like to hire for?")
            state["needs_clarification"] = True
    return state


def extract_node(state: HiringGraphState) -> HiringGraphState:
    try:
        if state["overrides"]:
            ov = state["overrides"]
            extraction_data = {
                "company_stage": (ov.get("company_stage") or "startup"),
                "timeline_weeks": int(ov.get("timeline_weeks") or 8),
                "budget_range": (ov.get("salary_range") or ov.get("budget_range") or "competitive"),
                "seniority": (ov.get("seniority") or "mid"),
                "employment_type": (ov.get("employment_type") or "full-time"),
                "location": (ov.get("location") or "remote"),
                "key_skills": ov.get("key_skills") or [],
                "roles": ov.get("roles") or []
            }
          
            try:
                prompt = ROLE_SKILL_NORMALIZE_PROMPT.format(
                    user_input=state["user_input"],
                    candidate_roles=", ".join(extraction_data.get("roles") or []),
                    candidate_skills=", ".join(extraction_data.get("key_skills") or []),
                )
                resp = ask_llm(prompt)
                norm = _loose_json_extract(resp) or {}
                nr = norm.get("roles") or []
                ns = norm.get("key_skills") or []
                if (not extraction_data.get("roles")) and nr:
                    extraction_data["roles"] = nr
                if ns:
                    extraction_data["key_skills"] = ns
            except Exception:
                pass
            if not extraction_data["roles"]:
                inferred = _infer_roles_from_text(state["user_input"]) or ["Data Analyst"]
                extraction_data["roles"] = inferred
                if not extraction_data["key_skills"]:
                    ks = ROLE_DEFAULT_SKILLS.get(inferred[0])
                    if ks:
                        extraction_data["key_skills"] = ks
        else:
            prompt = SLOT_EXTRACT_PROMPT.format(user_input=state["user_input"])
            response = ask_llm(prompt)
            extraction_data = _loose_json_extract(response) or {}
            if not extraction_data.get("roles"):
                inferred = _infer_roles_from_text(state["user_input"]) or ["Data Analyst"]
                extraction_data["roles"] = inferred
            try:
                prompt = ROLE_SKILL_NORMALIZE_PROMPT.format(
                    user_input=state["user_input"],
                    candidate_roles=", ".join(extraction_data.get("roles") or []),
                    candidate_skills=", ".join(extraction_data.get("key_skills") or []),
                )
                resp = ask_llm(prompt)
                norm = _loose_json_extract(resp) or {}
                nr = norm.get("roles") or []
                ns = norm.get("key_skills") or []
                if (not (state.get("overrides") and state["overrides"].get("roles"))) and nr:
                    if len(nr) > 1 and re.search(r"\b(a|an|the)\s+(designer|engineer|analyst|manager)\b", state["user_input"].lower()):
                        extraction_data["roles"] = [nr[0]]
                    else:
                        extraction_data["roles"] = nr
                if ns:
                    extraction_data["key_skills"] = ns
            except Exception:
                pass
            tm = re.search(r"(\d{1,2})\s*week", state["user_input"].lower())
            if tm and not extraction_data.get("timeline_weeks"):
                extraction_data["timeline_weeks"] = int(tm.group(1))
            defaults = {
                "company_stage": "startup",
                "timeline_weeks": 8,
                "budget_range": "competitive",
                "seniority": "mid",
                "employment_type": "full-time",
                "location": "remote",
                "key_skills": []
            }
            for k, dv in defaults.items():
                v = extraction_data.get(k)
                if v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, list) and len(v) == 0):
                    extraction_data[k] = dv
            try:
                extraction_data["timeline_weeks"] = int(extraction_data.get("timeline_weeks", 8))
            except Exception:
                extraction_data["timeline_weeks"] = 8
            if not isinstance(extraction_data.get("key_skills"), list):
                ks = extraction_data.get("key_skills")
                extraction_data["key_skills"] = [s.strip() for s in ks.split(",") if s.strip()] if isinstance(ks, str) else []

        defaults = {
            "company_stage": "startup",
            "timeline_weeks": 8,
            "budget_range": extraction_data.get("budget_range") or "competitive",
            "seniority": "mid",
            "employment_type": "full-time",
            "location": "remote",
            "key_skills": []
        }
        for k, dv in defaults.items():
            v = extraction_data.get(k)
            if v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, list) and len(v) == 0):
                extraction_data[k] = dv
        try:
            extraction_data["timeline_weeks"] = int(extraction_data.get("timeline_weeks", 8))
        except Exception:
            extraction_data["timeline_weeks"] = 8
        if not isinstance(extraction_data.get("key_skills"), list):
            ks = extraction_data.get("key_skills")
            extraction_data["key_skills"] = [s.strip() for s in ks.split(",") if s.strip()] if isinstance(ks, str) else []

        roles = extraction_data.get("roles") or ["Data Analyst"]
        roles = [_canonicalize_role(r) for r in roles]
        extraction_data["roles"] = roles
        if (not extraction_data.get("key_skills")) and roles:
            default_skills = ROLE_DEFAULT_SKILLS.get(roles[0])
            if default_skills:
                extraction_data["key_skills"] = default_skills

        try:
            state["extraction"] = SlotExtraction(**extraction_data)
        except ValidationError:
            extraction_data["seniority"] = extraction_data.get("seniority") or "mid"
            extraction_data["employment_type"] = extraction_data.get("employment_type") or "full-time"
            extraction_data["location"] = extraction_data.get("location") or "remote"
            extraction_data["budget_range"] = extraction_data.get("budget_range") or "competitive"
            extraction_data["company_stage"] = extraction_data.get("company_stage") or "startup"
            if not isinstance(extraction_data.get("key_skills"), list):
                extraction_data["key_skills"] = []
            try:
                extraction_data["timeline_weeks"] = int(extraction_data.get("timeline_weeks") or 8)
            except Exception:
                extraction_data["timeline_weeks"] = 8
            state["extraction"] = SlotExtraction(**extraction_data)

    except Exception as e:
        print(f"Error in extract_node: {e}")
        state["extraction"] = SlotExtraction(
            company_stage="startup",
            timeline_weeks=8,
            budget_range="competitive",
            seniority="mid",
            employment_type="full-time",
            location="remote",
            key_skills=["python", "sql"],
            roles=["data analyst"]
        )
    return state


def jd_node(state: HiringGraphState) -> HiringGraphState:
    if not state["extraction"]:
        return state

    job_descriptions = []
    ov = state.get("overrides", {}) or {}

 
    salary_override = str(ov.get("salary_range", "") or "").strip()

    min_exp = ov.get("min_experience", None)
    try:
        min_exp = int(min_exp) if min_exp is not None else None
    except Exception:
        min_exp = None
    min_qual = str(ov.get("min_qualification", "") or "").strip()

    for role in state["extraction"].roles:
        try:

            comp_budget = salary_override if salary_override else "competitive"
            comp_note = compose_comp_note(
                role=role,
                stage=state["extraction"].company_stage,
                hr_budget=comp_budget
            )
            if (not salary_override) and (str(comp_budget).lower() in ["competitive", "market rate", "tbd"]):
                default_budget = get_default_budget_for_role(
                    role=role,
                    seniority=state["extraction"].seniority,
                    location=state["extraction"].location
                )
                comp_note = compose_comp_note(role, state["extraction"].company_stage, default_budget)

            
            ov = state.get("overrides", {}) or {}
            dept = ov.get("department") or ""
            emp_type = state["extraction"].employment_type
            must_have_skills = ", ".join(ov.get("must_have_skills", []))
            nice_to_have_skills = ", ".join(ov.get("nice_to_have_skills", []))

            extra_key_skills = ", ".join([s for s in [must_have_skills, nice_to_have_skills] if s])
            combined_skills = ", ".join([s for s in [", ".join(state["extraction"].key_skills), extra_key_skills] if s])

            tweak = (ov.get("tweak") or "").strip()
            prompt = JD_GENERATION_PROMPT.format(
                company_stage=state["extraction"].company_stage,
                budget_range=comp_budget,  
                role_title=(role if not dept else f"{role} - {dept}"),
                seniority=state["extraction"].seniority,
                location=state["extraction"].location,
                key_skills=(combined_skills or ", ".join(state["extraction"].key_skills))
            ) + (f"\n\nAdditional context: Employment Type: {emp_type}." if emp_type else "") + (f"\n\nAdditional instructions: {tweak}" if tweak else "")
            response = ask_llm(prompt)

            try:
                jd_data = _loose_json_extract(response) or {}
                jd_data["title"] = _canonicalize_role(role)
                jd_data["compensation_note"] = comp_note

                responsibilities = jd_data.get("responsibilities") or []
                requirements = jd_data.get("requirements") or []
                if isinstance(responsibilities, str):
                    responsibilities = [r.strip("- • ") for r in responsibilities.split("\n") if r.strip()]
                if isinstance(requirements, str):
                    requirements = [r.strip("- • ") for r in requirements.split("\n") if r.strip()]

                injected_reqs = []
                if min_exp is not None and min_exp > 0:
                    injected_reqs.append(f"{min_exp}+ years of relevant experience")
                if min_qual:
                    injected_reqs.append(min_qual)
                existing_lower = {req.lower() for req in requirements}
                for ir in injected_reqs:
                    if ir.lower() not in existing_lower:
                        requirements.insert(0, ir)

                while len(responsibilities) < 5:
                    responsibilities.append("Contribute to fast-paced startup projects with measurable outcomes")
                while len(requirements) < 5:
                    requirements.append("Strong communication and collaboration skills in cross-functional teams")

                jd_data["responsibilities"] = responsibilities[:9]
                jd_data["requirements"] = requirements[:9]

                job_description = JobDescription(**jd_data)
                job_descriptions.append(job_description)

            except json.JSONDecodeError:
                fallback_reqs = [
                    f"3+ years of experience in {role.lower()} role",
                    "Strong analytical and problem-solving skills",
                    "Excellent communication and collaboration abilities",
                    "Experience in startup or fast-paced environment",
                    "Passion for learning and growth",
                ]
                if min_exp is not None and min_exp > 0:
                    fallback_reqs.insert(0, f"{min_exp}+ years of relevant experience")
                if min_qual:
                    fallback_reqs.insert(0, min_qual)

                fallback_jd = JobDescription(
                    title=role.title(),
                    summary=f"Join our startup as a {role} and make a direct impact on our growth.",
                    responsibilities=[
                        f"Lead {role.lower()} initiatives and projects",
                        "Collaborate with cross-functional teams",
                        "Drive key business metrics and outcomes",
                        "Contribute to company strategy and growth",
                        "Mentor team members and share knowledge"
                    ],
                    requirements=fallback_reqs[:9],
                    compensation_note=comp_note
                )
                job_descriptions.append(fallback_jd)

        except Exception as e:
            print(f"Error generating JD for {role}: {e}")
            continue

    state["job_descriptions"] = job_descriptions
    return state


def plan_node(state: HiringGraphState) -> HiringGraphState:
    if not state["extraction"]:
        return state

    try:
        style = (state.get("overrides", {}) or {}).get("plan_style", "Startup")
        prompt = PLAN_JSON_PROMPT.format(
            timeline_weeks=state["extraction"].timeline_weeks,
            roles=", ".join(state["extraction"].roles),
            company_stage=state["extraction"].company_stage
        ) + f"\n\nStyle: {style}. Adjust steps and tone for this style."
        response = ask_llm(prompt)

        try:
            plan_data = _loose_json_extract(response) or {}
            steps = plan_data.get("steps") or []
            owners = plan_data.get("owners") or []
            if isinstance(steps, str):
                steps = [s.strip("- • ") for s in steps.split("\n") if s.strip()]
            if isinstance(owners, str):
                owners = [o.strip("- • ") for o in owners.split("\n") if o.strip()]
            if len(owners) < len(steps):
                owners += ["Hiring Manager"] * (len(steps) - len(owners))
            elif len(owners) > len(steps):
                owners = owners[:len(steps)]
            timeline = state["extraction"].timeline_weeks
            state["hiring_plan"] = HiringPlan(steps=steps, owners=owners, timeline_weeks=timeline)

        except json.JSONDecodeError:
            fallback_steps = [
                "Post job descriptions on key job boards",
                "Source candidates through LinkedIn and referrals",
                "Conduct initial phone screenings",
                "Technical assessment and skills evaluation",
                "Team interviews and culture fit assessment",
                "Reference checks and background verification",
                "Final interviews with leadership",
                "Prepare and send offer letters",
                "Onboarding and first week planning"
            ]
            fallback_owners = [
                "HR Team", "Recruiting Team", "HR Team", "Hiring Manager",
                "Team Lead", "HR Team", "CEO", "HR Team", "Hiring Manager"
            ]
            state["hiring_plan"] = HiringPlan(
                steps=fallback_steps,
                owners=fallback_owners,
                timeline_weeks=state["extraction"].timeline_weeks
            )

    except Exception as e:
        print(f"Error in plan_node: {e}")
        state["hiring_plan"] = HiringPlan(
            steps=["Create job posting", "Source candidates", "Interview candidates", "Make offer"],
            owners=["HR", "Recruiter", "Hiring Manager", "HR"],
            timeline_weeks=state["extraction"].timeline_weeks if state["extraction"] else 8
        )
    return state


def interview_kit_node(state: HiringGraphState) -> HiringGraphState:
    if not state["extraction"]:
        return state

    try:
        prompt = INTERVIEW_KIT_PROMPT.format(
            roles=", ".join(state["extraction"].roles),
            seniority=state["extraction"].seniority,
            key_skills=", ".join(state["extraction"].key_skills),
            company_stage=state["extraction"].company_stage,
        )
        response = ask_llm(prompt)

        try:
            kit_data = _loose_json_extract(response) or {}
            if not isinstance(kit_data, dict):
                kit_data = {}
            sq = kit_data.get("screening_questions") or []
            sp = kit_data.get("screening_pairs") or []
            rb = kit_data.get("evaluation_rubric") or []
            sc = kit_data.get("scorecard") or []
            subj = kit_data.get("outreach_subject") or ""
            body = kit_data.get("outreach_body") or ""
            if isinstance(sq, str):
                sq = [q.strip("- • ") for q in sq.split("\n") if q.strip()]
            guide = []
            if isinstance(sp, list):
                for item in sp:
                    if isinstance(item, dict):
                        normalized = {}
                        for k, v in item.items():
                            nk = str(k).strip().strip('"').strip("'").lower().replace(" ", "_")
                            normalized[nk] = v
                        q = str(normalized.get("question", "") or normalized.get("q", "")).strip()
                        e = str(
                            normalized.get("what_to_evaluate", "")
                            or normalized.get("evaluate", "")
                            or normalized.get("evaluation", "")
                        ).strip()
                        if q:
                            guide.append({"question": q, "what_to_evaluate": e})
                    elif isinstance(item, str):
                        parts = item.split(":", 1)
                        q = parts[0].strip()
                        e = parts[1].strip() if len(parts) > 1 else ""
                        if q:
                            guide.append({"question": q, "what_to_evaluate": e})
            elif isinstance(sp, str):
                for line in sp.split("\n"):
                    line = line.strip("- • ").strip()
                    if not line:
                        continue
                    parts = line.split(":", 1)
                    q = parts[0].strip()
                    e = parts[1].strip() if len(parts) > 1 else ""
                    if q:
                        guide.append({"question": q, "what_to_evaluate": e})
            if isinstance(rb, str):
                rb = [r.strip("- • ") for r in rb.split("\n") if r.strip()]
            if isinstance(sc, str):
                sc = [s.strip("- • ") for s in sc.split("\n") if s.strip()]
            state["interview_kit"] = InterviewKit(
                screening_questions=sq[:12],
                screening_guide=guide[:12],
                evaluation_rubric=rb[:12],
                scorecard=sc[:12],
                outreach_subject=subj,
                outreach_body=body,
            )
        except Exception as parse_err:
            print(f"Parse error in interview_kit_node: {parse_err}")
            state["interview_kit"] = InterviewKit(
                screening_questions=[
                    "Tell us about a recent project related to this role.",
                    "Walk through a challenging problem you solved and trade-offs considered.",
                    "How do you ensure quality and impact in fast-paced environments?",
                ],
                screening_guide=[
                    {"question": "Describe a recent project and your specific impact.", "what_to_evaluate": "Clarity, ownership, measurable outcomes"},
                    {"question": "Walk me through a tough technical decision.", "what_to_evaluate": "Trade-offs, reasoning, stakeholder alignment"},
                ],
                evaluation_rubric=[
                    "Technical depth",
                    "Problem solving",
                    "Communication",
                    "Ownership & initiative",
                    "Collaboration & culture add",
                ],
                scorecard=[
                    "Technical Depth (1-5)",
                    "Problem Solving (1-5)",
                    "Communication (1-5)",
                    "Culture Add (1-5)",
                    "Overall Recommendation",
                    "Notes",
                ],
                outreach_subject="Exciting opportunity to make impact as {role}".format(role=state["extraction"].roles[0] if state["extraction"].roles else "key hire"),
                outreach_body="Hi [Name], we’re building something meaningful and think your background fits. Would you be open to a quick chat about our {role} role?".format(role=state["extraction"].roles[0] if state["extraction"].roles else "role"),
            )
    except Exception:
        skills = state["extraction"].key_skills if state.get("extraction") else []
        key = ", ".join(skills[:3]) if skills else "the core skills for this role"
        state["interview_kit"] = InterviewKit(
            screening_questions=[
                f"Can you walk me through a recent project where you applied {key}? What was your specific impact?",
                f"How do you measure success in this role, and what metrics did you move recently?",
                f"Describe a challenging problem you solved using {skills[0] if skills else 'your primary skill'}. What trade-offs did you consider?",
                "What does good code/analysis quality look like to you, and how do you ensure it under deadlines?",
                "Tell me about a time you collaborated cross-functionally. How did you handle misalignment?",
            ],
            screening_guide=[
                {"question": "Describe a recent project and your specific impact.", "what_to_evaluate": "Clarity, ownership, measurable outcomes"},
                {"question": "Walk me through a tough technical decision.", "what_to_evaluate": "Trade-offs, reasoning, stakeholder alignment"},
            ],
            evaluation_rubric=[
                "Technical depth",
                "Problem solving",
                "Communication",
                "Ownership & initiative",
                "Collaboration & culture add",
            ],
            scorecard=[
                "Technical Depth (1-5)",
                "Problem Solving (1-5)",
                "Communication (1-5)",
                "Culture Add (1-5)",
                "Overall Recommendation",
                "Notes",
            ],
        )
    return state


def _render_markdown(state: HiringGraphState) -> str:
    """Deterministic final report markdown. Uses JD compensation_note only."""
    roles = ", ".join([jd.title for jd in state["job_descriptions"]]) if state["job_descriptions"] else "the role"
    timeline = state["hiring_plan"].timeline_weeks if state["hiring_plan"] else (state["extraction"].timeline_weeks if state["extraction"] else 8)

    md = []
    md.append("# Hiring Plan Report")
    md.append("## Executive Summary")
    md.append(
        f"This hiring plan outlines the process to recruit **{roles}**. "
        f"The target timeline is **{timeline} week(s)**. "
        "The plan includes sourcing, screening, assessment, interviews, and onboarding."
    )
    md.append("\n## Job Descriptions")
    if state["job_descriptions"]:
        for jd in state["job_descriptions"]:
            md.append(f"### {jd.title}")
            md.append("#### Summary")
            md.append(jd.summary)
            if jd.responsibilities:
                md.append("#### Key Responsibilities")
                for r in jd.responsibilities:
                    md.append(f"- {r}")
            if jd.requirements:
                md.append("#### Requirements")
                for rq in jd.requirements:
                    md.append(f"- {rq}")
            if jd.compensation_note:
                md.append(f"**Compensation:** {jd.compensation_note}")
            md.append("")  
    else:
        md.append("_No job descriptions available._")

    md.append("\n## Hiring Process Plan")
    if state["hiring_plan"]:
        md.append(f"**Timeline:** {state['hiring_plan'].timeline_weeks} weeks")
        md.append("**Process Steps:**")
        for i, (step, owner) in enumerate(zip(state["hiring_plan"].steps, state["hiring_plan"].owners), start=1):
            md.append(f"{i}. {step}  \n   *Owner: {owner}*")
    else:
        md.append("_No hiring plan generated._")

# Interview Kit
    if state.get("interview_kit"):
        md.append("\n## Interview Kit")
        if state["interview_kit"].screening_questions:
            md.append("**Screening Questions:**")
            for q in state["interview_kit"].screening_questions:
                md.append(f"- {q}")
        if state["interview_kit"].evaluation_rubric:
            md.append("**Evaluation Rubric:**")
            for r in state["interview_kit"].evaluation_rubric:
                md.append(f"- {r}")
        if state["interview_kit"].scorecard:
            md.append("**Scorecard Fields:**")
            for s in state["interview_kit"].scorecard:
                md.append(f"- {s}")
        if state["interview_kit"].outreach_subject or state["interview_kit"].outreach_body:
            md.append("**Outreach Template:**")
            if state["interview_kit"].outreach_subject:
                md.append(f"Subject: {state['interview_kit'].outreach_subject}")
            if state["interview_kit"].outreach_body:
                md.append("")
                md.append(state["interview_kit"].outreach_body)

    md.append("\n## Next Steps")
    md.append("- Share this plan with stakeholders for alignment.")
    md.append("- Start sourcing and schedule initial screens.")
    md.append("- Prepare assessments and interview materials.")

    return "\n".join(md)


def assemble_node(state: HiringGraphState) -> HiringGraphState:
    """Assemble final output WITHOUT calling the LLM, so salary cannot be hallucinated."""
    try:
        markdown_output = _render_markdown(state)
        json_output = {
            "extraction": state["extraction"].model_dump() if state["extraction"] else {},
            "job_descriptions": [jd.model_dump() for jd in state["job_descriptions"]],
            "hiring_plan": state["hiring_plan"].model_dump() if state["hiring_plan"] else {},
            "interview_kit": state["interview_kit"].model_dump() if state.get("interview_kit") else {},
            "clarifications_asked": state["clarifications_asked"],
            "timestamp": "generated"
        }
        state["raw_markdown"] = markdown_output
        state["raw_json"] = json_output
    except Exception as e:
        print(f"Error in assemble_node: {e}")
        state["raw_markdown"] = f"# Hiring Plan\n\nRequest:\n{state['user_input']}\n\n(assembly failed: {e})"
        state["raw_json"] = {"error": "Assembly failed", "message": str(e)}
    return state


def create_hiring_graph() -> StateGraph:
    workflow = StateGraph(HiringGraphState)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("jd", jd_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("kit", interview_kit_node)
    workflow.add_node("assemble", assemble_node)
    workflow.add_edge("clarify", "extract")
    workflow.add_edge("extract", "jd")
    workflow.add_edge("jd", "plan")
    workflow.add_edge("plan", "kit")
    workflow.add_edge("kit", "assemble")
    workflow.add_edge("assemble", END)
    workflow.set_entry_point("clarify")
    return workflow


def run_hiring_workflow(user_input: str, overrides: Optional[Dict[str, Any]] = None) -> AgentOutput:
    try:
        graph = create_hiring_graph()
        compiled_graph = graph.compile()
        initial_state = {
            "user_input": user_input,
            "clarifications_asked": [],
            "extraction": None,
            "job_descriptions": [],
            "hiring_plan": None,
            "interview_kit": None,
            "raw_markdown": "",
            "raw_json": {},
            "overrides": overrides or {},
            "needs_clarification": False
        }
        final_state = compiled_graph.invoke(initial_state)
        result = AgentOutput(
            clarifications_asked=final_state["clarifications_asked"],
            jd_drafts=final_state["job_descriptions"],
            hiring_plan=final_state["hiring_plan"],
            raw_markdown=final_state["raw_markdown"],
            raw_json=final_state["raw_json"]
        )
        return result
    except Exception as e:
        print(f"Error running hiring workflow: {e}")
        return AgentOutput(
            clarifications_asked=[],
            jd_drafts=[],
            hiring_plan=None,
            raw_markdown=f"Error: {str(e)}",
            raw_json={"error": str(e)}
        )
