"""Main agent interface for the HR Hiring Copilot."""

import os
from typing import Dict, Any, Optional
from .graph import run_hiring_workflow
from .llm import ask_llm
from .prompts import SMALL_TALK_PROMPT, REWRITE_JD_PROMPT, REWRITE_KIT_PROMPT, INTENT_CLASSIFY_PROMPT
from .schema import AgentOutput
from .memory import record_run

_compiled_graph = None


def _classify_intent_llm(message: str) -> str:
    """Classify intent using LLM. Returns one of: hiring|refinement|small_talk."""
    try:
        import json as _json
        prompt = INTENT_CLASSIFY_PROMPT.format(user_input=message)
        resp = ask_llm(prompt)
        # Loose parse
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
                        obj = _json.loads(cleaned[start:i+1])
                        intent = str(obj.get("intent", "")).strip()
                        if intent in ("hiring", "refinement", "small_talk"):
                            return intent
        return "hiring" if _is_hiring_request(message) else "small_talk"
    except Exception:
        return "hiring" if _is_hiring_request(message) else "small_talk"


def run_agent_langgraph(user_input: str, overrides: Optional[Dict[str, Any]] = None) -> AgentOutput:
    """
    Main entry point for the HR agent using LangGraph.
    """
    try:
        provider = os.getenv("LLM_PROVIDER", "stub")

        intent = _classify_intent_llm(user_input)
        if intent == "hiring":
            result = run_hiring_workflow(user_input, overrides)

            roles = []
            if result.jd_drafts:
                roles = [jd.title for jd in result.jd_drafts]
            elif overrides and "roles" in overrides:
                roles = overrides["roles"]

            timeline_weeks = 8
            if result.hiring_plan:
                timeline_weeks = result.hiring_plan.timeline_weeks
            elif overrides and "timeline_weeks" in overrides:
                timeline_weeks = overrides["timeline_weeks"]

            analytics_data = {
                "session_id": "unknown",
                "provider": provider,
                "user_prompt": user_input[:200],
                "roles": roles,
                "timeline_weeks": timeline_weeks,
                "success": len(result.jd_drafts) > 0
            }

            record_run(analytics_data)
            return result
        elif intent == "refinement":
            result = run_hiring_workflow(user_input, overrides)
            record_run({
                "session_id": "unknown",
                "provider": provider,
                "user_prompt": user_input[:200],
                "roles": [jd.title for jd in result.jd_drafts] if result.jd_drafts else [],
                "timeline_weeks": result.hiring_plan.timeline_weeks if result.hiring_plan else 8,
                "success": len(result.jd_drafts) > 0,
            })
            return result
        else:
            response = small_talk_reply(user_input)
            return AgentOutput(
                clarifications_asked=[],
                jd_drafts=[],
                hiring_plan=None,
                raw_markdown=response,
                raw_json={"message": response, "type": "small_talk"}
            )

    except Exception as e:
        print(f"Error in run_agent_langgraph: {e}")
        return AgentOutput(
            clarifications_asked=[],
            jd_drafts=[],
            hiring_plan=None,
            raw_markdown=f"I apologize, but I encountered an error: {str(e)}. Please try again or contact support.",
            raw_json={"error": str(e)}
        )


def small_talk_reply(message: str) -> str:
    """
    Handle small talk and general conversation.
    """
    try:
        prompt = SMALL_TALK_PROMPT.format(user_input=message)
        response = ask_llm(prompt)
        return response
    except Exception as e:
        print(f"Error in small_talk_reply: {e}")
        return ("Hi there! I'm your HR Hiring Copilot. I can help you create job descriptions and hiring plans. "
                "Try asking me to 'help hire a data analyst' or 'create a hiring plan for a software engineer' to get started!")


def _is_hiring_request(user_input: str) -> bool:
    """
    Determine if user input is a hiring-related request.
    """
    hiring_keywords = [
        "hire", "hiring", "recruit", "recruiting", "job", "position",
        "role", "candidate", "employee", "staff", "team member",
        "job description", "jd", "job posting", "job spec",
        "talent", "workforce", "personnel", "staffing",
        "screening", "screening questions", "interview questions", "interview kit"
    ]

    user_lower = user_input.lower()
    for keyword in hiring_keywords:
        if keyword in user_lower:
            return True

    role_keywords = [
        "engineer", "developer", "analyst", "manager", "designer",
        "marketer", "salesperson", "accountant", "consultant",
        "specialist", "coordinator", "director", "lead", "senior",
        "junior", "intern", "contractor", "freelancer"
    ]
    for role in role_keywords:
        if role in user_lower:
            return True

    action_keywords = [
        "need a", "looking for", "want to hire", "seeking", "find a",
        "add to team", "expand team", "grow team", "staff up"
    ]
    for action in action_keywords:
        if action in user_lower:
            return True

    return False


def get_agent_status() -> Dict[str, Any]:
    """
    Get current agent status and configuration.
    """
    return {
        "provider": os.getenv("LLM_PROVIDER", "stub"),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "graph_loaded": _compiled_graph is not None,
        "version": "1.0.0"
    }


def refine_jd_with_llm(jd_json: Dict[str, Any], tweak: str) -> Optional[Dict[str, Any]]:
    """Ask the LLM to rewrite a single JD JSON according to a tweak. Returns parsed JSON or None."""
    try:
        import json as _json
        prompt = REWRITE_JD_PROMPT.format(tweak=tweak, jd_json=_json.dumps(dict(jd_json), ensure_ascii=False))
        resp = ask_llm(prompt)
        try:
            return _json.loads(resp)
        except Exception:
          
            import re as _re
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
                            candidate = cleaned[start:i+1]
                            try:
                                return _json.loads(candidate)
                            except Exception:
                                return None
        return None
    except Exception:
        return None


def refine_kit_with_llm(kit_json: Dict[str, Any], tweak: str) -> Optional[Dict[str, Any]]:
    """Ask the LLM to rewrite interview kit sections according to a tweak. Returns parsed JSON or None."""
    try:
        import json as _json
        prompt = REWRITE_KIT_PROMPT.format(tweak=tweak, kit_json=_json.dumps(dict(kit_json), ensure_ascii=False))
        resp = ask_llm(prompt)
        try:
            obj = _json.loads(resp)
        except Exception:
            import re as _re
            cleaned = resp.replace("“", '"').replace("”", '"').replace("’", "'")
            start = cleaned.find('{')
            if start == -1:
                return None
            depth = 0
            obj = None
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:i+1]
                        try:
                            obj = _json.loads(candidate)
                        except Exception:
                            obj = None
                        break
        if not isinstance(obj, dict):
            return None    
        allowed = {"screening_questions", "evaluation_rubric", "scorecard"}
        refined = {k: v for k, v in obj.items() if k in allowed}
        return refined
    except Exception:
        return None


def validate_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean override values from UI.
    NOTE: budget_range removed; salary_range is now the source of truth.
    """
    if not overrides:
        return {}

    cleaned: Dict[str, Any] = {}

    if "timeline_weeks" in overrides:
        try:
            weeks = int(overrides["timeline_weeks"])
            cleaned["timeline_weeks"] = max(1, min(weeks, 52))
        except (ValueError, TypeError):
            cleaned["timeline_weeks"] = 8

 
    string_fields = [
        "seniority",
        "employment_type",
        "location",
        "company_stage",
        "salary_range",
        "min_qualification",
        "tweak",
    ]
    for field in string_fields:
        if field in overrides and overrides[field] is not None:
            cleaned[field] = str(overrides[field]).strip()

    if "salary_range" not in cleaned and "budget_range" in overrides and overrides["budget_range"] is not None:
        cleaned["salary_range"] = str(overrides["budget_range"]).strip()

    if "key_skills" in overrides:
        if isinstance(overrides["key_skills"], list):
            cleaned["key_skills"] = [str(skill).strip() for skill in overrides["key_skills"] if skill]
        elif isinstance(overrides["key_skills"], str):
            cleaned["key_skills"] = [s.strip() for s in overrides["key_skills"].split(",") if s.strip()]

    if "roles" in overrides:
        if isinstance(overrides["roles"], list):
            cleaned["roles"] = [str(role).strip() for role in overrides["roles"] if role]
        elif isinstance(overrides["roles"], str):
            cleaned["roles"] = [role.strip() for role in overrides["roles"].split(",") if role.strip()]

    advanced_string_fields = [
        "department",
        "work_authorization",
        "target_start",
        "urgency",
        "ideal_background",
        "culture_values",
        "company_overview",
        "diversity_inclusion_goals",
    ]
    for field in advanced_string_fields:
        if field in overrides and overrides[field] is not None:
            cleaned[field] = str(overrides[field]).strip()

    advanced_list_fields = [
        "must_have_skills",
        "nice_to_have_skills",
        "deal_breakers",
        "process_stages",
        "decision_makers",
        "evaluation_focus",
        "unique_selling_points",
    ]
    for field in advanced_list_fields:
        if field in overrides:
            if isinstance(overrides[field], list):
                cleaned[field] = [str(item).strip() for item in overrides[field] if item]
            elif isinstance(overrides[field], str):
                cleaned[field] = [s.strip() for s in overrides[field].split(",") if s.strip()]

    if "min_experience" in overrides:
        try:
            cleaned["min_experience"] = max(0, int(overrides["min_experience"]))
        except (ValueError, TypeError):
            pass

    return cleaned
