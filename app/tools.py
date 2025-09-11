"""Simulated tools for the HR Hiring Copilot."""

import json
from typing import List, Dict, Any, Optional
import re
from app.llm import ask_llm
from app.prompts import SEARCH_RESULTS_PROMPT
from urllib.parse import quote_plus
from datetime import datetime
import os
import requests


def simulate_search(query: str) -> List[Dict[str, str]]:
    """
    Simulate a search tool that returns top 3 results for any query.
    
    Args:
        query: Search query string
        
    Returns:
        List of dictionaries with title, url, and snippet
    """
    # Try Google Custom Search if configured
    try:
        cx = os.getenv("GOOGLE_CSE_ID") or os.getenv("GOOGLE_SEARCH_CX")
        api_key = os.getenv("GOOGLE_API_KEY")
        if cx and api_key and query:
            cse = _google_cse_search(query, api_key, cx)
            if cse:
                return cse[:3]
    except Exception:
        pass

    # Try DuckDuckGo HTML (no API key)
    try:
        ddg = _ddg_html_search(query)
        if ddg:
            return ddg[:3]
    except Exception:
        pass

    # Prefer LLM-synthesized results for realism
    try:
        prompt = SEARCH_RESULTS_PROMPT.format(query=query)
        resp = ask_llm(prompt)
        cleaned = resp.replace("“", '"').replace("”", '"').replace("’", "'")
        # Find first JSON array
        start = cleaned.find('[')
        if start != -1:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '[':
                    depth += 1
                elif cleaned[i] == ']':
                    depth -= 1
                    if depth == 0:
                        import json as _json
                        arr = _json.loads(cleaned[start:i+1])
                        # Validate shape
                        norm = []
                        for item in arr[:3]:
                            if isinstance(item, dict) and all(k in item for k in ("title", "url", "snippet")):
                                norm.append({
                                    "title": str(item["title"]).strip(),
                                    "url": str(item["url"]).strip(),
                                    "snippet": str(item["snippet"]).strip(),
                                })
                        if norm:
                            return norm
    except Exception:
        pass

    # Fallback to curated stubs
    results: List[Dict[str, str]] = []
    ql = query.lower()
    if "data analyst" in ql:
        results = [
            {
                "title": "Top Data Analyst Interview Questions (SQL, Case Studies)",
                "url": "https://www.interviewbit.com/data-analyst-interview-questions/",
                "snippet": "Comprehensive questions across SQL, statistics, and business cases with sample answers."
            },
            {
                "title": "Data Analyst Salary Trends (2024)",
                "url": "https://www.levels.fyi/compensation/Data-Analyst/",
                "snippet": "Current compensation data for data analyst roles across companies and locations."
            },
            {
                "title": "How to Build a Data Team at a Startup",
                "url": "https://a16z.com/2019/05/11/building-data-science-team/",
                "snippet": "Frameworks and best practices for standing up analytics teams and roles."
            }
        ]
    elif "engineer" in ql or "developer" in ql:
        results = [
            {
                "title": "Software Engineer Hiring Guide",
                "url": "https://www.greenhouse.com/blog/how-to-hire-software-engineers",
                "snippet": "Steps, assessments, and interview guidance for engineering hires."
            },
            {
                "title": "System Design Interview Prep",
                "url": "https://github.com/donnemartin/system-design-primer",
                "snippet": "Open-source primer covering key system design concepts and practice."
            },
            {
                "title": "Remote Engineering Best Practices",
                "url": "https://about.gitlab.com/handbook/",
                "snippet": "GitLab's public handbook on remote work, collaboration, and engineering processes."
            }
        ]
    elif "marketing" in ql:
        results = [
            {
                "title": "Marketing Interview Questions (Strategy & Analytics)",
                "url": "https://www.glassdoor.com/Interview/marketing-interview-questions-SRCH_KO0,9.htm",
                "snippet": "Common questions with tips for evaluating strategic thinking and measurement."
            },
            {
                "title": "Content Marketing Playbook",
                "url": "https://contentmarketinginstitute.com/",
                "snippet": "Resources and guides for content strategy, distribution, and ROI."
            },
            {
                "title": "Growth Marketing Tactics",
                "url": "https://www.reforge.com/blog",
                "snippet": "Deep dives on growth loops, experimentation, and channel strategies."
            }
        ]
    else:
        results = [
            {
                "title": f"Top resources for: {query}",
                "url": google_search_url(query),
                "snippet": f"Open Google results for '{query}' and refine from authoritative sources."
            }
        ]

    return results[:3]


def email_writer(to: str, subject: str, body_outline: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate an email draft based on recipient, subject, and body outline.
    
    Args:
        to: Email recipient
        subject: Email subject line
        body_outline: Outline or key points for the email body
        
    Returns:
        Formatted email draft as string
    """
    # Create a professional email template
    ctx = context or {}
    tailored_body = _generate_email_body(body_outline, ctx)
    # Greeting
    name = _get_salutation(to)
    greeting = f"Hi {name}," if name else "Hi there,"
    # Sign-off
    recruiter_name = ctx.get("recruiter_name", "[Your Name]")
    recruiter_title = ctx.get("recruiter_title", "Talent")
    company_name = ctx.get("company_name", "[Company Name]")
    contact_info = ctx.get("contact_info", "[Contact Information]")

    email_draft = f"""To: {to}
Subject: {subject}

{greeting}

{tailored_body}

Best regards,
{recruiter_name}
{recruiter_title}
{company_name}
{contact_info}

---
This email was drafted by the HR Hiring Copilot. Please review and customize before sending."""
    
    return email_draft


def _get_salutation(to: str) -> str:
    """Extract appropriate salutation from email address for candidate outreach."""
    if "@" in to:
        name_part = to.split("@")[0]
        # Simple name extraction - could be enhanced
        if "." in name_part:
            first_name = name_part.split(".")[0]
            return first_name.title()
        elif "_" in name_part:
            first_name = name_part.split("_")[0]
            return first_name.title()
        elif name_part:
            return name_part.title()
    return "there"


def _generate_email_body(outline: str, ctx: Dict[str, Any]) -> str:
    """Generate email body based on outline and optional context from wizard/results."""
    def _sanitize_outline(text: str) -> str:
        lines = [ln for ln in (text or "").splitlines()]
        cleaned: List[str] = []
        for ln in lines:
            l = ln.strip()
            low = l.lower()
            if not l:
                cleaned.append(ln)
                continue
            if low.startswith(("hi ", "hello", "dear")):
                continue
            if low.startswith(("best regards", "regards", "thanks", "thank you", "sincerely")):
                continue
            cleaned.append(ln)
        return "\n".join(cleaned).strip()

    outline = _sanitize_outline(outline)
    role = ctx.get("role_title") or ctx.get("role") or "the role"
    seniority = ctx.get("seniority")
    location = ctx.get("location")
    company_stage = ctx.get("company_stage")
    salary = ctx.get("salary_range") or ctx.get("budget_range")
    key_skills = ctx.get("key_skills") or []
    usp = ctx.get("unique_selling_points") or []
    timeline_weeks = ctx.get("timeline_weeks")
    process_stages = ctx.get("process_stages") or []
    outreach_subject = ctx.get("outreach_subject")
    outreach_body = ctx.get("outreach_body")

    # If JD context exists, prefer composing from JD regardless of outline
    has_jd_ctx = bool(ctx.get("jd_summary") or ctx.get("jd_responsibilities"))
    # If interview kit outreach provided and no JD context, use it (strip greetings/signoffs)
    if outreach_body and not has_jd_ctx:
        return _sanitize_outline(outreach_body)
    # JD- or outreach-based candidate email
    if has_jd_ctx or ("candidate" in outline.lower() or "outreach" in outline.lower()):
        company_name = ctx.get("company_name") or "our team"
        skills_line = f"Your experience with {', '.join(key_skills[:6])} stood out to us." if key_skills else ""
        role_line = f"a {seniority + ' ' if seniority else ''}{role}".strip()
        location_line = f" This role is {location}." if location else ""
        comp_line = f" Compensation is {salary}." if salary else ""
        usp_line = f" Why {company_name}: {'; '.join(usp[:3])}." if usp else ""
        timeline_line = f" Our process is efficient (~{timeline_weeks} weeks)." if timeline_weeks else ""
        stages_line = f" Stages: {', '.join(process_stages[:5])}." if process_stages else ""

        # JD-driven details
        jd_summary = ctx.get("jd_summary")
        jd_resps: List[str] = ctx.get("jd_responsibilities") or []
        jd_comp = ctx.get("jd_compensation_note") or salary

        what_you_do = "\n".join([f"- {r}" for r in jd_resps[:4]]) if jd_resps else "- Make meaningful impact from day one\n- Collaborate cross-functionally\n- Ship and iterate quickly"
        comp_line = f" Compensation: {jd_comp}." if jd_comp else comp_line

        return f"""I'm {ctx.get('recruiter_name','an HR recruiter')} at {company_name}. We’re hiring {role_line}.{location_line}{comp_line}

{skills_line}

{jd_summary or outline}

What you'll do:
{what_you_do}

Would you be open to a short intro call to share more and learn about your interests?{timeline_line}{stages_line}

If there's a better time or channel to reach you, please let me know."""
    
    # Check if this is for interview scheduling
    elif "interview" in outline.lower() or "schedule" in outline.lower():
        return f"""Thank you for your interest in our open position. We were impressed with your application and would like to move forward with the next step in our process.

{outline}

Please let me know your availability for the coming week, and I'll send over a calendar invite with all the details.

Looking forward to speaking with you soon!"""
    
    # Generic professional email
    else:
        company_name = ctx.get("company_name") or "our company"
        context_line = " ".join(filter(None, [
            f"Role: {seniority + ' ' if seniority else ''}{role} at {company_name}.",
            f"Location: {location}." if location else "",
            f"Compensation: {salary}." if salary else "",
            f"Key skills: {', '.join(key_skills[:6])}." if key_skills else "",
        ])).strip()
        return f"""I hope you're doing well. {context_line}

{outline}

Please let me know if you have any questions or if there's anything else I can help clarify.

Thank you for your time and consideration."""


def linkedin_candidate_search(skills: List[str], title: str, location: str = "", seniority: str = "") -> Dict[str, Any]:
    """Build a LinkedIn public search URL and simulate candidate results.

    Returns a dict with boolean_query, search_url, and simulated results.
    """
    skills_terms = [s for s in [skill.strip() for skill in skills] if s]
    title_term = f'"{title}"' if title else ""
    seniority_term = seniority.upper() if seniority else ""

    terms = []
    if title_term:
        terms.append(title_term)
    terms.extend([f'"{t}"' for t in skills_terms])
    if seniority_term:
        terms.append(seniority_term)
    boolean_query = " AND ".join(terms)
    if location:
        boolean_query = f"{boolean_query} AND \"{location}\"" if boolean_query else f'"{location}"'

    keywords_param = quote_plus(boolean_query)
    search_url = f"https://www.linkedin.com/search/results/people/?keywords={keywords_param}&origin=GLOBAL_SEARCH_HEADER"

    results = []
    for i, skill in enumerate(skills_terms[:3], start=1):
        results.append({
            "name": f"Candidate {i}",
            "headline": f"{seniority + ' ' if seniority else ''}{title} | {skill}",
            "location": location or "Remote",
            "profile_url": f"https://www.linkedin.com/in/example-candidate-{i}"
        })

    return {
        "boolean_query": boolean_query,
        "search_url": search_url,
        "results": results
    }


def checklist_export(plan_json: Dict[str, Any]) -> str:
    """
    Export a hiring plan as a formatted markdown checklist.
    
    Args:
        plan_json: Dictionary containing hiring plan data
        
    Returns:
        Markdown formatted checklist string
    """
    try:
        steps = plan_json.get("steps", [])
        owners = plan_json.get("owners", [])
        timeline_weeks = plan_json.get("timeline_weeks", 8)
        
        
        checklist = f"""# Hiring Process Checklist

**Timeline:** {timeline_weeks} weeks
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Process Steps

"""
        
        for i, step in enumerate(steps):
            owner = owners[i] if i < len(owners) else "TBD"
            week_estimate = _estimate_week_for_step(i, len(steps), timeline_weeks)
            
            checklist += f"- [ ] **Week {week_estimate}**: {step}\n"
            checklist += f"  - *Owner:* {owner}\n"
            checklist += f"  - *Status:* Pending\n"
            checklist += f"  - *Notes:*\n\n"
        
        checklist += f"""## Additional Reminders

- [ ] Prepare interview questions and evaluation criteria
- [ ] Set up candidate tracking system/spreadsheet
- [ ] Coordinate interview panel schedules
- [ ] Prepare reference check questions
- [ ] Draft offer letter template
- [ ] Plan onboarding checklist for new hire

## Timeline Overview

**Week 1-2:** Sourcing and initial screening
**Week 3-4:** Technical assessments and first interviews
**Week 5-6:** Panel interviews and team meetings
**Week 7-8:** Final interviews, references, and offers

---
*Checklist exported from HR Hiring Copilot*"""
        
        return checklist
        
    except Exception as e:
        return f"Error generating checklist: {str(e)}"


def _estimate_week_for_step(step_index: int, total_steps: int, total_weeks: int) -> int:
    """Estimate which week a step should occur in."""
    if total_steps == 0:
        return 1
    
    # Distribute steps across weeks
    step_ratio = step_index / total_steps
    estimated_week = max(1, int(step_ratio * total_weeks) + 1)
    return min(estimated_week, total_weeks)


def get_available_tools() -> List[str]:
    """Return list of available tool names for the UI."""
    return [
        "simulate_search",
        "email_writer",
        "checklist_export",
        "linkedin_candidate_search",
        "ats_resume_check",
        "google_search_url",
    ]


def ats_resume_check(jd: Dict[str, Any], resume_text: str) -> Dict[str, Any]:
    """Heuristic ATS-like matcher that scores a resume against a JD.

    Returns a dict with score (0-100), matched_skills, missing_skills, and notes.
    """
    if not jd or not resume_text:
        return {"score": 0, "matched_skills": [], "missing_skills": [], "notes": ["Missing JD or resume."]}
    text = (resume_text or "").lower()
    reqs = jd.get("requirements", []) or []
    resps = jd.get("responsibilities", []) or []
    combined = " ".join(reqs + resps).lower()
    import re
    tokens = [t.strip() for t in re.split(r"[,/]|\n|;|\|", combined) if t.strip()]
    skills = []
    for t in tokens:
        parts = t.split()
        if not parts:
            continue
        candidate = " ".join(parts[:2])
        if len(candidate) >= 2 and candidate not in skills:
            skills.append(candidate)
    matched = []
    for s in skills:
        s_norm = s.lower()
        if s_norm in text:
            matched.append(s)
    missing = [s for s in skills if s not in matched]

    total = max(1, len(skills))
    score = int(round((len(matched) / total) * 100))
    
    notes = []
    if missing:
        notes.append(f"Consider highlighting: {', '.join(missing[:5])}")
    if score < 60:
        notes.append("Resume may be filtered by ATS; add more keywords from the JD.")
    else:
        notes.append("Resume aligns reasonably with JD keywords.")
    return {
        "score": score,
        "matched_skills": matched[:20],
        "missing_skills": missing[:20],
        "notes": notes
    }


def _google_cse_search(query: str, api_key: str, cx: str) -> List[Dict[str, str]]:
    """Use Google Custom Search JSON API to fetch top results."""
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"q": query, "key": api_key, "cx": cx, "num": 3},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json() or {}
        items = data.get("items") or []
        out: List[Dict[str, str]] = []
        for it in items[:3]:
            title = str(it.get("title") or "").strip()
            url = str(it.get("link") or "").strip()
            snippet = str(it.get("snippet") or it.get("htmlSnippet") or "").strip()
            if title and url:
                out.append({"title": title, "url": url, "snippet": snippet})
        return out
    except Exception:
        return []


def _ddg_html_search(query: str) -> List[Dict[str, str]]:
    """Fetch top results from DuckDuckGo HTML endpoint (no API key needed)."""
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    html = r.text
    matches = re.findall(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S)
    out: List[Dict[str, str]] = []
    for href, title_html in matches[:3]:
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = f"Web result for '{query}' (DuckDuckGo)."
        if href and title:
            out.append({"title": title, "url": href, "snippet": snippet})
    return out


def google_search_url(query: str) -> str:
    """Build a public Google search URL for a given query."""
    if not query:
        return "https://www.google.com/search?q="
    return f"https://www.google.com/search?q={quote_plus(query)}"
