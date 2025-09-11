"""Prompts for the HR Hiring Copilot."""

# Slot extraction prompt with escaped JSON braces
SLOT_EXTRACT_PROMPT = """You are an expert HR assistant helping to extract key information for a hiring process.

Given the user input: {user_input}

Extract the following information and return it as a valid JSON object with these exact keys:

{{
  "company_stage": "startup|scale-up|enterprise",
  "timeline_weeks": <number>,
  "budget_range": "<budget description>",
  "seniority": "junior|mid|senior|lead|principal",
  "employment_type": "full-time|part-time|contract|intern",
  "location": "<location description>",
  "key_skills": ["skill1", "skill2", ...],
  "roles": ["role1", "role2", ...]
}}

Guidelines:
- If information is not provided, use reasonable defaults
- Extract all mentioned roles and skills
- For budget, preserve any specific numbers or ranges mentioned
- For timeline, default to 8 weeks if not specified
- Company stage defaults to "startup"

Return only the JSON object, no other text."""

# Small talk prompt for general conversation
SMALL_TALK_PROMPT = """You are a friendly HR hiring assistant. The user said: {user_input}

Respond warmly in 1-2 sentences and suggest how you can help them with hiring. Keep it brief and professional.

If they're asking about hiring or recruitment, suggest they can ask you to "help hire [role]" to start the hiring wizard."""

# Job description generation prompt
JD_GENERATION_PROMPT = """You are an expert recruiter creating a LinkedIn-style job description for a startup.

Company details:
- Stage: {company_stage}
- Budget: {budget_range}

Role details:
- Title: {role_title}
- Seniority: {seniority}
- Location: {location}
- Key skills: {key_skills}

Create a job description with startup energy and tone. Return a JSON object with this structure:

{{
  "title": "<role title>",
  "summary": "<engaging 2-3 sentence summary>",
  "responsibilities": [
    "<responsibility 1>",
    "<responsibility 2>",
    "<5-9 bullets total>"
  ],
  "requirements": [
    "<requirement 1>",
    "<requirement 2>",
    "<5-9 bullets total>"
  ],
  "compensation_note": "<compensation info>"
}}

Guidelines:
- Use startup-friendly language (fast-paced, impact-driven, growth-minded)
- Responsibilities should be outcome-focused, not task-focused
- Requirements should balance must-haves with nice-to-haves
- Include 5-9 bullets for both responsibilities and requirements
- Use the compensation note from the budget information provided

Return only the JSON object."""

# Hiring plan generation prompt
PLAN_JSON_PROMPT = """You are an expert talent acquisition specialist creating a hiring process plan.

Create a structured hiring plan for the following scenario:
- Timeline: {timeline_weeks} weeks
- Roles: {roles}
- Company stage: {company_stage}

Return a JSON object with this exact structure:

{{
  "steps": [
    "step 1 description",
    "step 2 description",
    "<continue with all steps>"
  ],
  "owners": [
    "owner for step 1",
    "owner for step 2",  
    "<continue with owners for each step>"
  ],
  "timeline_weeks": {timeline_weeks}
}}

Guidelines:
- Include typical startup hiring steps: sourcing, screening, interviews, reference checks, offer, onboarding
- Steps and owners arrays must be the same length
- Owners should be realistic roles (Hiring Manager, HR, Team Lead, CEO, etc.)
- Adapt process complexity to timeline constraints
- Include modern recruiting practices (async video screening, skills assessments, etc.)

Return only the JSON object."""

# Final assembly prompt for markdown output
ASSEMBLY_PROMPT = """You are formatting the final hiring plan output for an HR professional.

Given the following information:
- Job Descriptions: {job_descriptions}
- Hiring Plan: {hiring_plan}
- Original Request: {original_request}

Create a comprehensive markdown report with the following structure:

# Hiring Plan Report

## Executive Summary
[Brief overview of the hiring initiative]

## Job Descriptions

### [Role Title 1]
**Summary:** [job summary]

**Key Responsibilities:**
- [responsibility 1]
- [responsibility 2]
- [etc.]

**Requirements:**
- [requirement 1]
- [requirement 2]
- [etc.]

**Compensation:** [compensation note]

[Repeat for each role]

## Hiring Process Plan

**Timeline:** {timeline_weeks} weeks

**Process Steps:**
1. [step 1] - *Owner: [owner 1]*
2. [step 2] - *Owner: [owner 2]*
[Continue for all steps]

## Next Steps
[Actionable next steps for the HR professional]

Make the output professional, actionable, and startup-focused. Use clear formatting and be specific."""


# Interview kit generation prompt
INTERVIEW_KIT_PROMPT = """You are an expert recruiter creating an interview kit for the role.

Role context:
- Title(s): {roles}
- Seniority: {seniority}
- Key skills: {key_skills}
- Company stage: {company_stage}

Return a JSON object with this exact structure:

{{
  "screening_questions": [
    "<10 concise screening/interview questions tailored to the role and seniority>"
  ],
  "screening_pairs": [
    {"question": "<screening question>", "what_to_evaluate": "<what good answers demonstrate>"}
  ],
  "evaluation_rubric": [
    "<6-10 criteria combining technical, problem-solving, communication, ownership, and culture add>"
  ],
  "scorecard": [
    "<5-8 fields such as Technical Depth (1-5), Problem Solving (1-5), Communication (1-5), Culture Add (1-5), Overall Recommendation, Notes>"
  ],
  "outreach_subject": "<concise subject line for candidate outreach>",
  "outreach_body": "<short, friendly email inviting the candidate to talk about the role>"
}}

Guidelines:
- Questions must be specific to the role(s) and seniority
- Avoid generic questions; prioritize practical, scenario-based prompts
- Scorecard fields should be unambiguous and easy to use
- Outreach should be warm, concise, and describe impact, team, and process
 - For screening_pairs, include 6-10 pairs: each with a clear question and what to evaluate

Return only the JSON object."""


# JD refinement prompt (apply user edits to an existing JD)
REWRITE_JD_PROMPT = """You are an expert recruiter editing a job description.

Apply the following change request to this JD and return a valid JSON object with the same keys.

Change request:
{tweak}

Current JD (JSON):
{jd_json}

Requirements:
- Keep the structure and keys: title, summary, responsibilities (5-9), requirements (5-9), compensation_note
- Make coherent, human-quality edits (no fragments like single words)
- If the request changes experience (e.g., 'change experience to 1+ years'), update requirements accordingly
- If the request changes salary to a single value, keep compensation_note aligned to that exact target
- Do not include any text besides the JSON object
"""


# Interview kit refinement prompt
REWRITE_KIT_PROMPT = """You are an expert recruiter updating an interview kit.

Apply the following change request to this interview kit and return a valid JSON object with these keys only:

{
  "screening_questions": ["<updated question 1>", "<updated question 2>", ...],
  "evaluation_rubric": ["<updated rubric item 1>", ...],
  "scorecard": ["<updated scorecard field 1>", ...]
}

Change request:
{tweak}

Current Interview Kit (JSON):
{kit_json}

Requirements:
- Make coherent, human-quality edits (no one-word fragments)
- Keep questions concise and role-aligned
- Keep lists to 5–10 items where appropriate
- Do not include any text besides the JSON object
"""


# Clarification prompt for ambiguous hiring intents
CLARIFY_ROLE_PROMPT = """You are an expert hiring copilot. The user wrote:

{user_input}

Determine if this request is specific enough to generate a hiring plan directly, or if it needs clarification.

Return ONLY a valid JSON object with these exact keys:

{
  "need_clarification": true|false,
  "question": "<one concise clarifying question, if clarification is needed; else empty string>",
  "options": ["<concise option 1>", "<concise option 2>", "<up to 3-6 total options>"],
  "suggested_roles": ["<normalized role titles if you can infer them, else empty>"]
}

Guidelines:
- Treat these as AMBIGUOUS and set need_clarification=true unless a specific role is named:
  - Generic domains: "data", "software", "product", "design", "marketing", "sales", "ops"/"operations", "finance"
  - Unqualified roles: "engineer", "developer", "analyst", "manager" (without modifiers like data/software/frontend/backend/ML/AI/UX/etc.)
- For ambiguous inputs, ask a concise question and propose 3–6 concrete options from the appropriate family:
  - data → ["Data Analyst", "Data Scientist", "Data Engineer", "Machine Learning Engineer"]
  - software/developer → ["Backend Engineer", "Frontend Engineer", "Full-Stack Engineer", "Mobile Engineer"]
  - engineer (generic) → ["Software Engineer", "Data Engineer", "ML Engineer", "DevOps Engineer", "Security Engineer", "Site Reliability Engineer"]
  - product → ["Product Manager", "Product Designer", "Product Analyst"]
  - design → ["Product Designer", "UX Designer", "Visual Designer"]
  - marketing → ["Growth Marketer", "Product Marketer", "Performance Marketer", "Content Marketer"]
  - sales → ["Account Executive", "Sales Development Representative", "Sales Manager"]
  - operations → ["Operations Manager", "People Ops", "Business Operations Analyst"]
  - finance → ["Financial Analyst", "Accountant", "Controller"]
- If the input clearly specifies a role (e.g., "hire a senior Data Analyst"), set need_clarification=false and fill suggested_roles with normalized titles.
- Keep question and options very short and unambiguous.
- Do NOT include any prose outside the JSON.

Examples (JSON only):
{"need_clarification": true, "question": "For data, which role do you mean?", "options": ["Data Analyst", "Data Scientist", "Data Engineer", "Machine Learning Engineer"], "suggested_roles": []}
{"need_clarification": true, "question": "Which type of engineer?", "options": ["Software Engineer", "Data Engineer", "ML Engineer", "DevOps Engineer"], "suggested_roles": []}
{"need_clarification": false, "question": "", "options": [], "suggested_roles": ["Data Analyst"]}
"""

# Intent classification prompt for routing
INTENT_CLASSIFY_PROMPT = """You are classifying a user's message for an HR hiring copilot.

User message: {user_input}

Return ONLY a valid JSON object with these exact keys:

{
  "intent": "hiring|refinement|small_talk",
  "reason": "<very short explanation>",
  "is_ambiguous": true|false
}

Guidelines:
- intent=hiring when the user asks to hire/recruit, create JD/plan/interview kit, or mentions roles (engineer, analyst, etc.).
- intent=refinement when the user asks to edit/change/tweak/update/shorten/expand the existing plan or JD.
- intent=small_talk for greetings, chit-chat, or unrelated questions.
- Keep the response strictly JSON with the keys above; no extra text.
"""

# Salary extraction prompt
SALARY_EXTRACT_PROMPT = """You extract compensation details from text.

Text: {text}

Return ONLY a valid JSON object with these exact keys:
{{
  "salary_single": "<e.g., $120k or 120000 or 120k, empty if N/A>",
  "salary_min": "<lower bound like $90k or 90000, empty if N/A>",
  "salary_max": "<upper bound like $140k or 140000, empty if N/A>",
  "currency": "USD|EUR|INR|... or empty",
  "period": "yearly|hourly|monthly|empty"
}}

Guidelines:
- Preserve any numbers mentioned; normalize to k when appropriate (e.g., 120000 -> $120k if currency implied).
- If both min and max exist, leave salary_single empty.
- If only a single target exists, fill salary_single and leave min/max empty.
- Keep strictly to the JSON above.
"""

# Role and skills normalization prompt
ROLE_SKILL_NORMALIZE_PROMPT = """You are normalizing roles and extracting key skills for hiring.

User input: {user_input}
Candidate roles (may be empty): {candidate_roles}
Candidate skills (may be empty): {candidate_skills}

Return ONLY a valid JSON object with these exact keys:
{{
  "roles": ["<Normalized Role Title 1>", "<Role 2>", ...],
  "key_skills": ["<skill>", "<skill>", ...]
}}

Guidelines:
- Normalize common roles (e.g., ML Engineer -> Machine Learning Engineer, GenAI -> Generative AI Engineer).
- Prefer concise, industry-standard titles.
- Provide 5–10 focused, role-appropriate skills.
"""

# Chat summary prompt
CHAT_SUMMARY_PROMPT = """You are generating a concise assistant message summarizing hiring outputs.

Inputs (JSON):
- Roles: {roles}
- Timeline weeks: {timeline_weeks}
- Primary JD: {primary_jd}
- Interview kit (optional): {interview_kit}

Produce a clean message suitable for chat with:
- Header: "Here’s your organized hiring summary for [roles] (timeline: X weeks):"
- JD title and 1–2 sentence summary
- Top 3 responsibilities and top 3 requirements as bullets
- Compensation line if available
- 3–5 initial screening questions (from interview kit if present, else infer from JD/skills)
- Close with: "See the Interview Kit on the right for full details."

Return ONLY the final message text (no JSON, no extra commentary).
"""

# JD skills extraction prompt (for LinkedIn defaults)
JD_SKILLS_EXTRACT_PROMPT = """You extract top skills for sourcing from a JD JSON.

JD JSON:
{jd_json}

Return ONLY a valid JSON object like:
{{
  "top_skills": ["<skill1>", "<skill2>", "<up to 5 skills total>"]
}}

Guidelines:
- Choose succinct, searchable skill terms.
- Prefer technical or tool keywords where relevant.
"""

# Search results synthesis prompt
SEARCH_RESULTS_PROMPT = """You are a search results generator.

Given the user query: {query}

Return ONLY a valid JSON array of at most 3 objects, each with exactly:
[
  {"title": "<concise title>", "url": "https://...", "snippet": "<1-2 sentence summary>"}
]

Guidelines:
- Make results directly relevant to the query.
- Use realistic URLs (avoid example.com). Prefer well-known sites when possible.
- Keep titles concise and snippets informative.
- Do not include any text outside the JSON array.
"""