"""Compensation note composition utilities."""

import re
from typing import Optional


def compose_comp_note(role: str, stage: str, hr_budget: str) -> str:
    """
    Compose a compensation note based on role, company stage, and HR budget/salary band.

    Args:
        role: The job role/title
        stage: Company stage (startup, early-stage, scale-up, enterprise)
        hr_budget: A salary/band text (e.g., '50k–70k', '$80k-$120k', 'competitive', etc.)

    Returns:
        Formatted compensation note string
    """

    normalized = (hr_budget or "").strip().replace("–", "-").replace("—", "-")
    normalized = normalized.replace(" k", "k").replace(" K", "k")
    normalized = normalized.replace(" ", " ").strip()
    if _contains_salary_range(normalized):
        base_comp = normalized
    elif _contains_single_target(normalized):
        base_comp = normalized
    else:
        base_comp = "Competitive salary"

    if stage.lower() in ["startup", "early-stage"]:
        if hr_budget and "equity" not in hr_budget.lower():
            base_comp += " plus equity package"

    if stage.lower() == "enterprise":
        benefits = " with comprehensive benefits"
    else:
        benefits = " with startup benefits (health, dental, unlimited PTO)"

    comp_note = (base_comp + benefits).strip()
    if comp_note and not comp_note[0].isupper():
        comp_note = comp_note[0].upper() + comp_note[1:]
    return comp_note


def _contains_salary_range(budget_str: str) -> bool:
    """
    Check if string contains a salary range like 50k–70k, $50k-$70k, 50000-70000, etc.
    Accepts en dash/em dash/hyphen, optional $, optional 'k/K'.
    """
    if not budget_str:
        return False
    s = budget_str.strip()
    s = s.replace("–", "-").replace("—", "-")
    pattern = r'(?<![\w$])\$?\s*\d{2,3}(?:,\d{3})?(?:[kK])?\s*-\s*\$?\s*\d{2,3}(?:,\d{3})?(?:[kK])?(?![\w])'
    return bool(re.search(pattern, s))


def _contains_single_target(budget_str: str) -> bool:
    """
    Check if string contains a single salary like 50k or $70,000.
    """
    if not budget_str:
        return False
    s = budget_str.strip()
    s = s.replace("–", "-").replace("—", "-")
    pattern_single = r'(?<![\w$])\$?\s*\d{2,3}(?:,\d{3})?(?:[kK])?(?![\w])'
    if "-" in s and _contains_salary_range(s):
        return False
    return bool(re.search(pattern_single, s))


def get_default_budget_for_role(role: str, seniority: str, location: str) -> str:
    """
    Get default budget range for a role based on seniority and location.
    This is used when no specific salary/budget is provided.
    """

    base_ranges = {
        "junior": {"min": 60, "max": 85},
        "mid": {"min": 80, "max": 120},
        "senior": {"min": 110, "max": 160},
        "lead": {"min": 140, "max": 200},
        "principal": {"min": 180, "max": 250}
    }


    location_multipliers = {
        "sf": 1.3, "san francisco": 1.3, "bay area": 1.3,
        "nyc": 1.25, "new york": 1.25,
        "seattle": 1.2,
        "remote": 1.0,
        "austin": 1.1, "denver": 1.1,
        "chicago": 1.1, "boston": 1.15
    }

    seniority_key = seniority.lower()
    if seniority_key not in base_ranges:
        seniority_key = "mid"  

    base_range = base_ranges[seniority_key]

    multiplier = 1.0
    location_lower = location.lower()
    for loc_key, mult in location_multipliers.items():
        if loc_key in location_lower:
            multiplier = mult
            break

    min_salary = int(base_range["min"] * multiplier)
    max_salary = int(base_range["max"] * multiplier)

    return f"${min_salary}k-${max_salary}k"
