"""Pydantic schemas for the HR Hiring Copilot."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    """A job description for a role."""
    title: str = Field(..., description="Job title")
    summary: str = Field(..., description="Brief job summary")
    responsibilities: List[str] = Field(..., description="List of 5-9 key responsibilities")
    requirements: List[str] = Field(..., description="List of 5-9 requirements")
    compensation_note: str = Field(..., description="Compensation information")


class HiringPlan(BaseModel):
    """A hiring plan with steps and timeline."""
    steps: List[str] = Field(..., description="List of hiring process steps")
    owners: List[str] = Field(..., description="List of owners for each step")
    timeline_weeks: int = Field(..., description="Total timeline in weeks")


class AgentOutput(BaseModel):
    """Complete output from the HR agent."""
    clarifications_asked: List[str] = Field(default_factory=list, description="Questions asked for clarification")
    jd_drafts: List[JobDescription] = Field(default_factory=list, description="Generated job descriptions")
    hiring_plan: Optional[HiringPlan] = Field(None, description="Generated hiring plan")
    raw_markdown: str = Field("", description="Complete output in markdown format")
    raw_json: Dict[str, Any] = Field(default_factory=dict, description="Complete output in JSON format")


class InterviewKit(BaseModel):
    """Interview kit including screening questions, evaluation rubric, scorecard, and outreach."""
    screening_questions: List[str] = Field(default_factory=list, description="Suggested screening and interview questions")
    screening_guide: List[Dict[str, str]] = Field(default_factory=list, description="List of {question, what_to_evaluate} pairs for initial screening")
    evaluation_rubric: List[str] = Field(default_factory=list, description="Evaluation rubric criteria or guidelines")
    scorecard: List[str] = Field(default_factory=list, description="Scorecard fields for interviewer feedback")
    outreach_subject: str = Field("", description="Subject line for outreach email")
    outreach_body: str = Field("", description="Email body template for candidate outreach")


class SlotExtraction(BaseModel):
    """Extracted slot values from user input."""
    company_stage: str = Field("startup", description="Company stage")
    timeline_weeks: int = Field(8, description="Hiring timeline in weeks")
    budget_range: str = Field("competitive", description="Budget range")
    seniority: str = Field("mid", description="Seniority level")
    employment_type: str = Field("full-time", description="Employment type")
    location: str = Field("remote", description="Location preference")
    key_skills: List[str] = Field(default_factory=list, description="Key skills required")
    roles: List[str] = Field(default_factory=list, description="Roles to hire for")


class SessionState(BaseModel):
    """Session state for memory management."""
    session_id: str
    last_input: str = ""
    last_output: Optional[AgentOutput] = None
    extraction: Optional[SlotExtraction] = None
    timestamp: str = ""
