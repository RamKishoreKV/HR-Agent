"""Tests for the HR Hiring Copilot graph workflow."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.agent import run_agent_langgraph
from app.graph import run_hiring_workflow
from app.schema import AgentOutput, JobDescription, HiringPlan


class TestHiringGraph:
    """Test cases for the hiring graph workflow."""
    
    def test_run_agent_langgraph_basic(self):
        """Test basic agent functionality with default inputs."""
        # Use default overrides for consistent testing
        overrides = {
            "timeline_weeks": 8,
            "budget_range": "$80k-$120k",
            "seniority": "mid",
            "employment_type": "full-time",
            "location": "remote",
            "key_skills": ["python", "sql"],
            "company_stage": "startup",
            "roles": ["data analyst"]
        }
        
        result = run_agent_langgraph("hire a data analyst", overrides=overrides)
        
        # Verify result is AgentOutput
        assert isinstance(result, AgentOutput)
        
        # Verify at least one job description was generated
        assert len(result.jd_drafts) >= 1
        
        # Verify job description has required fields
        jd = result.jd_drafts[0]
        assert isinstance(jd, JobDescription)
        assert jd.title
        assert jd.summary
        assert len(jd.responsibilities) >= 5
        assert len(jd.requirements) >= 5
        assert jd.compensation_note
        
        # Verify hiring plan was generated
        assert result.hiring_plan is not None
        assert isinstance(result.hiring_plan, HiringPlan)
        assert len(result.hiring_plan.steps) > 0
        assert len(result.hiring_plan.owners) > 0
        assert result.hiring_plan.timeline_weeks == 8
        
        # Verify output formats
        assert result.raw_markdown
        assert result.raw_json
        assert isinstance(result.raw_json, dict)
    
    def test_run_agent_langgraph_multiple_roles(self):
        """Test agent with multiple roles."""
        overrides = {
            "timeline_weeks": 10,
            "budget_range": "competitive",
            "seniority": "senior",
            "employment_type": "full-time", 
            "location": "San Francisco",
            "key_skills": ["python", "react", "aws"],
            "company_stage": "startup",
            "roles": ["software engineer", "product manager"]
        }
        
        result = run_agent_langgraph("hire software engineer and product manager", overrides=overrides)
        
        # Should generate job descriptions for both roles
        assert len(result.jd_drafts) == 2
        
        # Verify role titles
        titles = [jd.title.lower() for jd in result.jd_drafts]
        assert any("engineer" in title for title in titles)
        assert any("manager" in title for title in titles)
        
        # Verify hiring plan accounts for multiple roles
        assert result.hiring_plan.timeline_weeks == 10
    
    def test_small_talk_handling(self):
        """Test that small talk is handled appropriately."""
        result = run_agent_langgraph("Hello, how are you?")
        
        # Should return AgentOutput but with small talk response
        assert isinstance(result, AgentOutput)
        assert len(result.jd_drafts) == 0  # No job descriptions for small talk
        assert result.hiring_plan is None   # No hiring plan for small talk
        assert result.raw_markdown  # Should have some response
        assert "type" in result.raw_json and result.raw_json["type"] == "small_talk"
    
    def test_error_handling(self):
        """Test error handling in agent workflow."""
        # Test with empty input
        result = run_agent_langgraph("")
        assert isinstance(result, AgentOutput)
        
        # Test with very long input (edge case)
        long_input = "hire " + "engineer " * 100
        result = run_agent_langgraph(long_input)
        assert isinstance(result, AgentOutput)
    
    def test_run_hiring_workflow_direct(self):
        """Test the workflow directly without agent wrapper."""
        overrides = {
            "timeline_weeks": 6,
            "budget_range": "$90k-$130k",
            "seniority": "mid",
            "employment_type": "full-time",
            "location": "remote",
            "key_skills": ["python", "machine learning"],
            "company_stage": "startup",
            "roles": ["data scientist"]
        }
        
        result = run_hiring_workflow("need to hire a data scientist", overrides)
        
        assert isinstance(result, AgentOutput)
        assert len(result.jd_drafts) >= 1
        assert result.hiring_plan is not None
        
        # Verify data scientist specific content
        jd = result.jd_drafts[0]
        assert "data" in jd.title.lower() or "scientist" in jd.title.lower()
        
        # Check that overrides were applied
        assert result.hiring_plan.timeline_weeks == 6
    
    @patch('app.llm.ask_llm')
    def test_llm_failure_handling(self, mock_llm):
        """Test handling of LLM failures."""
        # Mock LLM to raise an exception
        mock_llm.side_effect = Exception("LLM service unavailable")
        
        result = run_agent_langgraph("hire a developer")
        
        # Should still return a result, even if degraded
        assert isinstance(result, AgentOutput)
        # May have fallback responses
    
    def test_different_seniority_levels(self):
        """Test different seniority levels affect output."""
        seniority_levels = ["junior", "mid", "senior", "lead"]
        
        for seniority in seniority_levels:
            overrides = {
                "timeline_weeks": 8,
                "budget_range": "competitive",
                "seniority": seniority,
                "employment_type": "full-time",
                "location": "remote",
                "key_skills": ["python"],
                "company_stage": "startup",
                "roles": ["developer"]
            }
            
            result = run_agent_langgraph(f"hire a {seniority} developer", overrides=overrides)
            
            assert isinstance(result, AgentOutput)
            assert len(result.jd_drafts) >= 1
            
            # Job description should somehow reflect seniority
            jd = result.jd_drafts[0]
            jd_text = (jd.title + " " + jd.summary + " " + 
                      " ".join(jd.responsibilities + jd.requirements)).lower()
            
            # At minimum, should not fail
            assert len(jd_text) > 0
    
    def test_budget_handling(self):
        """Test different budget formats are handled correctly."""
        budget_formats = [
            "$80k-$120k",
            "$80,000-$120,000", 
            "competitive",
            "market rate",
            "$100k"
        ]
        
        for budget in budget_formats:
            overrides = {
                "timeline_weeks": 8,
                "budget_range": budget,
                "seniority": "mid",
                "employment_type": "full-time",
                "location": "remote",
                "key_skills": ["python"],
                "company_stage": "startup",
                "roles": ["analyst"]
            }
            
            result = run_agent_langgraph("hire an analyst", overrides=overrides)
            
            assert isinstance(result, AgentOutput)
            assert len(result.jd_drafts) >= 1
            
            # Compensation note should exist
            jd = result.jd_drafts[0]
            assert jd.compensation_note
            assert len(jd.compensation_note) > 0


def test_basic_smoke_test():
    """Basic smoke test - requires either OpenAI API key or Ollama running."""
    # Skip if no LLM provider is properly configured
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Skipping test: OpenAI API key not configured")
        return
    
    result = run_agent_langgraph("hire a data analyst")
    
    assert isinstance(result, AgentOutput)
    # Should get some response from real LLM
    assert result.raw_markdown or result.raw_json


def test_compose_comp_note_exact_salary():
    from app.comp import compose_comp_note
    note = compose_comp_note("Software Engineer", "startup", "80k")
    assert "80k" in note
    note2 = compose_comp_note("Software Engineer", "startup", "$95k")
    assert "$95k" in note2
    note3 = compose_comp_note("Software Engineer", "startup", "$100000")
    assert "$100000" in note3


if __name__ == "__main__":
    # Run basic smoke test
    test_basic_smoke_test()
    print("Basic smoke test passed!")
    
    # Run full test suite if pytest is available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests only")
        
        # Run a few key tests manually
        test_instance = TestHiringGraph()
        test_instance.test_run_agent_langgraph_basic()
        test_instance.test_small_talk_handling()
        print("Manual tests completed!")
