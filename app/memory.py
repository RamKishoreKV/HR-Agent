"""Session memory management for the HR Hiring Copilot."""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from .schema import SessionState, AgentOutput


def get_session_id() -> str:
    """Generate a new session ID."""
    return str(uuid.uuid4())[:8]


def load_state(session_id: str) -> Optional[SessionState]:
    """
    Load session state from file.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        SessionState object if found, None otherwise
    """
    try:
        sessions_dir = Path("data/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = sessions_dir / f"{session_id}.json"
        
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return SessionState(**data)
        
        return None
        
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return None


def save_state(session_id: str, state: SessionState) -> bool:
    """
    Save session state to file.
    
    Args:
        session_id: Unique session identifier
        state: SessionState object to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        sessions_dir = Path("data/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = sessions_dir / f"{session_id}.json"
        
        # Update timestamp
        state.timestamp = datetime.now().isoformat()
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")
        return False


def clear_session(session_id: str) -> bool:
    """
    Clear/delete a session file.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        sessions_dir = Path("data/sessions")
        session_file = sessions_dir / f"{session_id}.json"
        
        if session_file.exists():
            session_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"Error clearing session {session_id}: {e}")
        return False


def get_overrides(session_id: str) -> Dict[str, Any]:
    """
    Get override values from the last session state.
    This allows users to reuse previous inputs.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Dictionary of override values
    """
    state = load_state(session_id)
    
    if state and state.extraction:
        return {
            "timeline_weeks": state.extraction.timeline_weeks,
            "budget_range": state.extraction.budget_range,
            "seniority": state.extraction.seniority,
            "employment_type": state.extraction.employment_type,
            "location": state.extraction.location,
            "key_skills": state.extraction.key_skills,
            "company_stage": state.extraction.company_stage
        }
    
    return {}


def record_run(meta_dict: Dict[str, Any]) -> bool:
    """
    Record analytics data to JSONL file.
    
    Args:
        meta_dict: Dictionary containing run metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            **meta_dict
        }
        
        
        analytics_file = data_dir / "analytics.jsonl"
        
        with open(analytics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        return True
        
    except Exception as e:
        print(f"Error recording analytics: {e}")
        return False


def get_analytics_summary() -> Dict[str, Any]:
    """
    Get a summary of analytics data for display.
    
    Returns:
        Dictionary with analytics summary
    """
    try:
        analytics_file = Path("data/analytics.jsonl")
        
        if not analytics_file.exists():
            return {"total_runs": 0, "providers_used": [], "common_roles": []}
        
        records = []
        with open(analytics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        
        total_runs = len(records)
        providers = set()
        all_roles = []
        
        for record in records:
            if "provider" in record:
                providers.add(record["provider"])
            if "roles" in record:
                all_roles.extend(record["roles"])
        
        role_counts = {}
        for role in all_roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        
    
        common_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_runs": total_runs,
            "providers_used": list(providers),
            "common_roles": [{"role": role, "count": count} for role, count in common_roles],
            "last_run": records[-1]["timestamp"] if records else None
        }
        
    except Exception as e:
        print(f"Error getting analytics summary: {e}")
        return {"total_runs": 0, "providers_used": [], "common_roles": []}


def list_sessions() -> List[Dict[str, str]]:
    """
    List all available session files.
    
    Returns:
        List of session info dictionaries
    """
    try:
        sessions_dir = Path("data/sessions")
        
        if not sessions_dir.exists():
            return []
        
        sessions = []
        for session_file in sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": session_file.stem,
                        "timestamp": data.get("timestamp", ""),
                        "last_input": data.get("last_input", "")[:100] + "..." if len(data.get("last_input", "")) > 100 else data.get("last_input", "")
                    })
            except Exception:
                continue  
        
     
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return sessions
        
    except Exception as e:
        print(f"Error listing sessions: {e}")
        return []




def get_prefs() -> Dict[str, Any]:
    """Load persisted user preferences from data/prefs.json."""
    try:
        prefs_file = Path("data/prefs.json")
        if prefs_file.exists():
            with open(prefs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading prefs: {e}")
    return {}


def save_prefs(prefs: Dict[str, Any]) -> bool:
    """Persist user preferences to data/prefs.json."""
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        prefs_file = data_dir / "prefs.json"
        with open(prefs_file, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving prefs: {e}")
        return False