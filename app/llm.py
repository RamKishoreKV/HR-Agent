"""LLM provider interface with OpenAI and Ollama, with robust fallbacks."""

import os
import requests
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def ask_llm(prompt: str) -> str:
    """Ask the LLM based on configured provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        return _ask_openai(prompt)
    elif provider == "ollama":
        return _ask_ollama(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'openai' or 'ollama'.")


def test_llm_connection() -> Dict[str, Any]:
    """Test connections to both providers."""
    results: Dict[str, Dict[str, Any]] = {
        "openai": {"available": False, "error": None},
        "ollama": {"available": False, "error": None}
    }

    # OpenAI
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            client = OpenAI(api_key=api_key)
            oc = client.with_options(timeout=10)  # correct way to set timeout
            resp = oc.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'OpenAI connection test successful'"}],
                max_tokens=50
            )
            txt = resp.choices[0].message.content if resp and resp.choices else ""
            if txt:
                results["openai"]["available"] = True
                results["openai"]["response"] = txt
            else:
                results["openai"]["error"] = "No response received"
        else:
            results["openai"]["error"] = "API key not configured"
    except Exception as e:
        results["openai"]["error"] = str(e)

    # Ollama
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        r = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Say 'Ollama connection test successful'",
                "stream": False,
                "options": {"num_predict": 50}
            },
            timeout=10
        )
        if r.status_code == 200:
            text = r.json().get("response", "")
            if text:
                results["ollama"]["available"] = True
                results["ollama"]["response"] = text
            else:
                results["ollama"]["error"] = "No response received"
        else:
            results["ollama"]["error"] = f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        results["ollama"]["error"] = str(e)

    return results


def _ask_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)
    try:
        oc = client.with_options(timeout=30)
        resp = oc.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Error calling OpenAI: {str(e)}"


def _ask_ollama(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 2000}
            },
            timeout=60
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"
