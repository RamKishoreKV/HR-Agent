#!/usr/bin/env python3
"""Simple script to test LLM connections."""

import os
import sys
from dotenv import load_dotenv

# Add app directory to path
sys.path.append('.')

from app.llm import test_llm_connection

def main():
    load_dotenv()
    
    print("🔍 Testing LLM Connections...")
    print("=" * 50)
    print(f"Current LLM_PROVIDER setting: {os.getenv('LLM_PROVIDER', 'ollama (default)')}")
    
    results = test_llm_connection()
    
    # OpenAI Test
    print("\n🤖 OpenAI Test:")
    if results["openai"]["available"]:
        print("   ✅ STATUS: Working")
        print(f"   📝 RESPONSE: {results['openai']['response']}")
    else:
        print("   ❌ STATUS: Failed")
        print(f"   💥 ERROR: {results['openai']['error']}")
    
    # Ollama Test
    print("\n🦙 Ollama Test:")
    if results["ollama"]["available"]:
        print("   ✅ STATUS: Working")
        print(f"   📝 RESPONSE: {results['ollama']['response']}")
    else:
        print("   ❌ STATUS: Failed")
        print(f"   💥 ERROR: {results['ollama']['error']}")
    
    # Summary & Recommendations
    print("\n" + "=" * 50)
    working_providers = []
    if results["openai"]["available"]:
        working_providers.append("openai")
    if results["ollama"]["available"]:
        working_providers.append("ollama")
    
    if working_providers:
        if len(working_providers) == 1:
            provider = working_providers[0]
            print(f"💡 RECOMMENDATION: Set LLM_PROVIDER={provider} in your .env file")
            print(f"   Current setting: {os.getenv('LLM_PROVIDER', 'not set')}")
        else:
            print("💡 RECOMMENDATION: Both providers working! You can use either.")
            print(f"   Current setting: {os.getenv('LLM_PROVIDER', 'not set')}")
        
        print("\n🚀 Ready to run: streamlit run ui/streamlit_app.py")
    else:
        print("⚠️  WARNING: No providers working!")
        print("\n🔧 TROUBLESHOOTING:")
        print("   For OpenAI: Check your OPENAI_API_KEY in .env")
        print("   For Ollama: Make sure it's running (ollama serve)")
    
    print()

if __name__ == "__main__":
    main()
