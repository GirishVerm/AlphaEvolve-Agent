#!/usr/bin/env python3
"""
Simple test to verify Azure OpenAI LLM is working.
"""
import asyncio
from llm_interface import LLMInterface, LLMConfig

async def test_llm():
    """Test the LLM interface."""
    print("Testing Azure OpenAI LLM...")
    
    # Initialize LLM
    config = LLMConfig()
    llm = LLMInterface(config)
    
    # Test simple generation
    try:
        response = await llm.generate(
            "Write a simple Python function that adds two numbers.",
            system_message="You are a helpful programming assistant."
        )
        print("✅ LLM Response:")
        print(response)
        print("\n✅ LLM is working correctly!")
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm()) 