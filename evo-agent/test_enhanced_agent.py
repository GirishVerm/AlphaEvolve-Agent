#!/usr/bin/env python3
"""
Test Enhanced Agent Features
===========================

This script demonstrates the enhanced evolution tracking features.

Usage: python3 test_enhanced_agent.py
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guided_agent import GuidedAgent, AgentConfig

async def test_enhanced_features():
    """Test the enhanced evolution tracking features."""
    print("🧪 TESTING ENHANCED EVOLUTION FEATURES")
    print("=" * 50)
    
    # Initialize agent with smaller config for testing
    config = AgentConfig(max_cost=5.0, evolution_frequency=1, max_generations=2)
    agent = GuidedAgent(config)
    
    # Set a simple task
    from guided_agent import TaskSpec
    task = TaskSpec(
        task_name="Simple Calculator",
        description="Create a simple calculator function",
        requirements=["Add", "subtract", "error handling"],
        success_criteria=["Works correctly", "handles errors"]
    )
    
    agent.task = task
    
    print("✅ Agent initialized with enhanced tracking")
    print(f"📝 Initial prompts: {len(agent.initial_prompts)}")
    print(f"🛠️ Initial tools: {len(agent.initial_tools)}")
    print(f"🧠 Initial memory: {len(agent.initial_memory)}")
    
    # Generate initial code
    print("\n💻 Generating initial code...")
    initial_code = await agent.generate_initial_code()
    agent.initial_code = initial_code
    print(f"✅ Initial code stored: {len(initial_code)} characters")
    
    # Run one evolution cycle
    print("\n🔄 Running evolution cycle...")
    improved_code = await agent.improve_code(initial_code, "Add better error handling")
    
    # Evolve agent components
    print("\n🔄 Evolving agent components...")
    await agent.evolve_agent_components()
    
    # Show evolution summary
    print("\n📊 Showing evolution summary...")
    await agent._show_evolution_summary()
    
    print("\n✅ Enhanced features test complete!")

if __name__ == "__main__":
    try:
        asyncio.run(test_enhanced_features())
    except Exception as e:
        print(f"❌ Test error: {e}") 