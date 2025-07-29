#!/usr/bin/env python3
"""
Interactive Agent Demo
=====================

Demonstrates how to use the interactive evolutionary agent.

Usage: python3 demo_usage.py
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive_agent import InteractiveAgent, AgentConfig, TaskSpec

async def demo():
    """Run a demo of the interactive agent."""
    print("🎭 INTERACTIVE AGENT DEMO")
    print("=" * 50)
    
    # Initialize agent
    config = AgentConfig(max_cost=10.0, evolution_frequency=2, population_size=3, max_generations=5)
    agent = InteractiveAgent(config)
    
    # Set a sample task
    task = TaskSpec(
        task_name="Markdown to HTML Converter",
        description="Create a function that converts markdown text to HTML",
        requirements=["Handle basic markdown syntax", "Support headers, bold, italic", "Handle links and lists"],
        success_criteria=["Passes basic tests", "Handles edge cases", "Produces valid HTML"]
    )
    
    await agent.set_task(task)
    
    # Analyze the task
    print("\n📝 Analyzing task...")
    analysis = await agent.analyze_task()
    print(f"Analysis: {analysis[:200]}...")
    
    # Generate initial code
    print("\n💻 Generating initial code...")
    initial_code = await agent.generate_initial_code()
    print(f"Initial code:\n{initial_code}")
    
    # Evaluate the code
    print("\n📊 Evaluating code...")
    evaluation = await agent.evaluate_code(initial_code)
    print(f"Evaluation: {evaluation}")
    
    # Improve the code
    print("\n🔄 Improving code...")
    improved_code = await agent.run_evolution_cycle(initial_code, "Add better error handling")
    print(f"Improved code:\n{improved_code}")
    
    # Execute the task
    print("\n🎯 Executing the evolved task...")
    result = await agent.execute_task(improved_code)
    
    if result["success"]:
        print(f"\n✅ TASK EXECUTION COMPLETE!")
        print(f"Function: {result['function_name']}")
        print(f"Success Rate: {result['success_rate']:.1%}")
        print(f"\n📝 FINAL WORKING CODE:")
        print("=" * 50)
        print(result["code"])
        print("=" * 50)
    else:
        print(f"\n❌ TASK EXECUTION FAILED: {result['error']}")
    
    # Show agent status
    print("\n🤖 Agent Status:")
    cost_stats = agent.llm.get_cost_stats()
    print(f"Total cost: ${cost_stats.get('total_cost', 0):.4f}")
    print(f"Total requests: {cost_stats.get('total_requests', 0)}")
    print(f"Generations: {agent.generation}")
    
    print("\n✅ Demo complete! Run 'python3 run_agent.py' for interactive mode.")

if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt") 