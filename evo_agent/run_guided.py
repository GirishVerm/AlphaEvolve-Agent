#!/usr/bin/env python3
"""
Guided Agent Launcher
=====================

Simple launcher for the guided evolutionary agent.

Usage: python3 run_guided.py
"""

import asyncio
import os
import sys
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guided_agent import main

if __name__ == "__main__":
    print("🚀 Starting Guided Evolutionary Agent...")
    print("=" * 50)
    print("This agent will guide you through:")
    print("• Setting up a task")
    print("• Analyzing the requirements")
    print("• Generating and improving code")
    print("• Evolving the agent itself")
    print("• Executing the final result")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt") 