#!/usr/bin/env python3
"""
Interactive Agent Launcher
=========================

Simple launcher for the interactive evolutionary agent.

Usage: python3 run_agent.py
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive_agent import main

if __name__ == "__main__":
    print("🚀 Starting Interactive Evolutionary Agent...")
    print("=" * 50)
    print("This agent can:")
    print("• Accept tasks and specifications")
    print("• Build and evolve code")
    print("• Evolve its own prompts, tools, and memory")
    print("• Interact with you through the terminal")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt") 