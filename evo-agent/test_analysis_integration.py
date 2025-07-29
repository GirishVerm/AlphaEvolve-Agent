#!/usr/bin/env python3
"""
Test Analysis Engine Integration
===============================
Simple test to verify the analysis engine works with the guided agent.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guided_agent import GuidedAgent, AgentConfig
from analysis_engine import AnalysisEngine, Recommendation

async def test_analysis_integration():
    """Test the analysis engine integration."""
    print("🧪 TESTING ANALYSIS ENGINE INTEGRATION")
    print("=" * 50)
    
    # Create agent with analysis engine
    config = AgentConfig(max_cost=5.0, max_generations=2)
    agent = GuidedAgent(config)
    
    print("✅ Agent created with analysis engine")
    print(f"📊 Analysis engine: {type(agent.analysis_engine).__name__}")
    
    # Test default recommendation
    print("\n🔄 Testing default recommendation...")
    default_rec = agent.analysis_engine._generate_default_recommendation()
    print(f"✅ Default recommendation generated:")
    print(f"   Should continue: {default_rec.should_continue}")
    print(f"   Confidence: {default_rec.confidence:.1%}")
    print(f"   Suggested feedback: {default_rec.suggested_feedback}")
    
    # Test recommendation display
    print("\n🔄 Testing recommendation display...")
    await agent._show_recommendations(default_rec)
    
    print("\n✅ Analysis engine integration test complete!")

if __name__ == "__main__":
    try:
        asyncio.run(test_analysis_integration())
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 