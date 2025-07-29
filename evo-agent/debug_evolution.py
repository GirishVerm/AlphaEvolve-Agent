#!/usr/bin/env python3
"""
Debug script to see what the LLM is actually generating.
"""
import asyncio
import re
from llm_interface import LLMInterface, LLMConfig

async def debug_llm_response():
    """Debug what the LLM is actually generating."""
    print("🔍 DEBUGGING LLM RESPONSE")
    print("="*60)
    
    # Initialize LLM
    llm_config = LLMConfig()
    llm = LLMInterface(llm_config)
    
    # Test the exact prompt we're using
    baseline_code = '''
def convert_markdown_to_html(text):
    """Basic markdown to HTML converter."""
    if not text:
        return ""
    
    # Simple conversions
    html = text
    html = html.replace("# ", "<h1>").replace("\\n", "</h1>\\n")
    html = html.replace("**", "<strong>").replace("**", "</strong>")
    html = html.replace("*", "<em>").replace("*", "</em>")
    
    return html
'''
    
    mutation_prompt = f"""
    Improve this markdown-to-HTML converter within the EVOLVE-BLOCK:
    
    EVOLVE-BLOCK START
    {baseline_code}
    EVOLVE-BLOCK END
    
    Requirements:
    - Improve correctness, performance, and robustness
    - Handle edge cases better
    - Optimize for speed (target: 50ms)
    - Add error handling and input validation
    - Support more markdown features (lists, links, etc.)
    
    Provide only the improved code within EVOLVE-BLOCK:
    """
    
    print("📤 Sending prompt to LLM...")
    print("="*60)
    print(mutation_prompt)
    print("="*60)
    
    try:
        response = await llm.generate(mutation_prompt)
        
        print("\n📥 LLM RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)
        
        # Try to extract EVOLVE-BLOCK
        evolve_pattern = r'EVOLVE-BLOCK START\s*(.*?)\s*EVOLVE-BLOCK END'
        match = re.search(evolve_pattern, response, re.DOTALL)
        
        if match:
            evolved_code = match.group(1).strip()
            print(f"\n✅ EXTRACTED CODE:")
            print("="*60)
            print(evolved_code)
            print("="*60)
        else:
            print(f"\n❌ NO EVOLVE-BLOCK FOUND")
            print("Response doesn't contain EVOLVE-BLOCK markers")
            
            # Try to find any code-like content
            code_blocks = re.findall(r'```.*?\n(.*?)```', response, re.DOTALL)
            if code_blocks:
                print(f"\n🔍 FOUND {len(code_blocks)} CODE BLOCKS:")
                for i, block in enumerate(code_blocks):
                    print(f"\nCode Block {i+1}:")
                    print("="*30)
                    print(block.strip())
                    print("="*30)
            else:
                print("No code blocks found either")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_llm_response()) 