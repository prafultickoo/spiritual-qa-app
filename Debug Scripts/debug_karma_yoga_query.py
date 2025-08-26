#!/usr/bin/env python3
"""
Debug script to trace step-by-step execution of "What is Karma Yoga" query.
Shows exact code invoked, what it does, and its output at each step.
"""

import os
import json
import sys
from dotenv import load_dotenv

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from answer_generator import AnswerGenerator

# Load environment variables
load_dotenv()

def debug_step(step_num, title, code_location):
    """Print step header"""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {title}")
    print(f"Code Location: {code_location}")
    print(f"{'='*80}")

def show_code_block(description, code):
    """Show the actual code being executed"""
    print(f"\n🔧 CODE INVOKED:")
    print(f"Description: {description}")
    print(f"```python")
    print(code)
    print(f"```")

def show_output(output_data):
    """Show the output of the code block"""
    print(f"\n📤 OUTPUT:")
    if isinstance(output_data, dict):
        print(json.dumps(output_data, indent=2, default=str))
    else:
        print(output_data)

def main():
    print("🕉️  DEBUG: Karma Yoga Query Step-by-Step Execution")
    print("Query: 'What is Karma Yoga'")
    print("Model: o3-mini")
    print("Reading Style: balanced")
    
    # STEP 1: Initialize Answer Generator
    debug_step(1, "Initialize Answer Generator", "answer_generator.py:__init__")
    
    show_code_block(
        "Creates AnswerGenerator instance with dual-source retrieval",
        """
generator = AnswerGenerator(
    vector_store_dir="./vector_store",
    embedding_model="openai", 
    llm_model="o3-mini",
    enable_dual_source=True
)
        """.strip()
    )
    
    try:
        generator = AnswerGenerator(
            vector_store_dir="./vector_store",
            embedding_model="openai",
            llm_model="o3-mini",
            enable_dual_source=True
        )
        
        init_output = {
            "retriever_type": type(generator.retriever).__name__,
            "dual_source_enabled": generator.enable_dual_source,
            "llm_model": generator.llm_model,
            "vector_store_dir": generator.retriever.vector_store_dir
        }
        show_output(init_output)
        
        # STEP 2: Call generate_answer method
        debug_step(2, "Call generate_answer Method", "answer_generator.py:generate_answer")
        
        show_code_block(
            "Main method that orchestrates the entire query processing",
            """
result = generator.generate_answer(
    query="What is Karma Yoga",
    k=5,
    use_mmr=True,
    diversity=0.3,
    reasoning_effort="medium"
)
            """.strip()
        )
        
        print("\n⏳ Executing generate_answer...")
        print("This will call multiple sub-steps internally...")
        
        # Execute with detailed tracing
        result = generator.generate_answer(
            query="What is Karma Yoga",
            k=5,
            use_mmr=True,
            diversity=0.3,
            reasoning_effort="medium"
        )
        
        # Show final result
        final_output = {
            "status": result.get("status"),
            "answer_length": len(result.get("answer", "")),
            "answer_preview": result.get("answer", "")[:200] + "..." if len(result.get("answer", "")) > 200 else result.get("answer", ""),
            "chunks_used": result.get("chunks_used", 0),
            "model": result.get("model"),
            "technique_used": result.get("technique_used")
        }
        show_output(final_output)
        
        print(f"\n🎯 COMPLETE ANSWER:")
        print(f"{'='*80}")
        print(result.get("answer", "No answer generated"))
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
