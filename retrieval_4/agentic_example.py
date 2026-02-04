"""
Example usage of Agentic Planner for Deterministic Temporal Retrieval
======================================================================

This demonstrates the agentic framework that performs deterministic,
temporally-aware retrieval with full reasoning transparency.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval_4.agentic_planner import AgenticPlanner
from retrieval_4.generator import LegalGenerator
from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
import os

def main():
    # Initialize agentic planner with Gemini
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    
    planner = AgenticPlanner(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        neo4j_database=NEO4J_DATABASE,
        gemini_api_key=gemini_api_key
    )
    
    # Initialize legal generator with Gemini
    generator = LegalGenerator(gemini_api_key=gemini_api_key)
    
    # Example queries with temporal constraints
    # queries = [
    #     {
    #         'query': "What did Section 24 of Act No. 5 of 2025 say in 2020?",
    #         'target_date': '2020-01-01'
    #     },
    #     {
    #         'query': "Find the provisions in Chapter 3 of the Bribery Act as of 2015",
    #         'target_date': '2015-01-01'
    #     },
    #     {
    #         'query': "What is the current law regarding bank fraud?",
    #         'target_date': None  # Will use current date
    #     }
    # ]
    queries = [
        {
            'query': "What did Section 24 of Act No. 5 of 2025 say in 2020?",
            'target_date': '2020-01-01'
        }
    ]

    for query_info in queries:
        query = query_info['query']
        target_date = query_info.get('target_date')
        
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        if target_date:
            print(f"Target Date: {target_date}")
        print("=" * 80)
        
        # Perform agentic retrieval
        result = planner.retrieve(query, target_date=target_date, top_k=5)
        
        # Generate legal answer
        print("\n" + "-" * 80)
        print("GENERATING ANSWER...")
        print("-" * 80)
        generation_result = generator.generate(result, query)
        
        # Print generated answer
        print("\n" + "=" * 80)
        print("LEGAL ANSWER:")
        print("=" * 80)
        print(generation_result['answer'])
        
        # Print cited sources
        if generation_result.get('sources'):
            print("\n" + "-" * 80)
            print("CITED SOURCES:")
            print("-" * 80)
            for i, source in enumerate(generation_result['sources'], 1):
                if isinstance(source, dict):
                    print(f"  {i}. {source.get('display_title')} [{source.get('raw_urn')}]")
                else:
                    print(f"  {i}. {source}")
        
        # Print temporal warnings
        if generation_result.get('temporal_warnings'):
            print("\n" + "-" * 80)
            print("TEMPORAL WARNINGS:")
            print("-" * 80)
            for warning in generation_result['temporal_warnings']:
                print(f"  ⚠️  {warning}")
        
        # Print summary (includes Lineage of Truth)
        print("\n" + result['summary'])
        
        # Print Lineage of Truth
        if result.get('lineage_of_truth'):
            print("\n" + "-" * 80)
            print("LINEAGE OF TRUTH:")
            print("-" * 80)
            for i, lineage in enumerate(result['lineage_of_truth'], 1):
                print(f"  {i}. {lineage}")
        
        # Print temporal collisions/drift events
        if result.get('temporal_collisions'):
            print("\n" + "-" * 80)
            print("TEMPORAL EVENTS DETECTED:")
            print("-" * 80)
            for collision in result['temporal_collisions']:
                collision_type = collision.get('type', 'collision')
                
                if collision_type == 'temporal_drift':
                    # New temporal drift format
                    print(f"\n⚠ TEMPORAL DRIFT:")
                    print(f"  URN: {collision.get('urn', 'unknown')}")
                    print(f"  Semantic Score: {collision.get('semantic_score', 0.0):.3f}")
                    print(f"  Temporal Score: {collision.get('temporal_score', 0.0):.3f}")
                    print(f"  Combined Score: {collision.get('final_score', 0.0):.3f}")
                    print(f"  Message: {collision.get('message', 'N/A')}")
                elif collision_type == 'temporal_collision':
                    # Old temporal collision format (backwards compatible)
                    print(f"\n⚠ TEMPORAL COLLISION:")
                    print(f"  Case URN: {collision.get('case_urn', 'unknown')}")
                    print(f"  Statute URN: {collision.get('statute_urn', 'unknown')}")
                    print(f"  Similarity: {collision.get('similarity', 0.0):.2%}")
                    print(f"  Priority: {collision.get('priority', 'statute').upper()}")
                    print(f"  Reasoning: {collision.get('reasoning', 'N/A')}")
                else:
                    # Generic fallback
                    print(f"\n⚠ TEMPORAL EVENT: {collision}")
    
    # Close connection
    planner.close()
    print("\n" + "=" * 80)
    print("Agentic retrieval complete!")

if __name__ == "__main__":
    main()
