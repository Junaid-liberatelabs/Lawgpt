"""
Example usage of the CaseRAGPipeline for legal case retrieval.

This script demonstrates how to:
1. Initialize the RAG pipeline
2. Load and index legal cases from JSON
3. Search for similar cases using text queries
"""

import os
from rag_case_pipeline import CaseRAGPipeline


def main():
    """Main function demonstrating the RAG pipeline usage"""
    
    # Initialize the pipeline (uses config automatically)
    pipeline = CaseRAGPipeline()
    
    # Path to the legal cases JSON file
    json_file_path = "../data/bd_legal_cases_complete.json"
    
    try:
        # Check if data is already indexed
        collection_info = pipeline.get_collection_info()
        print(f"Collection info: {collection_info}")
        
        # If collection is empty, load and index the cases
        if collection_info.get("vectors_count", 0) == 0:
            print("Loading and indexing legal cases...")
            success = pipeline.add_cases(json_file_path)
            if success:
                print("Legal cases indexed successfully!")
            else:
                print("Failed to index legal cases!")
                return 1
        else:
            print(f"Collection already contains {collection_info['vectors_count']} cases")
        
        # Example searches
       
        
        print("\n" + "="*50)
        print("EXAMPLE SEARCHES")
        print("="*50)
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            print("-" * 40)
            
            results = pipeline.search_by_text(query=query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    payload = result['payload']
                    print(f"\n{i}. {payload.get('case_title', 'N/A')}")
                    print(f"   Division: {payload.get('division', 'N/A')}")
                    print(f"   Law Act: {payload.get('law_act', 'N/A')}")
                    print(f"   Reference: {payload.get('reference', 'N/A')}")
                    print(f"   Similarity Score: {result['score']:.3f}")
                    print(f"   Details: {payload.get('case_details', 'N/A')[:200]}...")
            else:
                print("No similar cases found.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
