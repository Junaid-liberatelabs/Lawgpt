"""
Example usage of the LawRAGPipeline for law reference retrieval.

This script demonstrates how to:
1. Initialize the Law RAG pipeline
2. Load and index law references from JSON files
3. Search for similar law references using text queries
"""

import os
import glob
from lawgpt.data_pipeline.rag_law_pipeline import LawRAGPipeline


def main():
    """Main function demonstrating the Law RAG pipeline usage"""
    
    # Initialize the pipeline (uses config automatically)
    pipeline = LawRAGPipeline()
    
    # Get data directory path
    data_dir = "../data"
    
    # Find all law reference JSON files (excluding case files)
    json_pattern = os.path.join(data_dir, "*.json")
    all_json_files = glob.glob(json_pattern)
    law_json_files = [f for f in all_json_files if "bd_legal_cases_complete.json" not in f]
    
    try:
        # Check if data is already indexed
        collection_info = pipeline.get_collection_info()
        print(f"Collection info: {collection_info}")
        
        # If collection is empty, load and index the law references
        if collection_info.get("vectors_count", 0) == 0:
            if not law_json_files:
                print("No law reference JSON files found in data directory!")
                return 1
            
            print(f"Loading and indexing law references from {len(law_json_files)} files...")
            print("Files to process:")
            for i, file_path in enumerate(law_json_files, 1):
                print(f"  {i}. {os.path.basename(file_path)}")
            
            success = pipeline.add_multiple_law_files(law_json_files, batch_size=25, verbose=True)
            if success:
                print("Law references indexed successfully!")
            else:
                print("Failed to index law references!")
                return 1
        else:
            print(f"Collection already contains {collection_info['vectors_count']} law references")
        
        # Example searches
        search_queries = [
           "I have been arrested without a warrant. Is this legal, and what does the law say about it?"
           
        ]
        
        print("\n" + "="*50)
        print("EXAMPLE LAW REFERENCE SEARCHES")
        print("="*50)
        
        for query in search_queries:
            print(f"\nSearching for: '{query}'")
            print("-" * 40)
            
            results = pipeline.search_by_text(query=query, limit=5)
            
            if results:
                for i, result in enumerate(results, 1):
                    payload = result['payload']
                    print(f"\n{i}. {payload.get('part_section', 'N/A')}")
                    print(f"   Similarity Score: {result['score']:.3f}")
                    print(f"   Law Text: {payload.get('law_text', 'N/A')}...")
            else:
                print("No similar law references found.")
        
        # # Demonstrate specific legal concept searches
        # print("\n" + "="*50)
        # print("SPECIFIC LEGAL CONCEPT SEARCHES")
        # print("="*50)
        
        # specific_queries = [
        #     "power to make rules",
        #     "official gazette notification",
        #     "government authority",
        #     "legal proceedings",
        #     "penalty and punishment"
        # ]
        
        # for query in specific_queries:
        #     print(f"\nSearching for concept: '{query}'")
        #     print("-" * 40)
            
        #     results = pipeline.search_by_text(query=query, limit=2)
            
        #     if results:
        #         for i, result in enumerate(results, 1):
        #             payload = result['payload']
        #             print(f"\n{i}. Section: {payload.get('part_section', 'N/A')}")
        #             print(f"   Score: {result['score']:.3f}")
        #             law_text = payload.get('law_text', 'N/A')
        #             # Show more text for specific concept searches
        #             print(f"   Text: {law_text[:300]}{'...' if len(law_text) > 300 else ''}")
        #     else:
        #         print("No matching law references found.")
        
        # Show collection statistics
        print("\n" + "="*50)
        print("COLLECTION STATISTICS")
        print("="*50)
        final_info = pipeline.get_collection_info()
        print(f"Collection Name: {final_info.get('name', 'N/A')}")
        print(f"Total Vectors: {final_info.get('vectors_count', 0)}")
        print(f"Total Points: {final_info.get('points_count', 0)}")
        print(f"Status: {final_info.get('status', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
