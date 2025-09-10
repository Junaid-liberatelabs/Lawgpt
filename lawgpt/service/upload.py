"""
Upload script for indexing legal cases to Qdrant.

This script loads the bd_legal_cases_complete.json file and uploads each case
as a collection point to the Qdrant vector database.
"""

import sys
import os
import time
import json
from lawgpt.data_pipeline.rag_case_pipeline import CaseRAGPipeline


def main():
    """Main function to upload legal cases to Qdrant"""
    
    # Optional start index to resume from (0-based). Can be passed as --start N
    start_index = 4450
    if len(sys.argv) > 1:
        # very small arg parser to keep changes minimal
        args = sys.argv[1:]
        if args and args[0] in ['--help', '-h', 'help']:
            show_help()
            return 0
        if '--start' in args:
            try:
                idx = args.index('--start')
                start_index = int(args[idx + 1])
            except Exception:
                print("❌ Invalid --start value. Use an integer (0-based). Example: --start 4450 for case 4451.")
                return 1
        else:
            # support --start=NN form
            for a in args:
                if a.startswith('--start='):
                    try:
                        start_index = int(a.split('=', 1)[1])
                    except Exception:
                        print("❌ Invalid --start value. Use an integer (0-based). Example: --start=4450 for case 4451.")
                        return 1

    # Path to the JSON file
    json_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bd_legal_cases_complete.json")
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        print("Please ensure the bd_legal_cases_complete.json file exists in the data directory.")
        return 1
    
    try:
        print("🚀 Starting legal cases upload to Qdrant...")
        print(f"📁 Source file: {json_file_path}")
        
        # Initialize the RAG pipeline
        print("⚙️  Initializing RAG pipeline...")
        pipeline = CaseRAGPipeline()
        
        # Get collection info before upload
        collection_info = pipeline.get_collection_info()
        print(f"📊 Collection '{collection_info.get('name', 'Unknown')}' status: {collection_info.get('status', 'Unknown')}")
        print(f"📈 Current vectors count: {collection_info.get('vectors_count') or 0}")
        
        # Check how many cases we're about to upload
        print("📋 Analyzing JSON file...")
        with open(json_file_path, 'r', encoding='utf-8') as file:
            cases_data = json.load(file)
        total_cases = len(cases_data)
        print(f"📊 Found {total_cases} legal cases to upload")
        
        # Ask for confirmation if collection already has data
        vectors_count = collection_info.get("vectors_count") or 0
        if vectors_count > 0:
            response = input(f"\n⚠️  Collection already contains {vectors_count} vectors. Continue adding {total_cases} more? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Upload cancelled by user.")
                return 0
        
        # Start upload process
        if start_index and start_index > 0:
            print(f"\n🔄 Resuming upload from case {start_index + 1} of {total_cases}...")
        else:
            print(f"\n🔄 Starting upload of {total_cases} cases...")
        print("⏳ This may take a while for large datasets...")
        start_time = time.time()
        
        # Upload cases with detailed progress tracking
        print("🚀 Starting batch processing with detailed progress...")
        success = pipeline.add_cases(json_file_path, batch_size=25, verbose=True, start_index=start_index)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate performance metrics
        if elapsed_time > 0:
            cases_per_second = total_cases / elapsed_time
            print(f"⚡ Processing rate: {cases_per_second:.1f} cases/second")
        
        if success:
            # Get updated collection info
            updated_info = pipeline.get_collection_info()
            print(f"\n✅ Upload completed successfully!")
            print(f"⏱️  Time taken: {elapsed_time:.2f} seconds")
            print(f"📈 Total vectors in collection: {updated_info.get('vectors_count') or 0}")
            print(f"📊 Total points in collection: {updated_info.get('points_count') or 0}")
            
            # Test search functionality
            print("\n🔍 Testing search functionality...")
            test_results = pipeline.search_by_text("administrative tribunal", limit=2)
            if test_results:
                print(f"✅ Search test successful! Found {len(test_results)} results.")
                print(f"🎯 Top result: {test_results[0]['payload'].get('case_title', 'N/A')[:80]}...")
            else:
                print("⚠️  Search test returned no results.")
            
            return 0
        else:
            print(f"\n❌ Upload failed after {elapsed_time:.2f} seconds!")
            print("🔍 Check the logs above for error details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Upload interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during upload: {e}")
        return 1


def show_help():
    """Show help information"""
    print("""
📚 Legal Cases Upload Script

This script uploads legal cases from bd_legal_cases_complete.json to Qdrant.

Usage:
    python upload.py                      # Upload cases to Qdrant
    python upload.py --start 4450         # Resume from case 4451 (0-based index)
    python upload.py --start=4450         # Same as above
    python upload.py --help               # Show this help message

Prerequisites:
    1. Qdrant server running (default: http://localhost:6333)
    2. GOOGLE_API_KEY set in environment or .env file
    3. bd_legal_cases_complete.json file in ../data/ directory

What this script does:
    • Loads legal cases from JSON file
    • Creates embeddings using Google Gemini
    • Uploads each case as a vector point to Qdrant
    • Tests the search functionality after upload

Collection Details:
    • Collection name: bd_legal_cases (from config)
    • Vector dimension: 768 (Gemini text-embedding-004)
    • Distance metric: Cosine similarity
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)
    
    exit_code = main()
    sys.exit(exit_code)
