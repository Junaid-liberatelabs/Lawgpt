"""
Upload script for indexing law references to Qdrant.

This script loads law reference JSON files and uploads each reference
as a collection point to the Qdrant vector database.
Supports multiple JSON files and batch processing.
"""

import sys
import os
import time
import json
import glob
from lawgpt.data_pipeline.rag_law_pipeline import LawRAGPipeline


def main():
    """Main function to upload law references to Qdrant"""
    
    # Variables for resume functionality
    # Set these values to resume from a specific point
    resume_file = None  # Set to None for normal upload, or filename.json to resume
    resume_file_index = 0  # Start from index 0 (or specific index for resume)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if args and args[0] in ['--help', '-h', 'help']:
            show_help()
            return 0
        
        # Parse resume arguments
        if '--resume' in args:
            try:
                idx = args.index('--resume')
                if idx + 1 < len(args):
                    resume_parts = args[idx + 1].split(':')
                    resume_file = resume_parts[0]
                    if len(resume_parts) > 1:
                        resume_file_index = int(resume_parts[1])
                else:
                    print("❌ Invalid --resume format. Use: --resume filename.json:index")
                    print("Example: --resume madokdrobbo_niyontron_ain_2018.json:25")
                    return 1
            except (ValueError, IndexError):
                print("❌ Invalid --resume format. Use: --resume filename.json:index")
                return 1
    
    # Default data directory path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # Find all JSON files in data directory (excluding case files)
    json_pattern = os.path.join(data_dir, "*.json")
    all_json_files = glob.glob(json_pattern)
    
    # Filter out the case file to only get law reference files
    law_json_files = [f for f in all_json_files if "bd_legal_cases_complete.json" not in f]
    
    if not law_json_files:
        print(f"❌ No law reference JSON files found in {data_dir}")
        print("Please ensure law reference JSON files exist in the data directory.")
        print("Note: bd_legal_cases_complete.json is excluded as it's for cases, not law references.")
        return 1
    
    # Handle resume functionality
    if resume_file:
        # Find the file to resume from
        resume_file_path = None
        for file_path in law_json_files:
            if os.path.basename(file_path) == resume_file:
                resume_file_path = file_path
                break
        
        if not resume_file_path:
            print(f"❌ Resume file '{resume_file}' not found in law reference files.")
            print("Available files:")
            for f in law_json_files:
                print(f"   - {os.path.basename(f)}")
            return 1
        
        # Reorder files to start from the resume file
        resume_index = law_json_files.index(resume_file_path)
        law_json_files = law_json_files[resume_index:]
        
        print(f"🔄 Resume mode: Starting from '{resume_file}' at index {resume_file_index}")
    else:
        resume_file_index = 0
    
    try:
        print("🚀 Starting law references upload to Qdrant...")
        print(f"📁 Data directory: {data_dir}")
        print(f"📄 Found {len(law_json_files)} law reference files:")
        for i, file_path in enumerate(law_json_files, 1):
            print(f"   {i}. {os.path.basename(file_path)}")
        
        # Initialize the RAG pipeline
        print("\n⚙️  Initializing Law RAG pipeline...")
        pipeline = LawRAGPipeline()
        
        # Get collection info before upload
        collection_info = pipeline.get_collection_info()
        print(f"📊 Collection '{collection_info.get('name', 'Unknown')}' status: {collection_info.get('status', 'Unknown')}")
        print(f"📈 Current vectors count: {collection_info.get('vectors_count') or 0}")
        
        # Count total law references across all files
        print("\n📋 Analyzing JSON files...")
        total_references = 0
        file_stats = []
        
        for json_file_path in law_json_files:
            try:
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    law_data = json.load(file)
                file_ref_count = len(law_data)
                total_references += file_ref_count
                file_stats.append((os.path.basename(json_file_path), file_ref_count))
                print(f"   📄 {os.path.basename(json_file_path)}: {file_ref_count} references")
            except Exception as e:
                print(f"   ❌ Error reading {os.path.basename(json_file_path)}: {e}")
                return 1
        
        print(f"\n📊 Total law references to upload: {total_references}")
        
        # Ask for confirmation if collection already has data
        vectors_count = collection_info.get("vectors_count") or 0
        if vectors_count > 0:
            response = input(f"\n⚠️  Collection already contains {vectors_count} vectors. Continue adding {total_references} more? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Upload cancelled by user.")
                return 0
        
        # Start upload process
        print(f"\n🔄 Starting upload of {total_references} law references from {len(law_json_files)} files...")
        print("⏳ This may take a while for large datasets...")
        start_time = time.time()
        
        # Upload law references with detailed progress tracking
        print("🚀 Starting batch processing with detailed progress...")
        
        # Handle resume functionality or normal processing
        if resume_file and resume_file_index > 0:
            # Process the first file (resume file) with start index
            first_file = law_json_files[0]
            remaining_files = law_json_files[1:]
            
            print(f"🔄 Resuming '{os.path.basename(first_file)}' from index {resume_file_index}...")
            success = pipeline.add_law_references(first_file, batch_size=25, verbose=True, start_index=resume_file_index)
            
            if not success:
                print(f"❌ Failed to resume processing of {os.path.basename(first_file)}")
                print(f"💡 To continue from next batch, try: --resume {os.path.basename(first_file)}:{resume_file_index + 25}")
                return 1
            
            # Process remaining files normally if any
            if remaining_files:
                print(f"\n🔄 Continuing with remaining {len(remaining_files)} files...")
                success = pipeline.add_multiple_law_files(remaining_files, batch_size=25, verbose=True)
                
                if not success:
                    print(f"❌ Failed to process remaining files")
                    # Find which file failed and suggest resume command
                    for i, file_path in enumerate(remaining_files):
                        filename = os.path.basename(file_path)
                        print(f"💡 To resume from {filename}, try: --resume {filename}:0")
                        break
                    return 1
        else:
            # Normal processing of all files
            success = pipeline.add_multiple_law_files(law_json_files, batch_size=25, verbose=True)
            
            if not success:
                print(f"❌ Upload failed!")
                print(f"💡 Common resume commands:")
                print(f"   --resume madokdrobbo_niyontron_ain_2018.json:25  # Resume from where it typically fails")
                print(f"   --resume madokdrobbo_niyontron_ain_2018.json:50  # Skip to next batch if still failing") 
                print(f"   --resume {os.path.basename(law_json_files[0])}:0  # Start over with first file")
                return 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate performance metrics
        if elapsed_time > 0:
            references_per_second = total_references / elapsed_time
            print(f"⚡ Processing rate: {references_per_second:.1f} law references/second")
        
        if success:
            # Get updated collection info
            updated_info = pipeline.get_collection_info()
            print(f"\n✅ Upload completed successfully!")
            print(f"⏱️  Time taken: {elapsed_time:.2f} seconds")
            print(f"📈 Total vectors in collection: {updated_info.get('vectors_count') or 0}")
            print(f"📊 Total points in collection: {updated_info.get('points_count') or 0}")
            
            # Show chunking statistics if available
            vectors_count = updated_info.get('vectors_count') or 0
            if vectors_count > total_references:
                chunks_created = vectors_count - total_references
                print(f"📄 Chunks created: {chunks_created} (from {total_references} law references)")
                print(f"📊 Average chunks per reference: {vectors_count / total_references:.1f}")
            
            # Test search functionality
            print("\n🔍 Testing search functionality...")
            test_results = pipeline.search_by_text("consumer protection", limit=2)
            if test_results:
                print(f"✅ Search test successful! Found {len(test_results)} results.")
                top_result = test_results[0]['payload']
                part_section = top_result.get('part_section', 'N/A')[:80]
                is_chunked = top_result.get('is_chunked', False)
                chunk_info = f" (chunk {top_result.get('chunk_index', 0) + 1}/{top_result.get('total_chunks', 1)})" if is_chunked else ""
                print(f"🎯 Top result: {part_section}...{chunk_info}")
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
📚 Law References Upload Script

This script uploads law references from JSON files in the data directory to Qdrant.
It automatically discovers all JSON files except bd_legal_cases_complete.json.

Usage:
    python upload_law.py                                    # Upload all law reference files
    python upload_law.py --resume filename.json:index      # Resume from specific file and index
    python upload_law.py --help                            # Show this help message

Resume Examples:
    --resume madokdrobbo_niyontron_ain_2018.json:25        # Resume from index 25 in that file
    --resume vokta_odhikar_songrokkhon_ain_2009_english.json:0  # Start from beginning of that file

Prerequisites:
    1. Qdrant server running (default: http://localhost:6333)
    2. GOOGLE_API_KEY set in environment or .env file
    3. Law reference JSON files in ../data/ directory

Expected JSON Structure:
    [
        {
            "part_section": "The Act Name Chapter X Section Y",
            "law_text": "Detailed text of the law provision..."
        },
        ...
    ]

What this script does:
    • Discovers all JSON files in data directory (excluding case files)
    • Loads law references from each JSON file
    • Intelligently chunks large law texts using RecursiveTextSplitter
    • Creates embeddings using Google Gemini for each chunk
    • Uploads each chunk as a vector point to Qdrant (preserving part_section)
    • Tests the search functionality after upload

Collection Details:
    • Collection name: bd_law_reference (from config)
    • Vector dimension: 3072 (Gemini text-embedding-001)
    • Distance metric: Cosine similarity

Files Processing:
    • Processes multiple JSON files sequentially
    • Maintains unique point IDs across all files and chunks
    • Shows progress for each file and overall progress
    • Displays chunking statistics (chunks created, average per reference)
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)
    
    exit_code = main()
    sys.exit(exit_code)
