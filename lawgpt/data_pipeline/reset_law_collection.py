"""
Reset script for clearing the law references collection in Qdrant.

This script provides options to:
- Delete the entire law references collection
- Check collection status
- Recreate an empty collection
"""

import sys
from rag_law_pipeline import LawRAGPipeline


def main():
    """Main function to reset the law collection"""
    
    try:
        print("🔄 Law References Collection Reset Tool")
        print("=" * 40)
        
        # Initialize pipeline
        pipeline = LawRAGPipeline()
        
        # Get current collection info
        collection_info = pipeline.get_collection_info()
        
        if not collection_info or collection_info.get("vectors_count") == 0:
            print("ℹ️  Collection is already empty or doesn't exist.")
            print(f"📊 Current status: {collection_info}")
            return 0
        
        # Show current status
        print(f"📊 Collection: {collection_info.get('name', 'Unknown')}")
        print(f"📈 Current vectors: {collection_info.get('vectors_count', 0)}")
        print(f"📍 Current points: {collection_info.get('points_count', 0)}")
        print(f"🔄 Status: {collection_info.get('status', 'Unknown')}")
        
        # Ask for confirmation
        print(f"\n⚠️  WARNING: This will permanently delete all {collection_info.get('vectors_count', 0)} law references from the collection!")
        response = input("Are you sure you want to proceed? Type 'DELETE' to confirm: ")
        
        if response != 'DELETE':
            print("❌ Reset cancelled. Collection unchanged.")
            return 0
        
        # Delete collection
        print("\n🗑️  Deleting collection...")
        success = pipeline.delete_collection()
        
        if success:
            print("✅ Collection deleted successfully!")
            
            # Recreate empty collection
            print("🔄 Recreating empty collection...")
            new_pipeline = LawRAGPipeline()  # This will create a new empty collection
            
            new_info = new_pipeline.get_collection_info()
            print(f"✅ New empty collection created!")
            print(f"📊 New collection status: {new_info}")
            
            return 0
        else:
            print("❌ Failed to delete collection!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return 1


def show_status_only():
    """Show collection status without making changes"""
    try:
        print("📊 Law References Collection Status")
        print("=" * 35)
        
        pipeline = LawRAGPipeline()
        collection_info = pipeline.get_collection_info()
        
        if not collection_info:
            print("❌ Collection not found or error occurred.")
            return 1
        
        print(f"📂 Collection: {collection_info.get('name', 'Unknown')}")
        print(f"📈 Vectors: {collection_info.get('vectors_count', 0)}")
        print(f"📍 Points: {collection_info.get('points_count', 0)}")
        print(f"🔄 Status: {collection_info.get('status', 'Unknown')}")
        
        vectors_count = collection_info.get('vectors_count', 0)
        if vectors_count and vectors_count > 0:
            print("\n🔍 Testing search functionality...")
            results = pipeline.search_by_text("test", limit=1)
            if results:
                print("✅ Search is working correctly")
            else:
                print("⚠️  Search returned no results")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return 1


def show_help():
    """Show help information"""
    print("""
🗑️ Law References Collection Reset Tool

This script manages the law references collection in Qdrant.

Usage:
    python reset_law_collection.py           # Delete and recreate collection
    python reset_law_collection.py status    # Show collection status only
    python reset_law_collection.py --help    # Show this help

Commands:
    (no args)    Delete all law references and recreate empty collection
    status       Show current collection information
    --help       Show this help message

⚠️  WARNING: The reset operation is irreversible!
    Make sure you have a backup of your data before proceeding.

What the reset does:
    1. Shows current collection status
    2. Asks for confirmation (must type 'DELETE')
    3. Deletes the entire collection
    4. Creates a new empty collection

After reset, you'll need to run upload_law.py again to re-index your law references.
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            show_help()
            sys.exit(0)
        elif sys.argv[1] == 'status':
            exit_code = show_status_only()
            sys.exit(exit_code)
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information.")
            sys.exit(1)
    
    exit_code = main()
    sys.exit(exit_code)
