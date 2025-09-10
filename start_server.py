#!/usr/bin/env python3
"""
Script to start the LawGPT server with proper setup
"""
import subprocess
import sys
import os

def install_dependencies():
    """Install project dependencies using uv"""
    print("Installing dependencies with uv...")
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ uv not found. Please install uv first: https://docs.astral.sh/uv/")
        return False

def check_env_file():
    """Check if .env file exists and warn if not"""
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found")
        print("Please create a .env file with the following variables:")
        print("GOOGLE_API_KEY=your_gemini_api_key")
        print("OPENAI_API_KEY=your_openai_api_key") 
        print("QDRANT_URL=your_qdrant_url")
        print("QDRANT_API_KEY=your_qdrant_api_key")
        print()

def start_server():
    """Start the FastAPI server"""
    print("Starting LawGPT server...")
    try:
        subprocess.run([
            "uv", "run", "uvicorn", 
            "lawgpt.main:app",
            "--host", "0.0.0.0",
            "--port", "8000", 
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True

def main():
    print("🏛️  LawGPT - Legal AI Assistant")
    print("=" * 40)
    
    # Check environment
    check_env_file()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting server on http://localhost:8000")
    print("📖 API docs available at http://localhost:8000/docs")
    print("🔧 Test endpoint with: python test_chat_endpoint.py")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 40)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
