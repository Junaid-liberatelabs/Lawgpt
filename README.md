# LawGPT - Legal AI Assistant for Bangladesh Law

A comprehensive legal AI assistant system built with LangGraph, FastAPI, and RAG capabilities for Bangladesh law consultation.

## System Architecture

The system implements a LangGraph workflow with the following components:

### 1. **Chat API Endpoint** (`/api/v1/chat`)
- Accepts user messages with configurable model and RAG options
- Supports conversation threading for session continuity
- Returns structured AI responses

### 2. **LangGraph Workflow**
- **RAG Node**: Retrieves relevant legal cases and/or law references based on flags
- **LLM Node**: Generates responses using specified AI model with optional RAG context  
- **End Node**: Completes the workflow and returns the response

### 3. **Multi-Model Support**
- **Gemini**: Google's Gemini-2.0-flash-exp model
- **OpenAI**: GPT-4o-mini model
- **Custom LLM**: Configurable endpoint for fine-tuned Qwen model (Modal deployment ready)

### 4. **RAG Pipeline**
- **Case RAG**: Searches legal case collection using vector similarity
- **Law RAG**: Searches Bangladesh law references with intelligent chunking
- **Dual RAG**: Combines both case and law contexts when both flags are enabled


## API Usage

### Chat Request
```json
POST /api/v1/chat
{
  "message": "What are the penalties for theft in Bangladesh?",
  "llm_model_id": "gemini",        // "gemini" | "openai" | "custom_llm"
  "thread_id": "unique-thread-id",
  "is_case_rag": true,             // Enable case law RAG
  "is_law_rag": true               // Enable statutory law RAG
}
```

### Chat Response
```json
{
  "response": "Based on Bangladesh Penal Code 1860, theft is defined under Section 378..."
}
```

## Running the Application

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up environment:**
   ```bash
   # Create .env file with:
   GOOGLE_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

3. **Start the server:**
   ```bash
   uv run uvicorn lawgpt.main:app --reload
   ```

4. **Test the endpoint:**
   ```bash
   python test_chat_endpoint.py
   ```

## Model Support

This system supports multiple LLM models for comparison:
- **Gemini 2.0 Flash**: Latest Google model
- **GPT-4o Mini**: OpenAI's efficient model  
- **Custom Qwen**: Fine-tuned model for Bangladesh law (Modal deployment ready)

The system allows easy switching between models per request for performance comparison.

---

## Legacy RAG System Documentation

## Features

- **Vector Search**: Uses Qdrant for efficient similarity search
- **Gemini Embeddings**: Leverages Google's text-embedding-004 model for high-quality embeddings
- **Legal Case Indexing**: Processes and indexes legal cases from JSON data
- **Law Reference Processing**: Intelligent chunking of large law texts with RecursiveTextSplitter
- **Text-to-Text Search**: Find similar legal cases using natural language queries
- **Top-K Results**: Returns configurable number of most similar cases
- **CLI Interface**: Easy-to-use command-line interface for search and indexing

## Technology Stack

- **Vector Storage**: Qdrant (direct client)
- **Embeddings**: Google Gemini text-embedding-004  
- **Text Processing**: LangChain RecursiveCharacterTextSplitter for intelligent chunking
- **Language**: Python 3.13+
- **Configuration**: Pydantic Settings

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
GOOGLE_API_KEY=your_google_api_key_here
QDRANT_URL=http://localhost:6333
```

3. Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### 1. Programmatic Usage

```python
from data_pipeline.rag_pipeline import CaseRAGPipeline

# Initialize pipeline (uses config automatically)
pipeline = CaseRAGPipeline()

# Index legal cases
success = pipeline.add_cases("data/bd_legal_cases_complete.json")
if success:
    print("Cases indexed successfully!")

# Search for similar cases
results = pipeline.search_by_text(
    query="administrative tribunal jurisdiction",
    limit=5
)

for result in results:
    payload = result['payload']
    print(f"Case: {payload['case_title']}")
    print(f"Score: {result['score']}")
    print(f"Details: {payload['case_details'][:200]}...")
```

### 2. CLI Usage

#### Index legal cases:
```bash
cd data_pipeline
python cli_search.py index ../data/bd_legal_cases_complete.json
```

#### Search for cases:
```bash
cd data_pipeline
python cli_search.py search "administrative tribunal jurisdiction" --top-k 3
python cli_search.py search "service dismissal"
```

### 3. Example Usage Script

Run the example script to see the pipeline in action:
```bash
cd data_pipeline
python example_usage.py
```

## Data Format

The system expects legal cases in JSON format with the following structure:

```json
{
  "case-title": "Case Title Here",
  "case-details": "Detailed case information...",
  "division": "Appellate Division",
  "law_category": "Civil",
  "law_act": "The Administrative Tribunals Act, 1980",
  "reference": "Section 6"
}
```

## Configuration

The system uses environment variables for configuration:

- `GOOGLE_API_KEY`: Google API key for Gemini embeddings
- `QDRANT_URL`: Qdrant instance URL (default: http://localhost:6333)
- `QDRANT_LEGAL_CASES_COLLECTION_NAME`: Collection name (default: bd_legal_cases)

## API Reference

### CaseRAGPipeline Class

#### Methods:

- `add_cases(json_file_path: str) -> bool`: Load and index cases from JSON file
- `search_by_text(query: str, limit: int = 5) -> List[Dict]`: Search for similar cases using text
- `get_collection_info() -> Dict`: Get collection information
- `delete_collection() -> bool`: Delete the entire collection

#### Search Results Format:

```python
{
    "payload": {
        "case_title": str,
        "division": str,
        "law_category": str,
        "law_act": str,
        "reference": str,
        "case_details": str
    },
    "score": float,
    "id": int
}
```

## Legal Data Generation Workflow

### Agentic Workflow for Training Data

The system includes a LangGraph-based agentic workflow for generating high-quality legal Q&A pairs for AI model fine-tuning:

```python
from lawgpt.workflow import run_legal_data_generation

# Generate training data with specified iterations
# No user prompt needed - uses comprehensive built-in prompt templates
run_legal_data_generation(
    iterations=10,
    csv_file_path="legal_training_data.csv"
)
```

#### Workflow Components:

1. **Query Generation**: Uses Gemini LLM to generate 3 diverse legal queries per iteration
2. **Data Retrieval**: RAG pipeline searches for relevant legal cases for each query
3. **Answer Generation**: Gemini LLM generates structured answers based on retrieved data
4. **CSV Storage**: Saves Q&A pairs to CSV file for training data

#### Features:

- **Memory Checkpointer**: LangGraph in-memory checkpointer for conversation history
- **Structured Output**: Uses Pydantic schemas for consistent data format
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Configurable Iterations**: Specify how many Q&A sets to generate
- **Error Handling**: Robust error handling with continuation on failures

### Usage Examples:

```bash
# Run workflow example
python lawgpt/workflow_example.py

# Test workflow (single iteration)
python test_workflow.py

# Direct workflow execution
python -m lawgpt.workflow
```

## Law Text Chunking

The system now uses intelligent chunking for large law texts instead of truncation:

### Features:
- **RecursiveCharacterTextSplitter**: Splits text at natural boundaries (paragraphs, sentences, etc.)
- **Preserved Context**: Each chunk retains the original `part_section` for accurate referencing
- **Overlap Strategy**: 200-character overlap between chunks to maintain context
- **Chunk Size**: 8000 characters per chunk (conservative limit for embeddings)
- **Metadata Tracking**: Each chunk includes `chunk_index`, `total_chunks`, and `is_chunked` flags

### Usage Example:
```python
from lawgpt.data_pipeline.rag_law_pipeline import LawRAGPipeline

pipeline = LawRAGPipeline()

# Add law references with automatic chunking
success = pipeline.add_law_references("data/bangladesh_criminal_procedure_code.json")

# Search results will include chunk information
results = pipeline.search_by_text("criminal procedure jurisdiction")
for result in results:
    payload = result['payload']
    if payload.get('is_chunked'):
        print(f"Chunk {payload['chunk_index'] + 1}/{payload['total_chunks']}")
    print(f"Part Section: {payload['part_section']}")
```

## Patterns and Best Practices

- **Error Handling**: Comprehensive exception handling throughout the pipeline
- **Configuration Management**: Centralized configuration using Pydantic Settings
- **Type Safety**: Full type hints for better development experience
- **Modular Design**: Separate concerns between indexing, search, and configuration
- **CLI Interface**: User-friendly command-line tools for common operations
- **Agentic Workflows**: LangGraph-based workflows for complex AI tasks
- **Structured Output**: Pydantic schemas for consistent data generation
- **Intelligent Chunking**: Preserves legal context while handling large documents

## Development

- Keep files under 300 lines of code
- Use comprehensive error handling
- Write type hints for all functions
- Follow the existing configuration patterns
- Test thoroughly before deployment

## Requirements

- Python 3.13+
- Qdrant instance (local or remote)
- Google API key for Gemini embeddings
- Sufficient storage for vector embeddings (768 dimensions per case)
