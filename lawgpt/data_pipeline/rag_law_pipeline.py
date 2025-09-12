import json
import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lawgpt.core.config import settings

logger = logging.getLogger(__name__)


class LawRAGPipeline:
    """RAG Pipeline for Law References using Qdrant and Gemini Embeddings"""
    
    def __init__(self):
        """Initialize the LawRAGPipeline using settings from config"""
        self.collection_name = settings.QDRANT_LAW_REFERENCE_COLLECTION_NAME
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        
        # Initialize Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        
        # Initialize text splitter for chunking large law texts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Conservative chunk size to stay well under embedding limits
            chunk_overlap=100,  # Small overlap to maintain context
            length_function=len
        )
        
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collection_exists = self.qdrant_client.collection_exists(self.collection_name)
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=3072,  # Gemini text-embedding-004 dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def add_law_references(self, json_file_path: str, batch_size: int = 50, verbose: bool = True, start_index: int = 0) -> bool:
        """
        Load law references from JSON file and add them to Qdrant with progress tracking
        
        Args:
            json_file_path: Path to the JSON file containing law references
            batch_size: Number of references to process in each batch
            verbose: Whether to show detailed progress
            start_index: Index to start processing from (for resuming)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                law_data = json.load(file)
            
            total_references = len(law_data)
            # Normalize and validate start index
            if start_index < 0:
                start_index = 0
            if start_index >= total_references:
                if verbose:
                    print(f"ℹ️  start_index {start_index} is beyond total references ({total_references}). Nothing to do.")
                return True

            processed_count = 0
            
            if verbose:
                if start_index > 0:
                    print(f"📋 Resuming processing from reference {start_index + 1} of {total_references} in batches of {batch_size}")
                else:
                    print(f"📋 Processing {total_references} law references in batches of {batch_size}")
            
            # Process references in batches (respect start_index)
            for batch_start in range(start_index, total_references, batch_size):
                batch_end = min(batch_start + batch_size, total_references)
                batch_references = law_data[batch_start:batch_end]
                
                if verbose:
                    print(f"🔄 Processing batch {batch_start//batch_size + 1}/{(total_references + batch_size - 1)//batch_size} (references {batch_start + 1}-{batch_end})")
                
                points = []
                current_point_id = batch_start
                
                for idx, law_ref in enumerate(batch_references):
                    # Create chunks from the law reference
                    chunks = self._create_law_chunks(law_ref)
                    
                    part_section = law_ref.get('part_section', 'Unknown')[:60]
                    if verbose and (idx + 1) % 10 == 0:
                        print(f"  📝 Processing reference {current_point_id + 1}: {part_section}... ({len(chunks)} chunks)")
                    
                    # Process each chunk
                    for chunk_idx, chunk_data in enumerate(chunks):
                        # Generate text embedding for the chunk
                        text_embedding = self.embeddings.embed_query(chunk_data["content"])
                        
                        # Create metadata payload with chunk information
                        payload = {
                            "part_section": chunk_data["part_section"],
                            "law_text": law_ref.get("law_text", ""),  # Keep full text for reference
                            "chunk_content": chunk_data["chunk_content"],  # Store actual chunk content
                            "chunk_index": chunk_data["chunk_index"],
                            "total_chunks": chunk_data["total_chunks"],
                            "is_chunked": chunk_data["total_chunks"] > 1
                        }
                        
                        # Create point for Qdrant
                        point = models.PointStruct(
                            id=current_point_id,
                            vector=text_embedding,
                            payload=payload
                        )
                        points.append(point)
                        current_point_id += 1
                        
                        if verbose and chunk_data["total_chunks"] > 1:
                            logger.debug(f"  📄 Created chunk {chunk_idx + 1}/{chunk_data['total_chunks']} for '{part_section}...'")
                
                # Upload batch to collection
                if verbose:
                    print(f"  💾 Uploading batch to Qdrant...")
                
                self.qdrant_client.upload_points(
                    collection_name=self.collection_name,
                    points=points
                )
                
                processed_count += len(points)
                
                if verbose:
                    progress_percent = (batch_end / total_references) * 100
                    print(f"  ✅ Batch uploaded! {len(points)} chunks from {len(batch_references)} law references. Progress: {batch_end}/{total_references} ({progress_percent:.1f}%)")
                    print()
            
            logger.info(f"Successfully added {processed_count} chunks from law references to collection")
            if verbose:
                print(f"🎉 All {processed_count} chunks from law references uploaded successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding law references: {e}")
            if verbose:
                print(f"❌ Error: {e}")
            return False
    
    def add_multiple_law_files(self, json_file_paths: List[str], batch_size: int = 50, verbose: bool = True) -> bool:
        """
        Load law references from multiple JSON files and add them to Qdrant
        
        Args:
            json_file_paths: List of paths to JSON files containing law references
            batch_size: Number of references to process in each batch
            verbose: Whether to show detailed progress
            
        Returns:
            True if all files processed successfully, False otherwise
        """
        total_files = len(json_file_paths)
        current_point_id = 0
        
        # Get current collection info to determine starting point ID
        try:
            collection_info = self.get_collection_info()
            current_point_id = collection_info.get("points_count", 0)
            if verbose:
                print(f"📊 Starting from point ID: {current_point_id}")
        except Exception:
            if verbose:
                print("📊 Starting from point ID: 0 (new collection)")
        
        for file_idx, json_file_path in enumerate(json_file_paths, 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"📁 Processing file {file_idx}/{total_files}: {os.path.basename(json_file_path)}")
                print(f"{'='*60}")
            
            if not os.path.exists(json_file_path):
                logger.error(f"File not found: {json_file_path}")
                if verbose:
                    print(f"❌ File not found: {json_file_path}")
                continue
            
            # Process this file with custom point IDs
            success = self._add_law_file_with_custom_ids(
                json_file_path, 
                batch_size, 
                verbose, 
                current_point_id
            )
            
            if not success:
                logger.error(f"Failed to process file: {json_file_path}")
                if verbose:
                    print(f"❌ Failed to process file: {json_file_path}")
                return False
            
            # Update current_point_id for next file (need to count total chunks, not just law references)
            try:
                # Get the updated collection info to get the correct next point ID
                collection_info = self.get_collection_info()
                current_point_id = collection_info.get("points_count", current_point_id)
                logger.info(f"Updated current_point_id to {current_point_id} after processing {os.path.basename(json_file_path)}")
            except Exception as e:
                logger.error(f"Error updating point ID after processing {json_file_path}: {e}")
                # Fallback: estimate based on law references (this might be inaccurate with chunking)
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as file:
                        law_data = json.load(file)
                    current_point_id += len(law_data) * 2  # Conservative estimate assuming some chunking
                except Exception:
                    current_point_id += 100  # Very conservative fallback
        
        if verbose:
            print(f"\n🎉 Successfully processed all {total_files} law reference files!")
        
        return True
    
    def _add_law_file_with_custom_ids(self, json_file_path: str, batch_size: int, verbose: bool, start_point_id: int) -> bool:
        """Helper method to add law references with custom starting point IDs"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                law_data = json.load(file)
            
            total_references = len(law_data)
            processed_count = 0
            
            if verbose:
                print(f"📋 Processing {total_references} law references in batches of {batch_size}")
            
            # Process references in batches
            for batch_start in range(0, total_references, batch_size):
                batch_end = min(batch_start + batch_size, total_references)
                batch_references = law_data[batch_start:batch_end]
                
                if verbose:
                    print(f"🔄 Processing batch {batch_start//batch_size + 1}/{(total_references + batch_size - 1)//batch_size} (references {batch_start + 1}-{batch_end})")
                
                points = []
                skipped_count = 0
                current_point_id = start_point_id + batch_start
                
                for idx, law_ref in enumerate(batch_references):
                    try:
                        # Create chunks from the law reference
                        chunks = self._create_law_chunks(law_ref)
                        
                        part_section = law_ref.get('part_section', 'Unknown')[:60]
                        if verbose and (idx + 1) % 10 == 0:
                            print(f"  📝 Processing reference {current_point_id + 1}: {part_section}... ({len(chunks)} chunks)")
                        
                        # Process each chunk
                        for chunk_idx, chunk_data in enumerate(chunks):
                            try:
                                # Generate text embedding for the chunk
                                text_embedding = self.embeddings.embed_query(chunk_data["content"])
                                
                                # Create metadata payload with chunk information
                                payload = {
                                    "part_section": chunk_data["part_section"],
                                    "law_text": law_ref.get("law_text", ""),  # Keep full text for reference
                                    "chunk_content": chunk_data["chunk_content"],  # Store actual chunk content
                                    "chunk_index": chunk_data["chunk_index"],
                                    "total_chunks": chunk_data["total_chunks"],
                                    "is_chunked": chunk_data["total_chunks"] > 1
                                }
                                
                                # Create point for Qdrant
                                point = models.PointStruct(
                                    id=current_point_id,
                                    vector=text_embedding,
                                    payload=payload
                                )
                                points.append(point)
                                current_point_id += 1
                                
                                if verbose and chunk_data["total_chunks"] > 1:
                                    logger.debug(f"  📄 Created chunk {chunk_idx + 1}/{chunk_data['total_chunks']} for '{part_section}...'")
                                
                            except Exception as chunk_e:
                                skipped_count += 1
                                error_msg = str(chunk_e)
                                logger.error(f"Error processing chunk {chunk_idx + 1} for '{part_section}...': {chunk_e}")
                                if verbose:
                                    print(f"  ❌ Error processing chunk {chunk_idx + 1} for '{part_section}...': {error_msg[:100]}")
                                continue
                        
                    except Exception as e:
                        skipped_count += 1
                        error_msg = str(e)
                        part_section = law_ref.get('part_section', 'Unknown')[:100]
                        
                        logger.error(f"Error processing law reference '{part_section}...': {e}")
                        if verbose:
                            print(f"  ❌ Error processing law reference '{part_section}...': {error_msg[:100]}")
                        continue
                
                # Upload batch to collection (only if we have points to upload)
                if points:
                    if verbose:
                        print(f"  💾 Uploading batch to Qdrant ({len(points)} points)...")
                    
                    self.qdrant_client.upload_points(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    processed_count += len(points)
                else:
                    if verbose:
                        print(f"  ⚠️  No valid points in this batch (all {len(batch_references)} items skipped)")
                
                if verbose:
                    progress_percent = (batch_end / total_references) * 100
                    if skipped_count > 0:
                        print(f"  ✅ Batch processed! {len(points)} chunks uploaded, {skipped_count} items skipped. Progress: {batch_end}/{total_references} ({progress_percent:.1f}%)")
                    else:
                        print(f"  ✅ Batch uploaded! {len(points)} chunks from {len(batch_references)} law references. Progress: {batch_end}/{total_references} ({progress_percent:.1f}%)")
                    print()
            
            logger.info(f"Successfully added {processed_count} chunks from {total_references} law references from {json_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding law references from {json_file_path}: {e}")
            return False
    
    def _create_law_chunks(self, law_ref: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create chunks from a law reference using RecursiveTextSplitter
        
        Args:
            law_ref: Dictionary containing law reference information
            
        Returns:
            List of dictionaries containing chunked content with part_section preserved
        """
        part_section = law_ref.get('part_section', '')
        law_text = law_ref.get('law_text', '')
        
        # Check if the law text needs chunking
        if len(law_text) <= 1000:  # If small enough, don't chunk (lowered threshold for small models)
            logger.debug(f"Law text for '{part_section[:50]}...' is small enough ({len(law_text)} chars), no chunking needed")
            return [{
                "content": f"Part Section: {part_section}\nLaw Text: {law_text}",
                "part_section": part_section,
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_content": law_text  # Store the actual chunk content
            }]
        
        # Split the law text into chunks
        logger.info(f"Chunking law text for '{part_section[:50]}...' ({len(law_text)} chars)")
        chunks = self.text_splitter.split_text(law_text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            # Create content with part_section preserved for each chunk
            content = f"Part Section: {part_section}\nLaw Text (Chunk {i+1}/{len(chunks)}): {chunk}"
            
            chunk_data.append({
                "content": content,
                "part_section": part_section,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_content": chunk  # Store the actual chunk content separate from full content
            })
        
        logger.info(f"Created {len(chunks)} chunks for '{part_section[:50]}...'")
        return chunk_data
    
    def search_by_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar law references using text query
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching law references with scores
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                with_payload=True,
                limit=limit
            )
            # Sort results by higher score first
            results.points = sorted(results.points, key=lambda x: x.score, reverse=True)
            
            # Format results to prioritize chunk content
            formatted_results = []
            for point in results.points:
                # Use chunk content if available, otherwise fall back to full law_text
                chunk_content = point.payload.get("chunk_content", "")
                if not chunk_content:
                    # Fallback for legacy data without chunk_content
                    chunk_content = point.payload.get("law_text", "")
                
                formatted_results.append({
                    "type": "law",
                    "content": chunk_content,  # Return only the relevant chunk content
                    "metadata": {
                        "part_section": point.payload.get("part_section", ""),
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "total_chunks": point.payload.get("total_chunks", 1),
                        "is_chunked": point.payload.get("is_chunked", False)
                    },
                    "score": point.score,
                    "id": point.id,
                    "payload": point.payload  # Keep for backward compatibility
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search by text: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
