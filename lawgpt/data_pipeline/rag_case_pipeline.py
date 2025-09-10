import json
import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from lawgpt.core.config import settings
logger = logging.getLogger(__name__)


class CaseRAGPipeline:
    """RAG Pipeline for Legal Case References using Qdrant and Gemini Embeddings"""
    
    def __init__(self):
        """Initialize the CaseRAGPipeline using settings from config"""
        self.collection_name = settings.QDRANT_LEGAL_CASES_COLLECTION_NAME
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        
        # Initialize Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
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
    
    def add_cases(self, json_file_path: str, batch_size: int = 50, verbose: bool = True, start_index: int = 0) -> bool:
        """
        Load legal cases from JSON file and add them to Qdrant with progress tracking
        
        Args:
            json_file_path: Path to the JSON file containing legal cases
            batch_size: Number of cases to process in each batch
            verbose: Whether to show detailed progress
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                cases_data = json.load(file)
            
            total_cases = len(cases_data)
            # Normalize and validate start index
            if start_index < 0:
                start_index = 0
            if start_index >= total_cases:
                if verbose:
                    print(f"ℹ️  start_index {start_index} is beyond total cases ({total_cases}). Nothing to do.")
                return True

            processed_count = 0 if start_index == 0 else 0
            
            if verbose:
                if start_index > 0:
                    print(f"📋 Resuming processing from case {start_index + 1} of {total_cases} in batches of {batch_size}")
                else:
                    print(f"📋 Processing {total_cases} cases in batches of {batch_size}")
            
            # Process cases in batches (respect start_index)
            for batch_start in range(start_index, total_cases, batch_size):
                batch_end = min(batch_start + batch_size, total_cases)
                batch_cases = cases_data[batch_start:batch_end]
                
                if verbose:
                    print(f"🔄 Processing batch {batch_start//batch_size + 1}/{(total_cases + batch_size - 1)//batch_size} (cases {batch_start + 1}-{batch_end})")
                
                points = []
                for idx, case in enumerate(batch_cases):
                    case_id = batch_start + idx
                    
                    # Create comprehensive text content for embedding
                    content = self._create_case_content(case)
                    
                    if verbose and (idx + 1) % 10 == 0:
                        print(f"  📝 Embedding case {case_id + 1}: {case.get('case-title', 'N/A')[:60]}...")
                    
                    # Generate text embedding
                    text_embedding = self.embeddings.embed_query(content)
                    
                    # Create metadata payload
                    payload = {
                        "case_title": case.get("case-title", ""),
                        "division": case.get("division", ""),
                        "law_category": case.get("law_category", ""),
                        "law_act": case.get("law_act", ""),
                        "reference": case.get("reference", ""),
                        "case_details": case.get("case-details", "")
                    }
                    
                    # Create point for Qdrant
                    point = models.PointStruct(
                        id=case_id,
                        vector=text_embedding,
                        payload=payload
                    )
                    points.append(point)
                
                # Upload batch to collection
                if verbose:
                    print(f"  💾 Uploading batch to Qdrant...")
                
                self.qdrant_client.upload_points(
                    collection_name=self.collection_name,
                    points=points
                )
                
                processed_count += len(points)
                
                if verbose:
                    progress_percent = (batch_end / total_cases) * 100
                    print(f"  ✅ Batch uploaded! Progress: {batch_end}/{total_cases} ({progress_percent:.1f}%)")
                    print()
            
            logger.info(f"Successfully added {processed_count} legal cases to collection")
            if verbose:
                print(f"🎉 All {processed_count} cases uploaded successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding cases: {e}")
            if verbose:
                print(f"❌ Error: {e}")
            return False
    
    def _create_case_content(self, case: Dict[str, Any]) -> str:
        """
        Create comprehensive text content for a case to be embedded
        
        Args:
            case: Dictionary containing case information
            
        Returns:
            Formatted text content
        """
        content_parts = [
            f"Case Title: {case.get('case-title', '')}",
            f"Division: {case.get('division', '')}",
            f"Law Category: {case.get('law_category', '')}",
            f"Law Act: {case.get('law_act', '')}",
            f"Reference: {case.get('reference', '')}",
            f"Case Details: {case.get('case-details', '')}"
        ]
        
        return "\n".join(content_parts)
    
    def search_by_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar cases using text query
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching cases with scores
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                with_payload=True,
                limit=limit
            )
            #reverse sort the results by higher score first
            results.points = sorted(results.points, key=lambda x: x.score, reverse=True)
            
            return [
                {
                    "payload": point.payload,
                    "score": point.score,
                    "id": point.id
                }
                for point in results.points
            ]
            
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