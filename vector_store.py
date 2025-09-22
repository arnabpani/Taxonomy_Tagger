"""
Vector Store Module

Handles document embedding creation, storage, and retrieval for RAG implementation.
Provides vector search capabilities to enhance document classification with relevant context.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from openai import OpenAI


class VectorStore:
    """Manages document embeddings and provides retrieval capabilities for RAG."""
    
    def __init__(self, config: Dict):
        """Initialize vector store with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get API key from config or environment
        api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        # Embedding model and parameters
        self.embedding_model = config.get('embedding_model', 'text-embedding-3-large')
        self.embedding_dimensions = config.get('embedding_dimensions', 3072)  # Dimensions for text-embedding-3-large
        
        # Storage settings
        self.vector_store_path = Path(config.get('vector_store_path', './vector_store'))
        self.vector_store_path.mkdir(exist_ok=True)
        
        # Document chunks and embeddings
        self.document_chunks = {}
        self.document_embeddings = {}
        self.taxonomy_embeddings = {}
        
        # Load existing embeddings if available
        self.load_embeddings()
    
    def create_document_embeddings(self, document_id: str, chunks: List[str]) -> bool:
        """Create embeddings for document chunks and store them."""
        try:
            self.document_chunks[document_id] = chunks
            embeddings = []
            
            for chunk in chunks:
                embedding = self._get_embedding(chunk)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if embeddings:
                self.document_embeddings[document_id] = embeddings
                self.save_embeddings()
                self.logger.info(f"Created embeddings for document {document_id} with {len(chunks)} chunks")
                return True
            else:
                self.logger.error(f"Failed to create any embeddings for document {document_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating document embeddings: {e}")
            return False
    
    def create_taxonomy_embeddings(self, taxonomy_data: Dict) -> bool:
        """Create embeddings for taxonomy elements to enable semantic search."""
        try:
            for facet_name, facet_values in taxonomy_data.items():
                self.taxonomy_embeddings[facet_name] = {}
                
                for value in facet_values:
                    # Create a rich description for embedding
                    if isinstance(value, dict):
                        # Handle complex taxonomy entries with multiple fields
                        description = self._create_taxonomy_description(value)
                        key = next(iter(value.values()))  # Use first value as key
                    else:
                        # Handle simple string taxonomy values
                        description = value
                        key = value
                    
                    embedding = self._get_embedding(description)
                    if embedding is not None:
                        self.taxonomy_embeddings[facet_name][key] = embedding
            
            self.save_embeddings()
            self.logger.info(f"Created embeddings for {len(self.taxonomy_embeddings)} taxonomy facets")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating taxonomy embeddings: {e}")
            return False
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve most relevant document chunks for a query."""
        try:
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []
            
            results = []
            
            # Search through all document chunks
            for doc_id, embeddings in self.document_embeddings.items():
                chunks = self.document_chunks[doc_id]
                
                for i, embedding in enumerate(embeddings):
                    if i < len(chunks):  # Ensure we have a corresponding chunk
                        similarity = self._calculate_similarity(query_embedding, embedding)
                        results.append((similarity, chunks[i], doc_id))
            
            # Sort by similarity (highest first) and return top_k chunks
            results.sort(reverse=True, key=lambda x: x[0])
            return [chunk for _, chunk, _ in results[:top_k]]
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            return []
    
    def retrieve_taxonomy_matches(self, query: str, facet_name: str, top_k: int = 3) -> List[str]:
        """Find most relevant taxonomy values for a query within a specific facet."""
        try:
            query_embedding = self._get_embedding(query)
            if query_embedding is None or facet_name not in self.taxonomy_embeddings:
                return []
            
            results = []
            
            # Search through taxonomy values for this facet
            for value, embedding in self.taxonomy_embeddings[facet_name].items():
                similarity = self._calculate_similarity(query_embedding, embedding)
                results.append((similarity, value))
            
            # Sort by similarity (highest first) and return top_k values
            results.sort(reverse=True, key=lambda x: x[0])
            return [value for _, value in results[:top_k]]
            
        except Exception as e:
            self.logger.error(f"Error retrieving taxonomy matches: {e}")
            return []
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays for efficient calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _create_taxonomy_description(self, taxonomy_entry: Dict) -> str:
        """Create a rich description from taxonomy entry for better embedding."""
        description_parts = []
        
        # Add all fields to the description
        for key, value in taxonomy_entry.items():
            if isinstance(value, str) and value.strip():
                description_parts.append(f"{key}: {value}")
        
        return "\n".join(description_parts)
    
    def save_embeddings(self) -> bool:
        """Save embeddings to disk."""
        try:
            data = {
                'document_chunks': self.document_chunks,
                'document_embeddings': self.document_embeddings,
                'taxonomy_embeddings': self.taxonomy_embeddings
            }
            
            with open(self.vector_store_path / 'embeddings.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info("Embeddings saved to disk")
            return True
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self) -> bool:
        """Load embeddings from disk if available."""
        try:
            embedding_file = self.vector_store_path / 'embeddings.pkl'
            
            if embedding_file.exists():
                with open(embedding_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.document_chunks = data.get('document_chunks', {})
                self.document_embeddings = data.get('document_embeddings', {})
                self.taxonomy_embeddings = data.get('taxonomy_embeddings', {})
                
                self.logger.info(f"Loaded embeddings for {len(self.document_embeddings)} documents and {len(self.taxonomy_embeddings)} taxonomy facets")
                return True
            else:
                self.logger.info("No existing embeddings found")
                return False
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            return False