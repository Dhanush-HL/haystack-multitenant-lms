"""
Vector Store Implementation for HayStack Multi-Tenant LMS
Handles document embedding, storage, and semantic search
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class VectorStore:
    """Vector store for document embeddings and semantic search"""
    
    def __init__(self, 
                 collection_name: str = "haystack_documents",
                 persist_directory: str = "./chromadb_data",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"✅ Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ ChromaDB collection ready: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise
    
    def _generate_chunk_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID for document chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        return f"{metadata_hash}_{content_hash[:8]}"
    
    def add_document(self, 
                    content: str, 
                    metadata: Dict[str, Any],
                    chunk_size: int = 500,
                    chunk_overlap: int = 50) -> List[str]:
        """Add document to vector store with chunking"""
        
        try:
            # Split document into chunks
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
            chunk_ids = []
            
            for i, chunk_text in enumerate(chunks):
                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk_text)
                })
                
                # Generate chunk ID
                chunk_id = self._generate_chunk_id(chunk_text, chunk_metadata)
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk_text).tolist()
                
                # Store in ChromaDB
                self.collection.add(
                    documents=[chunk_text],
                    metadatas=[chunk_metadata],
                    embeddings=[embedding],
                    ids=[chunk_id]
                )
                
                chunk_ids.append(chunk_id)
                logger.debug(f"Added chunk {i+1}/{len(chunks)}: {chunk_id[:12]}...")
            
            logger.info(f"✅ Added document with {len(chunks)} chunks")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            raise
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.7:  # Don't break too early
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=metadata_filter if metadata_filter else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    search_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'similarity': 1 - results['distances'][0][i] if results['distances'] else 1.0
                    })
            
            logger.info(f"🔍 Found {len(search_results)} results for query")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document chunks by document ID"""
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=['ids']
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"✅ Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                logger.warning(f"⚠️ No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Get all IDs
            all_data = self.collection.get(include=['ids'])
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                logger.info(f"✅ Cleared {len(all_data['ids'])} documents from collection")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False

# Factory functions
def create_vector_store(tenant_key: str = "default", 
                       persist_dir: str = "./chromadb_data") -> VectorStore:
    """Create tenant-specific vector store"""
    collection_name = f"haystack_{tenant_key}"
    tenant_dir = os.path.join(persist_dir, tenant_key)
    
    return VectorStore(
        collection_name=collection_name,
        persist_directory=tenant_dir
    )

def create_shared_vector_store(persist_dir: str = "./chromadb_data") -> VectorStore:
    """Create shared vector store for all tenants"""
    return VectorStore(
        collection_name="haystack_shared",
        persist_directory=persist_dir
    )