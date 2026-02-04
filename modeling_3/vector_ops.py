"""
Embedding generation & Neo4j Vector Index writing with temporal metadata.
Handles temporal-aware similarity search.
"""

from typing import List, Dict, Optional
from datetime import datetime
import hashlib


class TemporalVectorOps:
    """Vector operations with temporal awareness."""
    
    def __init__(self, embedding_model=None):
        """
        Initialize vector operations.
        
        Args:
            embedding_model: Pre-initialized embedding model or model name
        """
        # Import here to avoid loading if not needed
        from sentence_transformers import SentenceTransformer
        from config.model_config import EMBEDDING_MODEL
        
        # Initialize embedding model
        if embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"[OK] Model loaded successfully (dimension: {self.model.get_sentence_embedding_dimension()})")
        elif isinstance(embedding_model, str):
            print(f"Loading embedding model: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
            print(f"[OK] Model loaded successfully (dimension: {self.model.get_sentence_embedding_dimension()})")
        else:
            self.model = embedding_model
            print("[OK] Using pre-initialized embedding model")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        if self.model is None:
            raise NotImplementedError("Embedding model not initialized. Please configure your embedding library.")
        
        try:
            # Generate embedding using sentence-transformers
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"[WARN] Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            raise NotImplementedError("Embedding model not initialized.")
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"[WARN] Error generating batch embeddings: {e}")
            raise
    
    def generate_embedding_with_temporal_context(self, text: str, temporal_metadata: Dict) -> List[float]:
        """
        Generate embedding with temporal context included.
        
        Args:
            text: Text to embed
            temporal_metadata: Dict with temporal metadata (dates, status, etc.)
        
        Returns:
            Embedding vector
        """
        # Include temporal metadata in text for embedding
        temporal_context = self._format_temporal_context(temporal_metadata)
        enhanced_text = f"{temporal_context}\n\n{text}"
        
        return self.generate_embedding(enhanced_text)
    
    def _format_temporal_context(self, metadata: Dict) -> str:
        """Format temporal metadata as text for embedding."""
        context_parts = []
        
        if 'effective_from' in metadata and metadata['effective_from']:
            context_parts.append(f"Effective from: {metadata['effective_from']}")
        
        if 'effective_to' in metadata and metadata['effective_to']:
            context_parts.append(f"Effective to: {metadata['effective_to']}")
        else:
            context_parts.append("Currently active")
        
        if 'status' in metadata:
            context_parts.append(f"Status: {metadata['status']}")
        
        if 'decision_date' in metadata and metadata['decision_date']:
            context_parts.append(f"Decision date: {metadata['decision_date']}")
        
        if 'section_number' in metadata:
            context_parts.append(f"Section: {metadata['section_number']}")
        
        return " | ".join(context_parts)
    
    def store_embedding_in_neo4j(self, neo4j_ops, chunk_id: str, embedding: List[float], metadata: Dict):
        """
        Store embedding in Neo4j vector index.
        
        Args:
            neo4j_ops: TemporalNeo4jOps instance
            chunk_id: Chunk identifier
            embedding: Embedding vector
            metadata: Temporal metadata
        """
        # Update chunk node with embedding
        # Note: Neo4j vector indexes are created separately
        # This stores the embedding as a property for now
        
        query = """
        MATCH (ch:Chunk {chunk_id: $chunk_id})
        SET ch.embedding = $embedding
        SET ch.effective_from = date($effective_from)
        SET ch.effective_to = $effective_to
        SET ch.status = $status
        RETURN ch.chunk_id AS chunk_id
        """
        
        parameters = {
            'chunk_id': chunk_id,
            'embedding': embedding,
            'effective_from': metadata.get('effective_from'),
            'effective_to': metadata.get('effective_to'),
            'status': metadata.get('status', 'active')
        }
        
        neo4j_ops._execute_write(query, parameters)
    
    def temporal_aware_similarity_search(
        self,
        neo4j_ops,
        query_embedding: List[float],
        query_date: str,
        doc_type: Optional[str] = None,
        top_k: int = 10,
        exclude_overruled: bool = True
    ) -> List[Dict]:
        """
        Perform temporal-aware similarity search.
        
        Args:
            neo4j_ops: TemporalNeo4jOps instance
            query_embedding: Query embedding vector
            query_date: Date to filter by (ISO format)
            doc_type: 'case', 'statute', or None for both
            top_k: Number of results to return
            exclude_overruled: Whether to exclude overruled/repealed documents
        
        Returns:
            List of similar chunks with scores
        """
        from config.model_config import VECTOR_INDEX_NAME
        
        # Build temporal and type filters
        where_clauses = [
            "node.effective_from <= date($query_date)",
            "(node.effective_to IS NULL OR node.effective_to >= date($query_date))"
        ]
        
        # Map user-friendly doc types to LRMoo types
        if doc_type == 'statute':
            where_clauses.append("(node.source_doc_type = 'statute' OR node.source_doc_type = 'Expression')")
        elif doc_type == 'case':
            where_clauses.append("node.source_doc_type = 'case'")
        elif doc_type:
            where_clauses.append(f"node.source_doc_type = '{doc_type}'")
        
        if exclude_overruled:
            where_clauses.append("node.status = 'active'")
        
        where_clause = " AND ".join(where_clauses)
        
        # Use Neo4j vector index for similarity search
        # Note: This requires Neo4j 5.11+ with vector index created
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node, score
        WHERE {where_clause}
        RETURN 
            node.chunk_id AS chunk_id,
            node.text AS text,
            node.source_doc_id AS source_doc_id,
            node.source_doc_type AS source_doc_type,
            node.effective_from AS effective_from,
            node.effective_to AS effective_to,
            node.status AS status,
            node.section_number AS section_number,
            score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        parameters = {
            'index_name': VECTOR_INDEX_NAME,
            'query_embedding': query_embedding,
            'query_date': query_date,
            'top_k': top_k * 2  # Fetch more to account for filtering
        }
        
        try:
            results = neo4j_ops._execute_read(query, parameters)
            return results[:top_k]  # Return top_k after filtering
        except Exception as e:
            print(f"[WARN] Vector search failed (index may not exist): {e}")
            print("  Falling back to simple temporal filtering...")
            return self._fallback_search(neo4j_ops, query_date, doc_type, exclude_overruled, top_k)
    
    def _fallback_search(self, neo4j_ops, query_date: str, doc_type: Optional[str], 
                        exclude_overruled: bool, top_k: int) -> List[Dict]:
        """Fallback search without vector index."""
        # Handle LRMoo mapping
        doc_type_filter = ""
        if doc_type == 'statute':
            doc_type_filter = "AND (ch.source_doc_type = 'statute' OR ch.source_doc_type = 'Expression')"
        elif doc_type == 'case':
            doc_type_filter = "AND ch.source_doc_type = 'case'"
        elif doc_type:
            doc_type_filter = f"AND ch.source_doc_type = '{doc_type}'"
            
        status_filter = "AND ch.status = 'active'" if exclude_overruled else ""
        
        query = f"""
        MATCH (ch:Chunk)
        WHERE ch.effective_from <= date($query_date)
          AND (ch.effective_to IS NULL OR ch.effective_to >= date($query_date))
          {doc_type_filter}
          {status_filter}
        RETURN 
            ch.chunk_id AS chunk_id,
            ch.text AS text,
            ch.source_doc_id AS source_doc_id,
            ch.source_doc_type AS source_doc_type,
            ch.effective_from AS effective_from,
            ch.effective_to AS effective_to,
            ch.status AS status,
            0.5 AS score
        ORDER BY ch.effective_from DESC
        LIMIT $top_k
        """
        
        parameters = {'query_date': query_date, 'top_k': top_k}
        return neo4j_ops._execute_read(query, parameters)
    
    def boost_recent_documents(self, results: List[Dict], query_date: str, recency_weight: float = 0.1) -> List[Dict]:
        """
        Boost scores for more recent documents.
        
        Args:
            results: List of search results with scores
            query_date: Query date
            recency_weight: Weight for recency boost (0-1)
        
        Returns:
            Results with boosted scores
        """
        query_dt = datetime.fromisoformat(query_date)
        
        for result in results:
            if 'effective_from' in result and result['effective_from']:
                try:
                    effective_dt = datetime.fromisoformat(str(result['effective_from']))
                    days_diff = (query_dt - effective_dt).days
                    
                    # Boost: more recent = higher boost
                    # Normalize to 0-1 range (assume max 10 years = 3650 days)
                    recency_score = max(0, 1 - (days_diff / 3650))
                    boost = recency_score * recency_weight
                    
                    result['score'] = result.get('score', 0.0) + boost
                except (ValueError, TypeError):
                    pass
        
        # Re-sort by boosted score
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        return results
