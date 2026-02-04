"""
Hybrid Retriever: Combines Vector Search (Semantic) with Cypher Traversal (Structural)
=====================================================================================

Implements a GraphRAG retrieval system that:
1. Uses LLM to extract legal entities and temporal constraints from queries
2. Performs vector search on :Segment embeddings (using segment_embeddings_index)
3. Traverses graph to find parent :Case and linked :Provision nodes
4. Applies temporal filtering based on decision_date
5. Packages context with URNs for citation

Note: Nodes have both :Segment (for vector index) and :CaseSegment (for graph structure) labels.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
from neo4j import GraphDatabase
import logging
import os
import re

logger = logging.getLogger(__name__)

# Try to import Gemini for query understanding
try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google.genai not available. Query understanding will use regex fallback.")

# Try to import sentence transformers for query embedding
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers not available. Vector search will be disabled.")


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and graph traversal.
    
    Architecture:
    - Query Understanding: LLM extracts legal entities and temporal constraints
    - Vector Search: Semantic similarity on :CaseSegment embeddings
    - Graph Traversal: Structural relationships (Case -> Segment -> Provision)
    - Temporal Filtering: decision_date on :Case nodes
    - Context Packaging: Structured evidence with URNs
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        gemini_api_key: Optional[str] = None,
        embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        """
        Initialize Hybrid Retriever.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Database name
            gemini_api_key: Gemini API key for query understanding (optional)
            embedding_model_name: Name of embedding model for query vectorization
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = neo4j_database
        
        # Verify vector index at startup
        self._verify_vector_index()
        
        # Initialize Gemini for query understanding
        if GEMINI_AVAILABLE:
            api_key = gemini_api_key or os.getenv('GOOGLE_API_KEY')
            if api_key:
                try:
                    self.gemini_client = genai.Client(api_key=api_key)
                    self.gemini_model_name = 'gemini-2.5-flash'
                    logger.info("Gemini initialized for query understanding")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}. Using regex fallback.")
                    self.gemini_client = None
            else:
                self.gemini_client = None
                logger.warning("Gemini API key not provided. Using regex fallback for query understanding.")
        else:
            self.gemini_client = None
        
        # Initialize embedding model for query vectorization
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
                logger.info(f"Embedding model loaded (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
    
    def _verify_vector_index(self):
        """
        Verify vector index exists and is ready for queries.
        
        Checks:
        - Index exists
        - Status is ONLINE
        - Population is 100%
        - Dimensions are 768
        """
        try:
            with self.driver.session(database=self.database) as session:
                check_query = """
                SHOW VECTOR INDEXES
                WHERE name = 'segment_embeddings_index'
                """
                
                result = session.run(check_query)
                records = list(result)
                
                if not records:
                    logger.warning(
                        "Vector index 'segment_embeddings_index' not found. "
                        "Vector search will be disabled. Please create the index first."
                    )
                    return False
                
                for record in records:
                    # Log all available fields for debugging
                    logger.debug(f"Vector index record fields: {list(record.keys())}")
                    
                    status = record.get('state') or record.get('status', 'UNKNOWN')
                    population_percent = record.get('populationPercent') or record.get('populationPercent', 0.0)
                    # Try different field names for dimensions (Neo4j versions vary)
                    dimensions = (record.get('dimensions') or 
                                 record.get('vectorDimensions') or 
                                 record.get('config', {}).get('vector.dimensions', None))
                    
                    # If dimensions is still None or 0, try to get from config
                    if not dimensions or dimensions == 0:
                        config = record.get('config') or record.get('options') or {}
                        if isinstance(config, dict):
                            dimensions = config.get('vector.dimensions') or config.get('dimensions', None)
                    
                    if status != 'ONLINE':
                        logger.error(
                            f"Vector index 'segment_embeddings_index' is not ONLINE: status={status}. "
                            "Vector search may fail."
                        )
                        return False
                    
                    if population_percent < 100.0:
                        logger.warning(
                            f"Vector index 'segment_embeddings_index' is not fully populated: "
                            f"{population_percent}%. Vector search may return incomplete results."
                        )
                    
                    # Only warn if dimensions is explicitly set and wrong (not if it's None/0 which means unavailable)
                    if dimensions and dimensions != 768:
                        logger.warning(
                            f"Vector index 'segment_embeddings_index' has dimension {dimensions}, "
                            f"expected 768. Embedding dimension mismatch may cause errors."
                        )
                    elif not dimensions or dimensions == 0:
                        logger.debug(
                            f"Vector index dimensions not available in metadata (this is normal for some Neo4j versions). "
                            f"Assuming 768 dimensions based on embedding model configuration."
                        )
                    
                    logger.info(
                        f"Vector index 'segment_embeddings_index' verified: "
                        f"status={status}, population={population_percent}%"
                        + (f", dimensions={dimensions}" if dimensions else ", dimensions=768 (assumed)")
                    )
                    return True
        except Exception as e:
            logger.warning(f"Could not verify vector index: {e}. Vector search may still work.")
            return False
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def retrieve(self, query: str, top_k: int = 10, query_date: Optional[str] = None) -> Dict:
        """
        Main retrieval method: Hybrid search combining vector and graph.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            query_date: Optional date filter (ISO format). If None, uses current date.
        
        Returns:
            Dictionary with:
            - 'query_analysis': Extracted entities and temporal constraints
            - 'results': List of context objects with text, metadata, and URNs
            - 'result_count': Number of results
        """
        # Step 1: Query Understanding
        logger.info(f"Analyzing query: {query[:100]}...")
        query_analysis = self._analyze_query(query)
        logger.info(f"Query analysis: {query_analysis}")
        
        # Resolve query date
        if not query_date:
            query_date = query_analysis.get('temporal_constraint') or datetime.now().date().isoformat()
        
        # Step 2: Vector Search on Segment embeddings
        logger.info("Performing vector search on Segment nodes (segment_embeddings_index)...")
        vector_results = self._vector_search(query, top_k * 2, query_date)
        logger.info(f"Vector search returned {len(vector_results)} segments")
        
        # Step 3: Graph Traversal for structural relationships
        logger.info("Performing graph traversal...")
        graph_results = self._graph_traversal(vector_results, query_analysis, query_date)
        logger.info(f"Graph traversal enriched {len(graph_results)} results")
        
        # Step 4: Temporal Filtering
        logger.info("Applying temporal filters...")
        filtered_results = self._temporal_filter(graph_results, query_date)
        logger.info(f"After temporal filtering: {len(filtered_results)} results")
        
        # Step 5: Context Packaging
        logger.info("Packaging context...")
        packaged_context = self._package_context(filtered_results, query_analysis)
        
        return {
            'query': query,
            'query_analysis': query_analysis,
            'query_date': query_date,
            'results': packaged_context[:top_k],
            'result_count': len(packaged_context)
        }
    
    def _analyze_query(self, query: str) -> Dict:
        """
        Step 1: Analyze query to extract legal entities and temporal constraints.
        
        Uses LLM (Gemini) if available, otherwise falls back to regex.
        
        Returns:
            Dictionary with:
            - 'legal_entities': List of statute names, section numbers
            - 'temporal_constraint': Date string (ISO format) or None
            - 'query_type': 'case', 'statute', or 'general'
        """
        if self.gemini_client:
            return self._analyze_query_with_llm(query)
        else:
            return self._analyze_query_with_regex(query)
    
    def _analyze_query_with_llm(self, query: str) -> Dict:
        """Use Gemini to extract legal entities and temporal constraints."""
        prompt = f"""You are a legal information extraction system for Sri Lankan law.

Analyze the following legal query and extract:
1. Legal entities: Statute names (e.g., "Bribery Act", "Act No. 19 of 1990"), Section numbers (e.g., "Section 24", "Sec 75")
2. Temporal constraints: Dates or years mentioned (e.g., "in 2024", "as of 2010", "before 2000")
3. Query type: 'case' (if asking about cases/judgments), 'statute' (if asking about laws/acts), or 'general'

Query: {query}

Return ONLY valid JSON in this format:
{{
    "legal_entities": ["statute name or section reference"],
    "temporal_constraint": "YYYY-MM-DD or null",
    "query_type": "case|statute|general"
}}

Example:
Query: "What did the Supreme Court say about Section 24 of the Bribery Act in 2020?"
Response:
{{
    "legal_entities": ["Bribery Act", "Section 24"],
    "temporal_constraint": "2020-01-01",
    "query_type": "case"
}}
"""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt
            )
            
            # Extract JSON from response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Try to extract JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group(0))
            else:
                logger.warning(f"Could not extract JSON from LLM response: {response_text}")
                return self._analyze_query_with_regex(query)
        except Exception as e:
            logger.warning(f"LLM query analysis failed: {e}. Falling back to regex.")
            return self._analyze_query_with_regex(query)
    
    def _analyze_query_with_regex(self, query: str) -> Dict:
        """Fallback: Extract entities using regex patterns."""
        legal_entities = []
        temporal_constraint = None
        
        # Extract statute names (common patterns)
        statute_patterns = [
            r'Act\s+No\.?\s*(\d+)\s+of\s+(\d{4})',  # "Act No. 19 of 1990"
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act',  # "Bribery Act"
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Code',  # "Civil Procedure Code"
        ]
        
        for pattern in statute_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if match.lastindex == 2:  # Act No. X of YYYY
                    legal_entities.append(f"Act No. {match.group(1)} of {match.group(2)}")
                else:
                    legal_entities.append(match.group(1))
        
        # Extract section numbers
        section_pattern = r'Section\s+(\d+)(?:\([^)]+\))?|Sec\.?\s+(\d+)'
        section_matches = re.finditer(section_pattern, query, re.IGNORECASE)
        for match in section_matches:
            section_num = match.group(1) or match.group(2)
            legal_entities.append(f"Section {section_num}")
        
        # Extract temporal constraints
        date_patterns = [
            r'in\s+(\d{4})',  # "in 2024"
            r'as\s+of\s+(\d{4}(?:-\d{2}-\d{2})?)',  # "as of 2024" or "as of 2024-01-01"
            r'(\d{4}-\d{2}-\d{2})',  # ISO date
            r'before\s+(\d{4})',  # "before 2020"
            r'after\s+(\d{4})',  # "after 2020"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                if len(date_str) == 4:  # Just year
                    temporal_constraint = f"{date_str}-01-01"
                else:
                    temporal_constraint = date_str
                break
        
        # Determine query type
        query_lower = query.lower()
        if any(word in query_lower for word in ['case', 'judgment', 'decision', 'precedent', 'court']):
            query_type = 'case'
        elif any(word in query_lower for word in ['statute', 'act', 'section', 'law', 'provision']):
            query_type = 'statute'
        else:
            query_type = 'general'
        
        return {
            'legal_entities': list(set(legal_entities)),  # Remove duplicates
            'temporal_constraint': temporal_constraint,
            'query_type': query_type
        }
    
    def _vector_search(self, query: str, top_k: int, query_date: str) -> List[Dict]:
        """
        Step 2: Vector search on :Segment embeddings.
        
        Uses the segment_embeddings_index for cosine similarity search.
        Nodes have both :Segment and :CaseSegment labels.
        
        Returns:
            List of segment dictionaries with similarity scores
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available. Skipping vector search.")
            return []
        
        try:
            # Generate query embedding (with Nomic prefix)
            query_text = f"search_query: {query}"
            query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True)
            query_embedding_list = query_embedding.tolist()
            
            # Vector search query
            # Try Neo4j 5.11+ vector index syntax first
            # Index is on :Segment label (nodes have both :Segment and :CaseSegment labels)
            # The index already filters by :Segment, so we don't need to check the label again
            # IMPORTANT: Access node properties via node.property, not segment.property
            query_cypher = """
            CALL db.index.vector.queryNodes('segment_embeddings_index', $top_k, $query_embedding)
            YIELD node, score
            WHERE node.embedding IS NOT NULL
            RETURN node.urn AS segment_urn,
                   node.text AS text,
                   node.segment_type AS segment_type,
                   score AS similarity_score
            ORDER BY score DESC
            """
            
            logger.debug(f"Executing vector search with top_k={top_k}, embedding_dim={len(query_embedding_list)}")
            
            with self.driver.session(database=self.database) as session:
                try:
                    result = session.run(query_cypher, {
                        'top_k': top_k,
                        'query_embedding': query_embedding_list
                    })
                    
                    segments = []
                    record_count = 0
                    for record in result:
                        record_count += 1
                        try:
                            segments.append({
                                'segment_urn': record['segment_urn'],
                                'text': record['text'],
                                'segment_type': record.get('segment_type'),
                                'similarity_score': record['similarity_score']
                            })
                        except KeyError as e:
                            logger.error(f"Missing key in vector search result: {e}. Record keys: {list(record.keys())}")
                            continue
                    
                    logger.info(f"Vector search returned {record_count} segments from index")
                except Exception as query_error:
                    logger.error(f"Vector index query failed: {query_error}", exc_info=True)
                    # Check if index exists and is accessible
                    check_index_query = "SHOW VECTOR INDEXES WHERE name = 'segment_embeddings_index'"
                    try:
                        index_check = session.run(check_index_query)
                        index_records = list(index_check)
                        if not index_records:
                            logger.error("Vector index 'segment_embeddings_index' does not exist!")
                        else:
                            for idx_record in index_records:
                                logger.error(f"Index exists but query failed. Index details: {dict(idx_record)}")
                    except Exception as check_error:
                        logger.error(f"Could not check index status: {check_error}")
                    raise
                
                if record_count == 0:
                    # Check if there are any segments in the database (try both labels)
                    check_queries = [
                        ("Segment", "MATCH (s:Segment) WHERE s.embedding IS NOT NULL RETURN count(s) AS segment_count"),
                        ("CaseSegment", "MATCH (s:CaseSegment) WHERE s.embedding IS NOT NULL RETURN count(s) AS segment_count"),
                        ("Any Segment", "MATCH (s) WHERE s:Segment OR s:CaseSegment AND s.embedding IS NOT NULL RETURN count(s) AS segment_count")
                    ]
                    
                    for label_name, check_query in check_queries:
                        check_result = session.run(check_query)
                        check_record = check_result.single()
                        segment_count = check_record['segment_count'] if check_record else 0
                        if segment_count > 0:
                                logger.warning(
                                    f"Found {segment_count} {label_name} nodes with embeddings, but vector search returned 0 results. "
                                    "This may indicate an issue with the vector index label or query syntax."
                                )
                                # Try to get a sample node to debug
                                sample_query = f"MATCH (s:{label_name}) WHERE s.embedding IS NOT NULL RETURN s.urn AS urn, labels(s) AS labels, keys(s) AS keys LIMIT 1"
                                sample_result = session.run(sample_query)
                                sample_record = sample_result.single()
                                if sample_record:
                                    urn = sample_record.get('urn') or 'N/A'
                                    labels = sample_record.get('labels') or []
                                    keys = sample_record.get('keys') or []
                                    logger.warning(
                                        f"Sample {label_name} node: urn={urn}, labels={labels}, keys={keys}"
                                    )
                                    
                                    # Check if node has :Segment label (required for vector index)
                                    if 'Segment' not in labels:
                                        logger.error(
                                            f"❌ CRITICAL: Sample node has labels {labels} but does NOT have :Segment label! "
                                            f"The vector index is on :Segment, so it won't find nodes with only :CaseSegment. "
                                            f"Nodes need BOTH :Segment and :CaseSegment labels. "
                                            f"Falling back to manual similarity search..."
                                        )
                                        # Force fallback to manual search
                                        return self._vector_search_fallback(query, top_k, query_date)
                                break
                    else:
                        logger.error(
                            "❌ No data found in database! "
                            "Vector search returned 0 results because there are no Segment or CaseSegment nodes with embeddings. "
                            "\nPlease run the case law ingestion pipeline first:"
                            "\n  python main.py --model --source cases --cases-new"
                        )
                
                return segments
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            # Fallback: Try alternative vector search syntax
            try:
                logger.info("Attempting fallback vector search...")
                return self._vector_search_fallback(query, top_k, query_date)
            except Exception as e2:
                logger.error(f"Fallback vector search also failed: {e2}", exc_info=True)
                return []
    
    def _vector_search_fallback(self, query: str, top_k: int, query_date: str) -> List[Dict]:
        """Fallback vector search using manual cosine similarity calculation."""
        if not self.embedding_model:
            return []
        
        # Generate query embedding
        query_text = f"search_query: {query}"
        query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True)
        
        # Get all segments with embeddings and calculate similarity manually
        # Query both :Segment and :CaseSegment labels
        query_cypher = """
        MATCH (s)
        WHERE (s:Segment OR s:CaseSegment) AND s.embedding IS NOT NULL
        RETURN s.urn AS segment_urn,
               s.text AS text,
               s.segment_type AS segment_type,
               s.embedding AS embedding
        LIMIT 1000
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query_cypher)
            
            segments_with_scores = []
            for record in result:
                segment_embedding = record['embedding']
                if segment_embedding:
                    # Calculate cosine similarity
                    import numpy as np
                    similarity = np.dot(query_embedding, segment_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(segment_embedding)
                    )
                    
                    segments_with_scores.append({
                        'segment_urn': record['segment_urn'],
                        'text': record['text'],
                        'segment_type': record['segment_type'],
                        'similarity_score': float(similarity)
                    })
            
            # Sort by similarity and return top_k
            segments_with_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
            return segments_with_scores[:top_k]
    
    def _graph_traversal(
        self,
        vector_results: List[Dict],
        query_analysis: Dict,
        query_date: str
    ) -> List[Dict]:
        """
        Step 3: Graph traversal to find parent :Case and linked :Provision nodes.
        
        For each segment from vector search:
        1. Find parent :Case node via :HAS_SEGMENT
        2. Find linked :Provision nodes via :APPLIES_TO (from :Judgment)
        3. Prioritize segments linked to mentioned statutes
        
        Returns:
            Enriched results with case and provision information
        """
        if not vector_results:
            return []
        
        segment_urns = [r['segment_urn'] for r in vector_results]
        legal_entities = query_analysis.get('legal_entities', [])
        
        # Query :CaseSegment label for graph traversal (nodes also have :Segment label)
        query_cypher = """
        MATCH (s:CaseSegment)
        WHERE s.urn IN $segment_urns
        OPTIONAL MATCH (c:Case)-[:HAS_SEGMENT]->(s)
        OPTIONAL MATCH (j:Judgment)-[:EXPRESSION_OF]->(c)
        OPTIONAL MATCH (j)-[:APPLIES_TO]->(p:Provision)
        RETURN s.urn AS segment_urn,
               s.text AS segment_text,
               s.segment_type AS segment_type,
               c.urn AS case_urn,
               c.title AS case_title,
               c.case_number AS case_number,
               c.decision_date AS decision_date,
               c.court AS court,
               collect(DISTINCT {
                   urn: p.urn,
                   label: p.label,
                   text: p.text
               }) AS provisions
        ORDER BY s.urn
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query_cypher, {'segment_urns': segment_urns})
            
            enriched_results = []
            for record in result:
                # Get original similarity score
                original_result = next(
                    (r for r in vector_results if r['segment_urn'] == record['segment_urn']),
                    None
                )
                similarity_score = original_result['similarity_score'] if original_result else 0.0
                
                # Check if linked to mentioned statute (boost score)
                provisions = record['provisions'] or []
                statute_match_boost = 0.0
                if legal_entities and provisions:
                    for prov in provisions:
                        prov_label = (prov.get('label') or '').lower()
                        prov_text = (prov.get('text') or '').lower()
                        for entity in legal_entities:
                            entity_lower = entity.lower()
                            if entity_lower in prov_label or entity_lower in prov_text:
                                statute_match_boost = 0.3  # Boost for statute match
                                break
                
                enriched_results.append({
                    'segment_urn': record['segment_urn'],
                    'segment_text': record['segment_text'],
                    'segment_type': record['segment_type'],
                    'similarity_score': similarity_score + statute_match_boost,
                    'case_urn': record['case_urn'],
                    'case_title': record['case_title'],
                    'case_number': record['case_number'],
                    'decision_date': record['decision_date'],
                    'court': record['court'],
                    'provisions': [p for p in provisions if p.get('urn')]  # Filter out None
                })
            
            return enriched_results
    
    def _temporal_filter(self, results: List[Dict], query_date: str) -> List[Dict]:
        """
        Step 4: Apply temporal filtering based on decision_date.
        
        Filters out cases decided after query_date.
        Prioritizes cases with decision_date matching or before query_date.
        
        Args:
            results: List of enriched results
            query_date: Query date in ISO format
        
        Returns:
            Filtered and prioritized results
        """
        try:
            query_dt = datetime.fromisoformat(query_date).date() if isinstance(query_date, str) else query_date
        except (ValueError, TypeError):
            logger.warning(f"Invalid query_date: {query_date}. Using current date.")
            query_dt = datetime.now().date()
        
        filtered = []
        for result in results:
            decision_date = result.get('decision_date')
            
            # Skip if no decision_date
            if not decision_date:
                result['temporal_score'] = 0.5  # Neutral score
                filtered.append(result)
                continue
            
            # Convert decision_date to date object
            try:
                if isinstance(decision_date, str):
                    if 'T' in decision_date:
                        decision_dt = datetime.fromisoformat(decision_date.replace('Z', '+00:00')).date()
                    else:
                        decision_dt = datetime.fromisoformat(decision_date).date()
                elif isinstance(decision_date, date):
                    decision_dt = decision_date
                else:
                    decision_dt = None
            except (ValueError, TypeError):
                logger.warning(f"Could not parse decision_date: {decision_date}")
                decision_dt = None
            
            if decision_dt:
                # Filter: exclude cases decided after query_date
                if decision_dt > query_dt:
                    continue  # Skip future cases
                
                # Calculate temporal relevance score
                days_diff = (query_dt - decision_dt).days
                if days_diff < 0:
                    result['temporal_score'] = 0.0
                elif days_diff == 0:
                    result['temporal_score'] = 1.0  # Exact match
                elif days_diff <= 365:
                    result['temporal_score'] = 0.9  # Within 1 year
                elif days_diff <= 1825:
                    result['temporal_score'] = 0.7  # Within 5 years
                else:
                    result['temporal_score'] = 0.5  # Older
                
                # Boost similarity score with temporal relevance
                result['similarity_score'] = result.get('similarity_score', 0.0) + (result['temporal_score'] * 0.2)
            else:
                result['temporal_score'] = 0.5
            
            filtered.append(result)
        
        # Sort by combined score (similarity + temporal)
        filtered.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        
        return filtered
    
    def _package_context(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """
        Step 5: Package context with structured evidence and URNs.
        
        Creates a structured context object for each result with:
        - Segment text and metadata
        - Parent Case information
        - Linked Provision information
        - URNs for citation
        
        Returns:
            List of packaged context dictionaries
        """
        packaged = []
        
        for result in results:
            context = {
                'evidence': {
                    'segment': {
                        'urn': result['segment_urn'],
                        'text': result['segment_text'],
                        'type': result['segment_type'],
                        'similarity_score': result.get('similarity_score', 0.0)
                    },
                    'case': {
                        'urn': result.get('case_urn'),
                        'title': result.get('case_title'),
                        'case_number': result.get('case_number'),
                        'decision_date': result.get('decision_date'),
                        'court': result.get('court')
                    },
                    'provisions': []
                },
                'metadata': {
                    'temporal_score': result.get('temporal_score', 0.5),
                    'combined_score': result.get('similarity_score', 0.0)
                }
            }
            
            # Add linked provisions
            for prov in result.get('provisions', []):
                context['evidence']['provisions'].append({
                    'urn': prov.get('urn'),
                    'label': prov.get('label'),
                    'text': prov.get('text', '')[:500] if prov.get('text') else ''  # Truncate long text
                })
            
            packaged.append(context)
        
        return packaged
