"""
CaseLawIngestor: Ingests JSON from CaseLawPipeline into Neo4j
=============================================================

Transforms case law JSON objects into a Hybrid Vector-Graph structure:
- Case nodes (Work anchor)
- Judgment nodes (Expression)
- Segment nodes (Facts, Ratio with embeddings)
- Links to Provision nodes via fuzzy matching
"""

from typing import Dict, List, Optional
from neo4j import GraphDatabase  # pyright: ignore[reportMissingImports]
import logging
import re

logger = logging.getLogger(__name__)


class CaseLawIngestor:
    """
    Ingests case law JSON data from CaseLawPipeline into Neo4j.
    
    Creates:
    - :Case nodes (Work anchor)
    - :Judgment nodes (Expression)
    - :Segment nodes (Facts, Ratio with embeddings)
    - Links to :Provision nodes via fuzzy matching
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize CaseLawIngestor with Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Initialize vector index
        self._ensure_vector_index()
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def _ensure_vector_index(self):
        """
        Ensure vector index exists for Segment embeddings.
        Creates index if it doesn't exist.
        
        The index is created on :Segment nodes (which CaseSegment nodes also match via label inheritance).
        """
        with self.driver.session(database=self.database) as session:
            # Check if index exists and verify its status
            check_query = """
            SHOW VECTOR INDEXES
            WHERE name = 'segment_embeddings_index'
            """
            
            result = session.run(check_query)
            records = list(result)
            index_exists = len(records) > 0
            
            # Verify index status if it exists
            if index_exists:
                for record in records:
                    status = record.get('state', 'UNKNOWN')
                    population_percent = record.get('populationPercent', 0.0)
                    if status != 'ONLINE' or population_percent < 100.0:
                        logger.warning(
                            f"Vector index 'segment_embeddings_index' exists but is not ready: "
                            f"status={status}, population={population_percent}%"
                        )
                    else:
                        logger.info(
                            f"Vector index 'segment_embeddings_index' is ONLINE and fully populated ({population_percent}%)"
                        )
            
            if not index_exists:
                create_query = """
                CREATE VECTOR INDEX segment_embeddings_index IF NOT EXISTS
                FOR (s:Segment)
                ON s.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
                
                try:
                    session.run(create_query)
                    logger.info("Created vector index: segment_embeddings_index for :Segment nodes (768 dimensions, cosine)")
                except Exception as e:
                    # Fallback: Try alternative syntax for older Neo4j versions
                    logger.warning(f"Could not create vector index with standard syntax: {e}")
                    try:
                        # Alternative: Create regular index (for older Neo4j)
                        alt_query = """
                        CREATE INDEX segment_embeddings_index IF NOT EXISTS
                        FOR (s:Segment)
                        ON (s.embedding)
                        """
                        session.run(alt_query)
                        logger.info("Created regular index for embeddings (vector search may not be available)")
                    except Exception as e2:
                        logger.warning(f"Could not create index: {e2}. Vector search may not be available.")
    
    def ingest_case(self, json_data: Dict) -> Dict[str, any]:
        """
        Main entry point: Ingest a single case law JSON into Neo4j.
        
        Args:
            json_data: Dictionary with 'urn', 'meta', 'content', 'vectors', 'edges'
        
        Returns:
            Dictionary with ingestion statistics:
            {
                'case_uid': str,
                'judgment_uid': str,
                'segments_created': int,
                'statute_links': int,
                'warnings': List[str]
            }
        """
        if 'urn' not in json_data:
            raise ValueError("json_data must contain 'urn' key")
        
        urn = json_data['urn']
        meta = json_data.get('meta', {})
        content = json_data.get('content', {})
        vectors = json_data.get('vectors', {})
        edges = json_data.get('edges', {})
        
        logger.info(f"Ingesting case: {urn}")
        
        stats = {
            'case_uid': urn,
            'judgment_uid': None,
            'segments_created': 0,
            'statute_links': 0,
            'warnings': []
        }
        
        with self.driver.session(database=self.database) as session:
            # Step 1: Create/Update Case node
            case_uid = self._create_case_node(session, urn, meta)
            stats['case_uid'] = case_uid
            
            # Step 2: Create/Update Judgment node
            judgment_uid = self._create_judgment_node(session, urn, meta, content)
            stats['judgment_uid'] = judgment_uid
            
            # Step 3: Link Judgment to Case
            self._link_judgment_to_case(session, judgment_uid, case_uid)
            
            # Step 4: Create CaseSegment nodes (Facts, Ratio, Legal Issues, etc.)
            # Segments link directly to Case via HAS_SEGMENT (not to Judgment)
            segments_created = self._create_segment_nodes(session, case_uid, content, vectors)
            stats['segments_created'] = segments_created
            
            # Step 5: Link to Provisions (fuzzy matching)
            statute_links, warnings = self._link_to_provisions(session, judgment_uid, edges.get('statutes', []))
            stats['statute_links'] = statute_links
            stats['warnings'] = warnings
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def _create_case_node(self, session, urn: str, meta: Dict) -> str:
        """
        Create or update Case node (idempotent).
        
        Args:
            session: Neo4j session
            urn: Case URN
            meta: Metadata dictionary
        
        Returns:
            Case URN
        """
        # Extract title from parties or case number
        title = self._extract_case_title(meta)
        case_number = meta.get('case_number', '')
        court = meta.get('court', 'Supreme Court')
        
        # Extract decision_date from meta (stored as Date property for TemporalReconstruction)
        decision_date_str = meta.get('decision_date') or meta.get('date')
        
        # Convert string to Neo4j date() type in Cypher for temporal queries
        # Use date() function to ensure proper date type, not string
        query = """
        MERGE (c:Case {urn: $urn})
        ON CREATE SET
            c.title = $title,
            c.case_number = $case_number,
            c.court = $court,
            c.decision_date = CASE WHEN $decision_date_str IS NULL THEN NULL ELSE date($decision_date_str) END,
            c.created_at = datetime()
        ON MATCH SET
            c.title = $title,
            c.case_number = $case_number,
            c.court = $court,
            c.decision_date = CASE WHEN $decision_date_str IS NULL THEN NULL ELSE date($decision_date_str) END,
            c.updated_at = datetime()
        RETURN c.urn AS urn
        """
        
        result = session.run(query, {
            'urn': urn,
            'title': title,
            'case_number': case_number,
            'court': court,
            'decision_date_str': decision_date_str
        })
        
        record = result.single()
        return record['urn'] if record else urn
    
    def _extract_case_title(self, meta: Dict) -> str:
        """
        Extract case title from metadata.
        
        Args:
            meta: Metadata dictionary
        
        Returns:
            Case title string
        """
        parties = meta.get('parties', {})
        petitioner = parties.get('petitioner', '')
        respondent = parties.get('respondent', '')
        
        if petitioner and respondent:
            return f"{petitioner} v. {respondent}"
        elif petitioner:
            return petitioner
        elif respondent:
            return respondent
        else:
            # Fallback to case number
            return meta.get('case_number', 'Unknown Case')
    
    def _create_judgment_node(self, session, urn: str, meta: Dict, content: Dict) -> str:
        """
        Create or update Judgment node (idempotent).
        
        Args:
            session: Neo4j session
            urn: Case URN
            meta: Metadata dictionary
            content: Content dictionary
        
        Returns:
            Judgment URN (urn + !main)
        """
        judgment_uid = f"{urn}!main"
        date = meta.get('date')
        outcome = content.get('outcome', '')
        
        # Clean outcome text
        outcome = self._clean_text(outcome)
        
        query = """
        MERGE (j:Judgment {urn: $judgment_uid})
        ON CREATE SET
            j.date = $date,
            j.outcome = $outcome,
            j.status = 'current',
            j.created_at = datetime()
        ON MATCH SET
            j.date = $date,
            j.outcome = $outcome,
            j.updated_at = datetime()
        RETURN j.urn AS urn
        """
        
        result = session.run(query, {
            'judgment_uid': judgment_uid,
            'date': date,
            'outcome': outcome
        })
        
        record = result.single()
        return record['urn'] if record else judgment_uid
    
    def _link_judgment_to_case(self, session, judgment_uid: str, case_uid: str):
        """
        Create EXPRESSION_OF relationship between Judgment and Case.
        
        Args:
            session: Neo4j session
            judgment_uid: Judgment URN
            case_uid: Case URN
        """
        query = """
        MATCH (j:Judgment {urn: $judgment_uid})
        MATCH (c:Case {urn: $case_uid})
        MERGE (j)-[r:EXPRESSION_OF]->(c)
        ON CREATE SET r.created_at = datetime()
        RETURN r
        """
        
        session.run(query, {
            'judgment_uid': judgment_uid,
            'case_uid': case_uid
        })
    
    def _create_segment_nodes(self, session, case_uid: str, content: Dict, vectors: Dict) -> int:
        """
        Create CaseSegment nodes for all semantic segments (Facts, Ratio, Legal Issues, etc.).
        Each segment is linked to the Case node via HAS_SEGMENT relationship.
        
        Args:
            session: Neo4j session
            case_uid: Case URN (not judgment_uid - segments link directly to Case)
            content: Content dictionary with all segments
            vectors: Vectors dictionary
        
        Returns:
            Number of segments created
        """
        segments_created = 0
        
        # Define all segment types to create
        # Note: Handle both 'ratio' and 'ratio_decidendi' keys (ratio_decidendi is preferred)
        segment_types = [
            ('facts', 'Facts', 'facts_embedding'),
            ('ratio_decidendi', 'Ratio', 'ratio_embedding'),  # Preferred key from Gemini
            ('ratio', 'Ratio', 'ratio_embedding'),  # Fallback if ratio_decidendi not present
            ('legal_issues', 'LegalIssues', None),  # No embedding for now
            ('arguments', 'Arguments', None),
            ('obiter_dicta', 'ObiterDicta', None),
            ('outcome', 'Outcome', None)
        ]
        
        # Track which segments we've already created to avoid duplicates
        created_segments = set()
        
        for content_key, segment_label, embedding_key in segment_types:
            # Skip if we already created a segment of this type (e.g., ratio vs ratio_decidendi)
            if segment_label in created_segments:
                continue
                
            # Try primary key first, then fallback
            segment_text = content.get(content_key, '')
            if not segment_text and content_key == 'ratio':
                # Try ratio_decidendi as fallback
                segment_text = content.get('ratio_decidendi', '')
            
            if not segment_text:
                continue
            
            # Get embedding if available
            embedding = vectors.get(embedding_key) if embedding_key else None
            
            # Create segment UID (use consistent key for ratio)
            if content_key in ['ratio', 'ratio_decidendi']:
                segment_uid = f"{case_uid}!ratio"
            else:
                segment_uid = f"{case_uid}!{content_key}"
            
            if self._create_case_segment(session, segment_uid, case_uid, segment_label, segment_text, embedding):
                segments_created += 1
                created_segments.add(segment_label)  # Mark as created
        
        return segments_created
    
    def _create_case_segment(self, session, segment_uid: str, case_uid: str, 
                             segment_type: str, text: str, embedding: Optional[List[float]] = None) -> bool:
        """
        Create a single CaseSegment node linked to Case via HAS_SEGMENT.
        
        Args:
            session: Neo4j session
            segment_uid: Segment URN
            case_uid: Case URN (segments link directly to Case, not Judgment)
            segment_type: Type of segment ('Facts', 'Ratio', 'LegalIssues', etc.)
            text: Segment text
            embedding: Optional embedding vector (768 dimensions)
        
        Returns:
            True if created, False otherwise
        """
        # Clean text
        text = self._clean_text(text)
        if not text:
            return False
        
        # Validate embedding if provided
        if embedding is not None and len(embedding) != 768:
            logger.warning(f"Embedding dimension mismatch: expected 768, got {len(embedding)}. Skipping embedding.")
            embedding = None
        
        # Build query - only set embedding if provided
        # Note: Node has both :Segment (for vector index) and :CaseSegment (for graph structure)
        if embedding:
            query = f"""
            MERGE (s:Segment:CaseSegment:{segment_type} {{urn: $segment_uid}})
            ON CREATE SET
                s.text = $text,
                s.embedding = $embedding,
                s.segment_type = $segment_type,
                s.created_at = datetime()
            ON MATCH SET
                s.text = $text,
                s.embedding = $embedding,
                s.updated_at = datetime()
            WITH s
            MATCH (c:Case {{urn: $case_uid}})
            MERGE (c)-[r:HAS_SEGMENT]->(s)
            ON CREATE SET r.created_at = datetime()
            RETURN s.urn AS urn
            """
            params = {
                'segment_uid': segment_uid,
                'case_uid': case_uid,
                'text': text,
                'embedding': embedding,
                'segment_type': segment_type
            }
        else:
            query = f"""
            MERGE (s:CaseSegment:{segment_type} {{urn: $segment_uid}})
            ON CREATE SET
                s.text = $text,
                s.segment_type = $segment_type,
                s.created_at = datetime()
            ON MATCH SET
                s.text = $text,
                s.updated_at = datetime()
            WITH s
            MATCH (c:Case {{urn: $case_uid}})
            MERGE (c)-[r:HAS_SEGMENT]->(s)
            ON CREATE SET r.created_at = datetime()
            RETURN s.urn AS urn
            """
            params = {
                'segment_uid': segment_uid,
                'case_uid': case_uid,
                'text': text,
                'segment_type': segment_type
            }
        
        result = session.run(query, params)
        return result.single() is not None
    
    def _link_to_provisions(self, session, judgment_uid: str, statute_citations: List[str]) -> tuple[int, List[str]]:
        """
        Link Judgment to Provision nodes via fuzzy matching.
        
        Args:
            session: Neo4j session
            judgment_uid: Judgment URN
            statute_citations: List of citation strings
        
        Returns:
            Tuple of (number of links created, list of warnings)
        """
        links_created = 0
        warnings = []
        
        for citation in statute_citations:
            # Try to find matching Provision
            provision_urn = self._fuzzy_match_provision(session, citation)
            
            if provision_urn:
                # Create APPLIES_TO relationship
                # Note: Both Provision and Judgment nodes use 'urn' property
                query = """
                MATCH (j:Judgment {urn: $judgment_uid})
                MATCH (p:Provision {urn: $provision_urn})
                MERGE (j)-[r:APPLIES_TO]->(p)
                ON CREATE SET r.citation_text = $citation, r.created_at = datetime()
                RETURN r
                """
                
                result = session.run(query, {
                    'judgment_uid': judgment_uid,
                    'provision_urn': provision_urn,
                    'citation': citation
                })
                
                if result.single():
                    links_created += 1
            else:
                warning = f"Statute Citation Missing: {citation}"
                warnings.append(warning)
                logger.warning(warning)
        
        return links_created, warnings
    
    def _fuzzy_match_provision(self, session, citation: str) -> Optional[str]:
        """
        Perform fuzzy lookup to find matching Provision node.
        
        Strategy:
        1. Extract section number and act name from citation
        2. Try exact matches first
        3. Try fuzzy matches on label/text
        
        Args:
            session: Neo4j session
            citation: Citation string (e.g., "Proceeds of Crime Act, Sec 75")
        
        Returns:
            Provision URN if found, None otherwise
        """
        # Normalize citation
        citation_lower = citation.lower()
        
        # Extract section number
        section_match = re.search(r'sec(?:tion)?\s+(\d+)', citation_lower)
        section_num = section_match.group(1) if section_match else None
        
        # Extract act name (everything before "Sec" or "Section")
        act_match = re.search(r'^(.+?)(?:\s*,\s*sec|$)', citation_lower)
        act_name = act_match.group(1).strip() if act_match else citation_lower
        
        # Strategy 1: Try to match by section number in URN
        if section_num:
            query = """
            MATCH (p:Provision)
            WHERE p.urn CONTAINS $section_pattern
            RETURN p.urn AS urn, p.label AS label
            LIMIT 10
            """
            
            section_pattern = f"!sec{section_num}"
            result = session.run(query, {'section_pattern': section_pattern})
            
            # Try to match act name in label
            for record in result:
                label = record.get('label', '').lower()
                if act_name in label or any(word in label for word in act_name.split() if len(word) > 3):
                    return record['urn']
        
        # Strategy 2: Fuzzy match on label
        query = """
        MATCH (p:Provision)
        WHERE p.label IS NOT NULL
        RETURN p.urn AS urn, p.label AS label
        LIMIT 100
        """
        
        result = session.run(query)
        
        # Score matches
        best_match = None
        best_score = 0
        
        for record in result:
            label = record.get('label', '').lower()
            
            # Calculate similarity score
            score = self._calculate_similarity(citation_lower, label, act_name, section_num)
            
            if score > best_score and score > 0.5:  # Threshold for match
                best_score = score
                best_match = record['urn']
        
        return best_match
    
    def _calculate_similarity(self, citation: str, label: str, act_name: str, section_num: Optional[str]) -> float:
        """
        Calculate similarity score between citation and label.
        
        Args:
            citation: Full citation string
            label: Provision label
            act_name: Extracted act name
            section_num: Extracted section number
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        score = 0.0
        
        # Check if section number matches
        if section_num and section_num in label:
            score += 0.4
        
        # Check if act name words match
        act_words = [w for w in act_name.split() if len(w) > 3]
        matched_words = sum(1 for word in act_words if word in label)
        if act_words:
            score += 0.3 * (matched_words / len(act_words))
        
        # Check for common act keywords
        act_keywords = ['act', 'code', 'law', 'ordinance']
        if any(keyword in label for keyword in act_keywords if keyword in citation):
            score += 0.2
        
        # Check if citation substring is in label
        if len(citation) > 10:
            # Try first 20 chars of citation
            citation_prefix = citation[:20]
            if citation_prefix in label:
                score += 0.1
        
        return min(score, 1.0)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text of excessive whitespace.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def ingest_cases(self, json_data_list: List[Dict]) -> Dict[str, any]:
        """
        Ingest multiple case law JSON objects.
        
        Args:
            json_data_list: List of case law JSON dictionaries
        
        Returns:
            Dictionary with aggregate statistics
        """
        total_stats = {
            'cases_processed': 0,
            'total_segments': 0,
            'total_statute_links': 0,
            'total_warnings': 0,
            'warnings': []
        }
        
        for json_data in json_data_list:
            try:
                stats = self.ingest_case(json_data)
                total_stats['cases_processed'] += 1
                total_stats['total_segments'] += stats['segments_created']
                total_stats['total_statute_links'] += stats['statute_links']
                total_stats['total_warnings'] += len(stats['warnings'])
                total_stats['warnings'].extend(stats['warnings'])
            except Exception as e:
                logger.error(f"Error ingesting case {json_data.get('urn', 'unknown')}: {e}")
                total_stats['warnings'].append(f"Error processing case: {e}")
        
        logger.info(f"Batch ingestion complete: {total_stats}")
        return total_stats


if __name__ == "__main__":
    # Example usage
    from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
    
    sample_json = {
        "urn": "urn:lex:lk:case:sc:appeal:65:2025",
        "meta": {
            "case_number": "SC/APPEAL/65/2025",
            "date": "2025-10-10",
            "court": "Supreme Court",
            "parties": {
                "petitioner": "Merchant Bank",
                "respondent": "Perera"
            }
        },
        "content": {
            "facts": "The plaintiff instituted this action...",
            "ratio": "The jurisdiction is vested exclusively...",
            "outcome": "Appeal Allowed"
        },
        "vectors": {
            "facts_embedding": [0.012] * 768,  # Placeholder
            "ratio_embedding": [0.88] * 768    # Placeholder
        },
        "edges": {
            "statutes": ["Civil Procedure Code, Sec 142"]
        }
    }
    
    ingestor = CaseLawIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    
    try:
        stats = ingestor.ingest_case(sample_json)
        print(f"Ingestion complete: {stats}")
    finally:
        ingestor.close()
