"""
CRUD operations for Neo4j with temporal awareness.
Handles cases, statutes, versions, sections, and temporal relationships.
"""

from typing import Dict, List, Optional
from datetime import datetime, date
from neo4j import GraphDatabase


class TemporalNeo4jOps:
    """Neo4j operations with temporal awareness."""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def _execute_write(self, query: str, parameters: Dict = None):
        """Execute a write query and return single record data."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            # Extract data before session closes to avoid ResultConsumedError
            record = result.single()
            return record.data() if record else None
    
    def _execute_read(self, query: str, parameters: Dict = None):
        """Execute a read query."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    # ============================================
    # CASE OPERATIONS
    # ============================================
    
    def create_case_with_temporal_metadata(self, case_data: Dict) -> str:
        """
        Create a case node with temporal metadata.
        
        Args:
            case_data: Dict with case_id, name, court, decision_date, etc.
        
        Returns:
            case_id
        """
        query = """
        CREATE (c:Case {
            case_id: $case_id,
            name: $name,
            court: $court,
            decision_date: date($decision_date),
            effective_from: date($effective_from),
            effective_to: $effective_to,
            status: $status,
            summary: $summary,
            judges: $judges
        })
        RETURN c.case_id AS case_id
        """
        
        parameters = {
            'case_id': case_data['case_id'],
            'name': case_data.get('case_name'),
            'court': case_data.get('court'),
            'decision_date': case_data.get('decision_date'),
            'effective_from': case_data.get('effective_from') or case_data.get('decision_date'),
            'effective_to': case_data.get('effective_to'),
            'status': case_data.get('status', 'active'),
            'summary': case_data.get('summary'),
            'judges': case_data.get('judges', [])
        }
        
        result = self._execute_write(query, parameters)
        return result.single()['case_id']
    
    def update_case_status(self, case_id: str, status: str, effective_to: Optional[str] = None):
        """Update case status (e.g., mark as overruled)."""
        query = """
        MATCH (c:Case {case_id: $case_id})
        SET c.status = $status
        SET c.effective_to = $effective_to
        RETURN c.case_id AS case_id
        """
        
        parameters = {
            'case_id': case_id,
            'status': status,
            'effective_to': effective_to
        }
        
        self._execute_write(query, parameters)
    
    # ============================================
    # STATUTE OPERATIONS
    # ============================================
    
    def create_statute(self, statute_data: Dict) -> str:
        """Create a statute node."""
        query = """
        CREATE (s:Statute {
            statute_id: $statute_id,
            title: $title,
            act_number: $act_number,
            act_year: $act_year,
            enactment_date: date($enactment_date),
            effective_date: date($effective_date),
            last_amended: $last_amended,
            version: $version,
            status: $status
        })
        RETURN s.statute_id AS statute_id
        """
        
        parameters = {
            'statute_id': statute_data['statute_id'],
            'title': statute_data.get('title'),
            'act_number': statute_data.get('act_number'),
            'act_year': statute_data.get('act_year'),
            'enactment_date': statute_data.get('enactment_date'),
            'effective_date': statute_data.get('effective_date'),
            'last_amended': statute_data.get('last_amended'),
            'version': statute_data.get('version', '1.0'),
            'status': statute_data.get('status', 'active')
        }
        
        result = self._execute_write(query, parameters)
        return result['statute_id'] if result else None
    
    def create_statute_version(self, version_data: Dict) -> str:
        """
        Create a statute version node and link to parent statute.
        
        Args:
            version_data: Dict with version_id, statute_id, version, effective_from, etc.
        
        Returns:
            version_id
        """
        query = """
        MATCH (s:Statute {statute_id: $statute_id})
        CREATE (sv:StatuteVersion {
            version_id: $version_id,
            statute_id: $statute_id,
            version: $version,
            effective_from: date($effective_from),
            effective_to: $effective_to,
            amendment_gazette_id: $amendment_gazette_id,
            content_hash: $content_hash
        })
        CREATE (s)-[:HAS_VERSION]->(sv)
        RETURN sv.version_id AS version_id
        """
        
        parameters = {
            'version_id': version_data['version_id'],
            'statute_id': version_data['statute_id'],
            'version': version_data.get('version'),
            'effective_from': version_data.get('effective_from'),
            'effective_to': version_data.get('effective_to'),
            'amendment_gazette_id': version_data.get('amendment_gazette_id'),
            'content_hash': version_data.get('content_hash')
        }
        
        result = self._execute_write(query, parameters)
        return result['version_id'] if result else None
    
    def get_statute_version_at_date(self, statute_id: str, query_date: str) -> Optional[Dict]:
        """
        Get the statute version that was effective at a specific date.
        
        Args:
            statute_id: Statute identifier
            query_date: Date in ISO format (YYYY-MM-DD)
        
        Returns:
            Version dict or None
        """
        query = """
        MATCH (s:Statute {statute_id: $statute_id})-[:HAS_VERSION]->(sv:StatuteVersion)
        WHERE sv.effective_from <= date($query_date)
          AND (sv.effective_to IS NULL OR sv.effective_to >= date($query_date))
        RETURN sv
        ORDER BY sv.effective_from DESC
        LIMIT 1
        """
        
        result = self._execute_read(query, {'statute_id': statute_id, 'query_date': query_date})
        return result[0] if result else None
    
    # ============================================
    # SECTION OPERATIONS
    # ============================================
    
    def create_section(self, section_data: Dict) -> str:
        """Create a section node and link to statute version."""
        query = """
        MATCH (sv:StatuteVersion {version_id: $version_id})
        CREATE (sec:Section {
            section_id: $section_id,
            section_number: $section_number,
            parent_section_id: $parent_section_id,
            statute_id: $statute_id,
            effective_from: date($effective_from),
            effective_to: $effective_to,
            content: $content
        })
        CREATE (sv)-[:CONTAINS]->(sec)
        RETURN sec.section_id AS section_id
        """
        
        parameters = {
            'section_id': section_data['section_id'],
            'version_id': section_data['version_id'],
            'section_number': section_data['section_number'],
            'parent_section_id': section_data.get('parent_section_id'),
            'statute_id': section_data['statute_id'],
            'effective_from': section_data.get('effective_from'),
            'effective_to': section_data.get('effective_to'),
            'content': section_data.get('content')
        }
        
        result = self._execute_write(query, parameters)
        return result['section_id'] if result else None
    
    # ============================================
    # TEMPORAL RELATIONSHIP OPERATIONS
    # ============================================
    
    def link_temporal_relationship(self, relationship_data: Dict):
        """
        Create a temporal relationship between nodes.
        
        Args:
            relationship_data: Dict with type, source_id, target_id, and temporal properties
        """
        rel_type = relationship_data['type']
        source_type = relationship_data['source_type']  # 'Case' or 'StatuteVersion'
        target_type = relationship_data['target_type']
        source_id_field = f"{source_type.lower()}_id"
        target_id_field = f"{target_type.lower()}_id"
        
        # Build relationship properties
        props = []
        for key, value in relationship_data.items():
            if key not in ['type', 'source_type', 'target_type', 'source_id', 'target_id']:
                if 'date' in key.lower() and value:
                    props.append(f"{key}: date('{value}')")
                else:
                    props.append(f"{key}: ${key}")
        
        props_str = ", ".join(props) if props else ""
        props_str = f" {{{props_str}}}" if props_str else ""
        
        query = f"""
        MATCH (source:{source_type} {{{source_id_field}: $source_id}})
        MATCH (target:{target_type} {{{target_id_field}: $target_id}})
        CREATE (source)-[r:{rel_type}{props_str}]->(target)
        RETURN r
        """
        
        parameters = {
            'source_id': relationship_data['source_id'],
            'target_id': relationship_data['target_id'],
            **{k: v for k, v in relationship_data.items() 
               if k not in ['type', 'source_type', 'target_type', 'source_id', 'target_id'] and 'date' not in k.lower()}
        }
        
        self._execute_write(query, parameters)
    
    # ============================================
    # TEMPORAL QUERY OPERATIONS
    # ============================================
    
    def get_active_law_as_of_date(self, query_date: str, doc_type: str = None) -> List[Dict]:
        """
        Get all active cases/statutes as of a specific date.
        
        Args:
            query_date: Date in ISO format
            doc_type: 'case' or 'statute' (None for both)
        
        Returns:
            List of active documents
        """
        if doc_type == 'case':
            query = """
            MATCH (c:Case)
            WHERE c.effective_from <= date($query_date)
              AND (c.effective_to IS NULL OR c.effective_to >= date($query_date))
              AND c.status = 'active'
            RETURN c
            ORDER BY c.decision_date DESC
            """
        elif doc_type == 'statute':
            # Updated for LRMoo Schema (Work -> Expression)
            query = """
            MATCH (w:Work)-[:REALIZED_AS]->(e:Expression)
            WHERE e.start_date <= date($query_date)
              AND (e.end_date IS NULL OR e.end_date > date($query_date))
              AND e.status = 'active'
            RETURN {
                statute_id: w.work_id,
                title: w.title,
                text: e.text,
                section_number: e.section_number,
                version: e.version,
                source_doc_type: 'statute',
                effective_from: toString(e.start_date),
                effective_to: toString(e.end_date)
            } AS s
            """
        else:
            # Return both
            query = """
            MATCH (c:Case)
            WHERE c.effective_from <= date($query_date)
              AND (c.effective_to IS NULL OR c.effective_to >= date($query_date))
              AND c.status = 'active'
            WITH {
                case_id: c.case_id,
                title: c.name,
                text: c.summary,
                source_doc_type: 'case',
                effective_from: toString(c.effective_from)
            } AS result
            RETURN result
            UNION
            MATCH (w:Work)-[:REALIZED_AS]->(e:Expression)
            WHERE e.start_date <= date($query_date)
              AND (e.end_date IS NULL OR e.end_date > date($query_date))
              AND e.status = 'active'
            WITH {
                statute_id: w.work_id,
                title: w.title,
                text: e.text,
                section_number: e.section_number,
                version: e.version,
                source_doc_type: 'statute',
                effective_from: toString(e.start_date),
                effective_to: toString(e.end_date)
            } AS result
            RETURN result
            """
            return [record['result'] for record in self._execute_read(query, {'query_date': query_date})]
        
        return self._execute_read(query, {'query_date': query_date})
    
    def get_overruled_cases_after_date(self, query_date: str) -> List[Dict]:
        """Get all cases that were overruled after a specific date."""
        query = """
        MATCH (c:Case)-[r:OVERRULES]->(overruled:Case)
        WHERE r.overruled_date >= date($query_date)
        RETURN c, overruled, r.overruled_date AS overruled_date
        ORDER BY r.overruled_date DESC
        """
        
        return self._execute_read(query, {'query_date': query_date})
    
    def get_amendment_history(self, statute_id: str) -> List[Dict]:
        """Get chronological amendment history of a statute."""
        query = """
        MATCH (s:Statute {statute_id: $statute_id})-[:HAS_VERSION]->(sv:StatuteVersion)
        RETURN sv
        ORDER BY sv.effective_from ASC
        """
        
        return self._execute_read(query, {'statute_id': statute_id})
    
    # ============================================
    # CHUNK OPERATIONS
    # ============================================
    
    def create_chunk(self, chunk_data: Dict) -> str:
        """Create a chunk node with temporal metadata."""
        query = """
        MERGE (ch:Chunk {chunk_id: $chunk_id})
        ON CREATE SET
            ch.source_doc_id = $source_doc_id,
            ch.source_doc_type = $source_doc_type,
            ch.chunk_index = $chunk_index,
            ch.text = $text,
            ch.effective_from = date($effective_from),
            ch.effective_to = $effective_to,
            ch.section_number = $section_number
        RETURN ch.chunk_id AS chunk_id
        """
        
        parameters = {
            'chunk_id': chunk_data['chunk_id'],
            'source_doc_id': chunk_data['source_doc_id'],
            'source_doc_type': chunk_data['source_doc_type'],
            'chunk_index': chunk_data.get('chunk_index'),
            'text': chunk_data.get('text'),
            'effective_from': chunk_data.get('effective_from'),
            'effective_to': chunk_data.get('effective_to'),
            'section_number': chunk_data.get('section_number')
        }
        
        result = self._execute_write(query, parameters)
        return result['chunk_id'] if result else None
