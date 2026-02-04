"""
GraphIngestor: Ingests JSON from StatutoryExtraction into Neo4j
================================================================

Transforms flat JSON objects (from StatutoryExtraction) into a hierarchical
Property Graph in Neo4j using the Provision-based schema.

Design:
- Node Labels: :Provision + dynamic labels (:Section, :Chapter, :Part, :Act, :Schedule)
- Relationships: [:PARENT_OF] for hierarchy, [:CITES] for citations
- Ghost Nodes: Created for citations that don't exist yet
- Batch Processing: 1000 nodes per transaction for performance
"""

from typing import Dict, List, Optional
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)


class GraphIngestor:
    """
    Ingests statute JSON data from StatutoryExtraction into Neo4j.
    
    Implements:
    - Phase A: Idempotent node ingestion (MERGE on URN)
    - Phase B: Hierarchy reconstruction (PARENT_OF relationships)
    - Phase C: Citation linking with ghost nodes
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize GraphIngestor with Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.batch_size = 1000  # Process 1000 nodes per batch
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def ingest_statutes(self, json_data: Dict) -> Dict[str, int]:
        """
        Main entry point: Ingest statute provisions into Neo4j.
        
        Args:
            json_data: Dictionary with 'metadata' and 'provisions' keys
                - metadata: Document metadata (doc_type, doc_id, act_number, etc.)
                - provisions: List of provision dictionaries from StatutoryExtraction
        
        Returns:
            Dictionary with ingestion statistics:
            {
                'nodes_created': int,
                'nodes_updated': int,
                'hierarchy_links': int,
                'citation_links': int,
                'ghost_nodes_created': int
            }
        """
        if 'provisions' not in json_data:
            raise ValueError("json_data must contain 'provisions' key")
        
        provisions = json_data['provisions']
        metadata = json_data.get('metadata', {})
        
        if not provisions:
            logger.warning("No provisions to ingest")
            return {
                'nodes_created': 0,
                'nodes_updated': 0,
                'hierarchy_links': 0,
                'citation_links': 0,
                'ghost_nodes_created': 0
            }
        
        logger.info(f"Ingesting {len(provisions)} provisions into Neo4j")
        
        stats = {
            'nodes_created': 0,
            'nodes_updated': 0,
            'hierarchy_links': 0,
            'citation_links': 0,
            'ghost_nodes_created': 0
        }
        
        # Phase A: Node Ingestion (idempotent)
        # First, create the Act node if it doesn't exist
        act_created = self._create_act_node(metadata)
        if act_created:
            stats['nodes_created'] += 1
        
        # Then ingest all provisions
        node_stats = self._ingest_nodes(provisions, metadata)
        stats['nodes_created'] += node_stats['created']
        stats['nodes_updated'] += node_stats['updated']
        
        # Phase B: Hierarchy Reconstruction
        stats['hierarchy_links'] = self._reconstruct_hierarchy(provisions, metadata)
        
        # Phase C: Citation Linking (with ghost nodes)
        citation_stats = self._link_citations(provisions)
        stats['citation_links'] = citation_stats['links']
        stats['ghost_nodes_created'] = citation_stats['ghost_nodes']
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def _ingest_nodes(self, provisions: List[Dict], metadata: Dict) -> Dict[str, int]:
        """
        Phase A: Idempotent node ingestion.
        
        Uses MERGE on urn (URN) to ensure no duplicates.
        Sets properties using ON CREATE SET and ON MATCH SET.
        
        Args:
            provisions: List of provision dictionaries
            metadata: Document metadata
        
        Returns:
            Dictionary with 'created' and 'updated' counts
        """
        created = 0
        updated = 0
        
        # Process in batches
        for i in range(0, len(provisions), self.batch_size):
            batch = provisions[i:i + self.batch_size]
            
            with self.driver.session(database=self.database) as session:
                for prov in batch:
                    result = self._merge_provision_node(session, prov, metadata)
                    if result == 'created':
                        created += 1
                    elif result == 'updated':
                        updated += 1
        
        logger.info(f"Phase A: Created {created} nodes, Updated {updated} nodes")
        return {'created': created, 'updated': updated}
    
    def _merge_provision_node(self, session, provision: Dict, metadata: Dict) -> str:
        """
        MERGE a single provision node.
        
        Args:
            session: Neo4j session
            provision: Provision dictionary
            metadata: Document metadata
        
        Returns:
            'created' or 'updated'
        """
        urn = provision['urn']
        subtype = provision.get('subtype', 'Section')
        content = provision.get('content', {})
        properties = provision.get('properties', {})
        
        # Build dynamic labels: :Provision + :Section/:Chapter/:Part/:Act/:Schedule
        labels = ['Provision', subtype]
        
        # Extract text and label
        text = content.get('text', '')
        label = content.get('label', urn)
        valid_from = properties.get('valid_from') or metadata.get('valid_from')
        
        # Build label string for Cypher
        label_str = ':'.join(labels)
        
        query = f"""
        MERGE (p:{label_str} {{urn: $urn}})
        ON CREATE SET
            p.text = $text,
            p.label = $label,
            p.valid_from = $valid_from,
            p.embedding = null,
            p.created_at = datetime()
        ON MATCH SET
            p.text = $text,
            p.label = $label,
            p.valid_from = $valid_from,
            p.updated_at = datetime()
        RETURN p.urn AS urn, 
               CASE WHEN p.created_at IS NOT NULL THEN 'created' ELSE 'updated' END AS action
        """
        
        parameters = {
            'urn': urn,
            'text': text,
            'label': label,
            'valid_from': valid_from
        }
        
        result = session.run(query, parameters)
        record = result.single()
        
        if record:
            return record['action']
        return 'updated'  # Default if unclear
    
    def _create_act_node(self, metadata: Dict) -> bool:
        """
        Create the root Act node if it doesn't exist.
        
        Args:
            metadata: Document metadata
        
        Returns:
            True if node was created, False if it already existed
        """
        doc_type = metadata.get('doc_type')
        doc_id = metadata.get('doc_id')
        
        if not doc_type or not doc_id:
            return None
        
        # Construct Act URN
        if doc_type == 'act':
            act_urn = f"urn:lex:lk:act:{doc_id}"
        elif doc_type == 'cap':
            act_urn = f"urn:lex:lk:cap:{doc_id}"
        else:
            return None
        
        with self.driver.session(database=self.database) as session:
            query = """
            MERGE (a:Provision:Act {urn: $urn})
            ON CREATE SET
                a.text = $text,
                a.label = $label,
                a.valid_from = $valid_from,
                a.embedding = null,
                a.created_at = datetime()
            ON MATCH SET
                a.updated_at = datetime()
            RETURN a.urn AS urn
            """
            
            act_number = metadata.get('act_number', '')
            act_year = metadata.get('act_year', '')
            if doc_type == 'act':
                label = f"Act No. {act_number} of {act_year}" if act_number and act_year else act_urn
            else:
                label = f"Chapter {doc_id}"
            
            result = session.run(query, {
                'urn': act_urn,
                'text': None,  # Act node doesn't have text content
                'label': label,
                'valid_from': metadata.get('valid_from')
            })
            
            record = result.single()
            if record:
                # Check if it was created (has created_at) or updated (has updated_at)
                # We can't easily determine this from the query, so we'll check separately
                check_query = """
                MATCH (a:Provision:Act {urn: $urn})
                RETURN a.created_at IS NOT NULL AS was_created
                """
                check_result = session.run(check_query, {'urn': act_urn})
                check_record = check_result.single()
                if check_record:
                    return check_record['was_created']
            return False
    
    def _reconstruct_hierarchy(self, provisions: List[Dict], metadata: Dict) -> int:
        """
        Phase B: Reconstruct hierarchy using PARENT_OF relationships.
        
        Uses the hierarchy field from provisions to create parent-child links.
        Hierarchy structure: Act -> Part -> Chapter -> Section
        
        Args:
            provisions: List of provision dictionaries
        
        Returns:
            Number of hierarchy links created
        """
        links_created = 0
        
        # Process in batches
        for i in range(0, len(provisions), self.batch_size):
            batch = provisions[i:i + self.batch_size]
            
            with self.driver.session(database=self.database) as session:
                for prov in batch:
                    hierarchy = prov.get('hierarchy', {})
                    child_urn = prov['urn']
                    
                    # Determine parent URN based on hierarchy
                    # Priority: part_id > chapter_id > act_id
                    parent_urn = None
                    parent_type = None
                    
                    if hierarchy.get('part_id'):
                        # Extract Part URN from part_id (e.g., "PART I" -> construct URN)
                        parent_urn = self._construct_part_urn(prov, hierarchy['part_id'])
                        parent_type = 'Part'
                    elif hierarchy.get('chapter_id'):
                        # Extract Chapter URN
                        parent_urn = self._construct_chapter_urn(prov, hierarchy['chapter_id'])
                        parent_type = 'Chapter'
                    elif hierarchy.get('act_id'):
                        parent_urn = hierarchy['act_id']
                        parent_type = 'Act'
                    
                    # Also check if this provision itself is an Act/Part/Chapter
                    # If so, we need to link it to its parent Act
                    if prov.get('subtype') in ['Part', 'Chapter']:
                        # Extract Act URN from the provision's URN
                        act_urn = self._extract_act_urn_from_provision(prov)
                        if not act_urn:
                            # Fallback: construct from metadata
                            doc_type = metadata.get('doc_type')
                            doc_id = metadata.get('doc_id')
                            if doc_type == 'act':
                                act_urn = f"urn:lex:lk:act:{doc_id}"
                            elif doc_type == 'cap':
                                act_urn = f"urn:lex:lk:cap:{doc_id}"
                        
                        if act_urn and act_urn != child_urn:
                            # Ensure Act exists
                            self._ensure_parent_exists(session, act_urn, 'Act', prov)
                            if self._create_parent_relationship(session, act_urn, child_urn, 'Act', prov.get('subtype')):
                                links_created += 1
                    
                    # Create parent relationship if parent exists
                    if parent_urn and parent_urn != child_urn:
                        # First, ensure parent node exists (MERGE it as a ghost if needed)
                        self._ensure_parent_exists(session, parent_urn, parent_type, prov)
                        
                        # Create relationship
                        if self._create_parent_relationship(session, parent_urn, child_urn, parent_type, prov.get('subtype')):
                            links_created += 1
        
        logger.info(f"Phase B: Created {links_created} hierarchy links")
        return links_created
    
    def _construct_part_urn(self, provision: Dict, part_id: str) -> Optional[str]:
        """
        Construct URN for a Part from part_id (e.g., "PART I").
        
        Args:
            provision: Provision dictionary (to extract Act URN)
            part_id: Part identifier (e.g., "PART I")
        
        Returns:
            URN string or None
        """
        # Extract base Act URN from provision
        act_urn = self._extract_act_urn_from_provision(provision)
        if not act_urn:
            return None
        
        # Extract part number from part_id (e.g., "PART I" -> "I")
        import re
        part_match = re.search(r'PART\s+([IVX]+|[A-Z0-9]+)', part_id, re.IGNORECASE)
        if part_match:
            part_num = part_match.group(1)
            # Construct URN: urn:lex:lk:act:2023:9!partI
            return f"{act_urn}!part{part_num}"
        return None
    
    def _construct_chapter_urn(self, provision: Dict, chapter_id: str) -> Optional[str]:
        """
        Construct URN for a Chapter from chapter_id (e.g., "CHAPTER I").
        
        Args:
            provision: Provision dictionary (to extract Act URN)
            chapter_id: Chapter identifier (e.g., "CHAPTER I")
        
        Returns:
            URN string or None
        """
        # Extract base Act URN from provision
        act_urn = self._extract_act_urn_from_provision(provision)
        if not act_urn:
            return None
        
        # Extract chapter number from chapter_id (e.g., "CHAPTER I" -> "I")
        import re
        chapter_match = re.search(r'CHAPTER\s+([IVX]+|[A-Z0-9]+)', chapter_id, re.IGNORECASE)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            # Construct URN: urn:lex:lk:act:2023:9!chapterI
            return f"{act_urn}!chapter{chapter_num}"
        return None
    
    def _extract_act_urn_from_provision(self, provision: Dict) -> Optional[str]:
        """
        Extract the base Act URN from a provision's URN.
        
        Example: "urn:lex:lk:act:2023:9!sec12" -> "urn:lex:lk:act:2023:9"
        
        Args:
            provision: Provision dictionary
        
        Returns:
            Act URN string or None
        """
        urn = provision.get('urn', '')
        # Extract everything before the '!' character
        if '!' in urn:
            return urn.split('!')[0]
        return urn
    
    def _ensure_parent_exists(self, session, parent_urn: str, parent_type: str, child_provision: Dict):
        """
        Ensure parent node exists. If not, create it as a ghost node.
        
        Args:
            session: Neo4j session
            parent_urn: URN of parent node
            parent_type: Type of parent ('Act', 'Part', 'Chapter')
            child_provision: Child provision (for context)
        """
        # Check if parent exists
        check_query = """
        MATCH (p:Provision {urn: $urn})
        RETURN p.urn AS urn
        LIMIT 1
        """
        
        result = session.run(check_query, {'urn': parent_urn})
        if result.single():
            return  # Parent exists
        
        # Create ghost parent node
        labels = ['Provision', parent_type, 'GhostNode']
        label_str = ':'.join(labels)
        
        create_query = f"""
        MERGE (p:{label_str} {{urn: $urn}})
        ON CREATE SET
            p.text = null,
            p.label = $label,
            p.embedding = null,
            p.created_at = datetime()
        """
        
        label = f"{parent_type} {parent_urn.split('!')[-1] if '!' in parent_urn else parent_urn}"
        session.run(create_query, {'urn': parent_urn, 'label': label})
    
    def _create_parent_relationship(self, session, parent_urn: str, child_urn: str, 
                                   parent_type: str, child_type: str) -> bool:
        """
        Create PARENT_OF relationship between parent and child.
        
        Args:
            session: Neo4j session
            parent_urn: URN of parent node
            child_urn: URN of child node
            parent_type: Type of parent
            child_type: Type of child
        
        Returns:
            True if relationship was created, False otherwise
        """
        query = """
        MATCH (parent:Provision {urn: $parent_urn})
        MATCH (child:Provision {urn: $child_urn})
        MERGE (parent)-[r:PARENT_OF]->(child)
        ON CREATE SET r.created_at = datetime()
        RETURN r
        """
        
        result = session.run(query, {
            'parent_urn': parent_urn,
            'child_urn': child_urn
        })
        
        return result.single() is not None
    
    def _link_citations(self, provisions: List[Dict]) -> Dict[str, int]:
        """
        Phase C: Link citations with ghost node strategy.
        
        For each provision's edges.refers_to, create CITES relationships.
        If target URN doesn't exist, create it as a GhostNode.
        
        Args:
            provisions: List of provision dictionaries
        
        Returns:
            Dictionary with 'links' and 'ghost_nodes' counts
        """
        links_created = 0
        ghost_nodes_created = 0
        
        # Collect all citations first
        citations = []
        for prov in provisions:
            source_urn = prov['urn']
            edges = prov.get('edges', {})
            refers_to = edges.get('refers_to', [])
            
            for target_urn in refers_to:
                citations.append((source_urn, target_urn))
        
        # Process citations in batches
        for i in range(0, len(citations), self.batch_size):
            batch = citations[i:i + self.batch_size]
            
            with self.driver.session(database=self.database) as session:
                for source_urn, target_urn in batch:
                    # Ensure target exists (create ghost if needed)
                    if self._ensure_citation_target_exists(session, target_urn):
                        ghost_nodes_created += 1
                    
                    # Create CITES relationship
                    if self._create_citation_relationship(session, source_urn, target_urn):
                        links_created += 1
        
        logger.info(f"Phase C: Created {links_created} citation links, {ghost_nodes_created} ghost nodes")
        return {'links': links_created, 'ghost_nodes': ghost_nodes_created}
    
    def _ensure_citation_target_exists(self, session, target_urn: str) -> bool:
        """
        Ensure citation target exists. Create as ghost node if it doesn't.
        
        Args:
            session: Neo4j session
            target_urn: URN of target node
        
        Returns:
            True if ghost node was created, False if node already existed
        """
        # Check if target exists
        check_query = """
        MATCH (p:Provision {urn: $urn})
        RETURN p.urn AS urn
        LIMIT 1
        """
        
        result = session.run(check_query, {'urn': target_urn})
        if result.single():
            return False  # Node exists, not a ghost
        
        # Determine type from URN
        # e.g., "urn:lex:lk:act:2023:9!sec12" -> "Section"
        # e.g., "urn:lex:lk:cap:26" -> "Act"
        target_type = self._infer_type_from_urn(target_urn)
        
        # Create ghost node
        labels = ['Provision', target_type, 'GhostNode']
        label_str = ':'.join(labels)
        
        create_query = f"""
        MERGE (p:{label_str} {{urn: $urn}})
        ON CREATE SET
            p.text = null,
            p.label = $label,
            p.embedding = null,
            p.created_at = datetime()
        """
        
        # Generate label from URN
        label = self._generate_label_from_urn(target_urn)
        
        session.run(create_query, {'urn': target_urn, 'label': label})
        return True  # Ghost node created
    
    def _infer_type_from_urn(self, urn: str) -> str:
        """
        Infer provision type from URN.
        
        Args:
            urn: URN string
        
        Returns:
            Type string ('Act', 'Section', 'Part', 'Chapter', 'Schedule')
        """
        if '!sec' in urn:
            return 'Section'
        elif '!part' in urn:
            return 'Part'
        elif '!chapter' in urn:
            return 'Chapter'
        elif '!sched' in urn:
            return 'Schedule'
        else:
            # No component ID means it's the Act itself
            return 'Act'
    
    def _generate_label_from_urn(self, urn: str) -> str:
        """
        Generate human-readable label from URN.
        
        Args:
            urn: URN string
        
        Returns:
            Label string
        """
        # Extract component from URN
        if '!sec' in urn:
            section_num = urn.split('!sec')[1]
            return f"Section {section_num}"
        elif '!part' in urn:
            part_num = urn.split('!part')[1]
            return f"Part {part_num}"
        elif '!chapter' in urn:
            chapter_num = urn.split('!chapter')[1]
            return f"Chapter {chapter_num}"
        elif '!sched' in urn:
            sched_num = urn.split('!sched')[1]
            return f"Schedule {sched_num}"
        else:
            # Act URN
            if 'act:' in urn:
                parts = urn.split('act:')[1].split(':')
                if len(parts) >= 2:
                    return f"Act No. {parts[1]} of {parts[0]}"
            elif 'cap:' in urn:
                cap_num = urn.split('cap:')[1].split('!')[0]
                return f"Chapter {cap_num}"
            return urn
    
    def _create_citation_relationship(self, session, source_urn: str, target_urn: str) -> bool:
        """
        Create CITES relationship from source to target.
        
        Args:
            session: Neo4j session
            source_urn: URN of source node
            target_urn: URN of target node
        
        Returns:
            True if relationship was created, False otherwise
        """
        query = """
        MATCH (source:Provision {urn: $source_urn})
        MATCH (target:Provision {urn: $target_urn})
        MERGE (source)-[r:CITES]->(target)
        ON CREATE SET r.created_at = datetime()
        RETURN r
        """
        
        result = session.run(query, {
            'source_urn': source_urn,
            'target_urn': target_urn
        })
        
        return result.single() is not None
    
    def heal_ghost_nodes(self, json_data: Dict) -> int:
        """
        Self-healing: Update ghost nodes with real data when statutes are ingested.
        
        When a statute that was previously cited is now ingested, this method
        updates the ghost node with real text and removes the GhostNode label.
        
        Args:
            json_data: Dictionary with 'provisions' from StatutoryExtraction
        
        Returns:
            Number of ghost nodes healed
        """
        provisions = json_data.get('provisions', [])
        healed = 0
        
        with self.driver.session(database=self.database) as session:
            for prov in provisions:
                urn = prov['urn']
                
                # Check if this URN exists as a ghost node
                check_query = """
                MATCH (p:Provision:GhostNode {urn: $urn})
                RETURN p.urn AS urn
                LIMIT 1
                """
                
                result = session.run(check_query, {'urn': urn})
                if result.single():
                    # Heal the ghost node
                    content = prov.get('content', {})
                    properties = prov.get('properties', {})
                    
                    subtype = prov.get('subtype', 'Section')
                    text = content.get('text', '')
                    label = content.get('label', urn)
                    valid_from = properties.get('valid_from')
                    
                    heal_query = f"""
                    MATCH (p:Provision:GhostNode {{urn: $urn}})
                    SET p.text = $text,
                        p.label = $label,
                        p.valid_from = $valid_from,
                        p.updated_at = datetime()
                    REMOVE p:GhostNode
                    """
                    
                    session.run(heal_query, {
                        'urn': urn,
                        'text': text,
                        'label': label,
                        'valid_from': valid_from
                    })
                    
                    healed += 1
        
        if healed > 0:
            logger.info(f"Healed {healed} ghost nodes")
        
        return healed


if __name__ == "__main__":
    # Example usage
    from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
    from extraction_2.statutory_extraction import extract_statute_structure
    
    # Sample JSON data (would come from StatutoryExtraction)
    sample_json = {
        "metadata": {
            "doc_type": "act",
            "doc_id": "2023:9",
            "act_number": "9",
            "act_year": "2023",
            "valid_from": "2023-08-08"
        },
        "provisions": [
            {
                "urn": "urn:lex:lk:act:2023:9!sec1",
                "type": "Provision",
                "subtype": "Section",
                "content": {
                    "text": "1. (1) This Act may be cited...",
                    "label": "Section 1"
                },
                "hierarchy": {},
                "properties": {
                    "is_definition_node": False,
                    "valid_from": "2023-08-08"
                },
                "edges": {
                    "refers_to": ["urn:lex:lk:cap:26"]
                }
            }
        ]
    }
    
    # Initialize ingestor
    ingestor = GraphIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    
    try:
        # Ingest provisions
        stats = ingestor.ingest_statutes(sample_json)
        print(f"Ingestion complete: {stats}")
    finally:
        ingestor.close()
