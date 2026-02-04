"""
Agentic Primitives: Atomic Operations for Deterministic Temporal Graph Interaction
===================================================================================

This module implements atomic primitives that can be composed by the Agentic Planner
to perform deterministic, temporally-aware retrieval from the Neo4j graph.

Primitives:
1. Temporal Scoping Primitive: Point-in-time version resolution
2. Structural Discovery Primitive: Hierarchy traversal
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from neo4j import GraphDatabase
import logging
import re

logger = logging.getLogger(__name__)


class TemporalScopingPrimitive:
    """
    Temporal Scoping Primitive: Performs Point-in-Time walk to find exact version.
    
    Logic:
    - Given a URN and target_date, finds the specific TemporalVersion node where:
      valid_start <= target_date < valid_end
    - If no valid_end exists, assumes version is currently active (infinity)
    - Returns only the specific version valid on that day (zero temporal hallucination)
    """
    
    def __init__(self, driver, database: str = "neo4j"):
        """
        Initialize Temporal Scoping Primitive.
        
        Args:
            driver: Neo4j driver instance
            database: Database name
        """
        self.driver = driver
        self.database = database
    
    def scope_urn(self, urn: str, target_date: str) -> Dict:
        """
        Perform Point-in-Time walk to find the exact version of a URN at target_date.
        
        Args:
            urn: Canonical identifier (e.g., "urn:lex:lk:act:5:2025!sec24")
            target_date: ISO format date string (e.g., "2020-01-01")
        
        Returns:
            Dictionary with validity info.
        """
        # Simply call the core validity checker logic
        with self.driver.session(database=self.database) as session:
            return self._get_validity_for_urn(session, urn, target_date)

    def _get_validity_for_urn(self, session, urn: str, target_date: str) -> Dict:
        """
        Core validity check logic moved from scope_urn.
        """
        try:
            # Parse target date
            if isinstance(target_date, str):
                target_dt = datetime.fromisoformat(target_date).date()
            else:
                target_dt = target_date
            
            # Strategy 1: Check if there's an explicit TemporalVersion node
            version_query = """
            MATCH (p:Provision {urn: $urn})
            OPTIONAL MATCH (p)-[:HAS_VERSION]->(v:TemporalVersion)
            WHERE (v.valid_start IS NULL OR date(v.valid_start) <= date($target_date))
              AND (v.valid_end IS NULL OR date(v.valid_end) > date($target_date))
            RETURN v.urn AS version_urn,
                   v.valid_start AS valid_start,
                   v.valid_end AS valid_end,
                   v.text AS text,
                   p.text AS provision_text,
                   p.urn AS provision_urn,
                   labels(p) AS labels
            ORDER BY v.valid_start DESC
            LIMIT 1
            """
            
            result = session.run(version_query, {
                'urn': urn,
                'target_date': target_date
            })
            
            record = result.single()
            
            if record and record.get('version_urn'):
                # Found explicit temporal version
                valid_start_str = record['valid_start'].isoformat() if hasattr(record['valid_start'], 'isoformat') else str(record['valid_start'])
                valid_end_str = record['valid_end'].isoformat() if record['valid_end'] and hasattr(record['valid_end'], 'isoformat') else (str(record['valid_end']) if record['valid_end'] else None)
                
                return {
                    'urn': urn,
                    'target_date': target_date,
                    'version_urn': record['version_urn'],
                    'valid_start': valid_start_str,
                    'valid_end': valid_end_str,
                    'is_active': record['valid_end'] is None,
                    'text': record['text'] or record['provision_text'],
                    'status': 'found',
                    'reasoning': (
                        f"Found TemporalVersion node {record['version_urn']} "
                        f"valid from {valid_start_str} to "
                        f"{valid_end_str if valid_end_str else 'present'}"
                    )
                }
            
            # Strategy 2: If no TemporalVersion, check if Provision has temporal properties
            # Use date() function for proper Neo4j date comparison
            provision_query = """
            MATCH (p:Provision {urn: $urn})
            RETURN p.urn AS urn,
                   p.text AS text,
                   p.valid_from AS valid_from,
                   p.valid_end AS valid_end,
                   p.enacted_date AS enacted_date,
                   labels(p) AS labels
            LIMIT 1
            """
            
            result = session.run(provision_query, {'urn': urn})
            record = result.single()
            
            if record:
                # Use Neo4j date comparison query for accuracy
                temporal_check_query = """
                MATCH (p:Provision {urn: $urn})
                WHERE p.valid_from IS NOT NULL OR p.enacted_date IS NOT NULL OR p.valid_end IS NOT NULL
                RETURN 
                    p.urn AS urn,
                    p.text AS text,
                    p.valid_from AS valid_from,
                    p.valid_end AS valid_end,
                    p.enacted_date AS enacted_date,
                    CASE 
                        WHEN p.valid_from IS NOT NULL AND date(p.valid_from) > date($target_date) THEN 'not_yet_valid'
                        WHEN p.valid_end IS NOT NULL AND date(p.valid_end) <= date($target_date) THEN 'expired'
                        WHEN p.enacted_date IS NOT NULL AND date(p.enacted_date) > date($target_date) THEN 'not_yet_enacted'
                        ELSE 'valid'
                    END AS temporal_status
                LIMIT 1
                """
                
                temporal_result = session.run(temporal_check_query, {
                    'urn': urn,
                    'target_date': target_date
                })
                temporal_record = temporal_result.single()
                
                if temporal_record:
                    temporal_status = temporal_record['temporal_status']
                    valid_from = temporal_record.get('valid_from')
                    valid_to = temporal_record.get('valid_end')
                    enacted_date = temporal_record.get('enacted_date')
                    
                    # Convert dates to strings for JSON serialization
                    valid_from_str = None
                    if valid_from:
                        valid_from_str = valid_from.isoformat() if hasattr(valid_from, 'isoformat') else str(valid_from)
                    
                    valid_to_str = None
                    if valid_to:
                        valid_to_str = valid_to.isoformat() if hasattr(valid_to, 'isoformat') else str(valid_to)
                    
                    enacted_date_str = None
                    if enacted_date:
                        enacted_date_str = enacted_date.isoformat() if hasattr(enacted_date, 'isoformat') else str(enacted_date)
                    
                    if temporal_status == 'not_yet_valid':
                        return {
                            'urn': urn,
                            'target_date': target_date,
                            'version_urn': None,
                            'valid_start': valid_from_str,
                            'valid_end': None,
                            'is_active': False,
                            'text': None,
                            'status': 'not_found',
                            'reasoning': (
                                f"Law not yet enacted as of {target_date}. "
                                f"Provision {urn} becomes valid on {valid_from_str or enacted_date_str}"
                            )
                        }
                    
                    if temporal_status == 'expired':
                        return {
                            'urn': urn,
                            'target_date': target_date,
                            'version_urn': None,
                            'valid_start': valid_from_str,
                            'valid_end': valid_to_str,
                            'is_active': False,
                            'text': None,
                            'status': 'not_found',
                            'reasoning': (
                                f"Provision {urn} is no longer valid at {target_date}. "
                                f"It was valid until {valid_to_str}"
                            )
                        }
                    
                    if temporal_status == 'not_yet_enacted':
                        return {
                            'urn': urn,
                            'target_date': target_date,
                            'version_urn': None,
                            'valid_start': enacted_date_str,
                            'valid_end': None,
                            'is_active': False,
                            'text': None,
                            'status': 'not_found',
                            'reasoning': (
                                f"Law not yet enacted as of {target_date}. "
                                f"Act was enacted on {enacted_date_str}"
                            )
                        }
                
                # Provision is valid at target_date
                valid_from = record.get('valid_from')
                valid_to = record.get('valid_end')
                valid_from_str = valid_from.isoformat() if valid_from and hasattr(valid_from, 'isoformat') else (str(valid_from) if valid_from else None)
                valid_to_str = valid_to.isoformat() if valid_to and hasattr(valid_to, 'isoformat') else (str(valid_to) if valid_to else None)
                
                return {
                    'urn': urn,
                    'target_date': target_date,
                    'version_urn': urn,  # Use provision URN as version URN
                    'valid_start': valid_from_str,
                    'valid_end': valid_to_str,
                    'is_active': valid_to is None,
                    'text': record['text'],
                    'status': 'found',
                    'reasoning': (
                        f"Using current version of {urn} "
                        f"(valid from {valid_from_str or 'unknown'} to {valid_to_str or 'present'})"
                    )
                }
            
            # Strategy 3: Check Act-level temporal properties
            # Extract act URN from provision URN (e.g., "urn:lex:lk:act:5:2025!sec24" -> "urn:lex:lk:act:5:2025")
            if '!' in urn:
                act_urn = urn.split('!')[0]
                act_query = """
                MATCH (a:Act {urn: $act_urn})
                RETURN a.enacted_date AS enacted_date,
                       a.valid_from AS valid_from,
                       a.valid_end AS valid_end
                LIMIT 1
                """
                
                result = session.run(act_query, {'act_urn': act_urn})
                try:
                    act_record = result.single()
                except Exception:
                    # Handle cases where result is consumed or error occurs
                    act_record = None
                
                if act_record and act_record.get('enacted_date'):
                    enacted_date = act_record['enacted_date']
                    enacted_date_str = enacted_date.isoformat() if hasattr(enacted_date, 'isoformat') else str(enacted_date)
                    enacted_date_dt = datetime.fromisoformat(enacted_date_str).date() if isinstance(enacted_date_str, str) else enacted_date
                    
                    if enacted_date_dt > target_dt:
                        return {
                            'urn': urn,
                            'target_date': target_date,
                            'version_urn': None,
                            'valid_start': enacted_date_str,
                            'valid_end': None,
                            'is_active': False,
                            'text': None,
                            'status': 'not_found',
                            'reasoning': (
                                f"Law not yet enacted as of {target_date}. "
                                f"Act was enacted on {enacted_date_str}"
                            )
                        }
            
            # Strategy 4: No temporal info found
            return {
                'urn': urn,
                'target_date': target_date,
                'version_urn': None,
                'valid_start': None,
                'valid_end': None,
                'is_active': False,
                'text': None,
                'status': 'not_found',
                'reasoning': (
                    f"Provision {urn} not found in database or has no temporal information. "
                    f"Cannot determine validity at {target_date}"
                )
            }
        
        except Exception as e:
            logger.error(f"Error in temporal scoping for {urn} at {target_date}: {e}", exc_info=True)
            return {
                'urn': urn,
                'target_date': target_date,
                'version_urn': None,
                'valid_start': None,
                'valid_end': None,
                'is_active': False,
                'text': None,
                'status': 'error',
                'reasoning': f"Error during temporal scoping: {str(e)}"
            }


class StructuralDiscoveryPrimitive:
    """
    Structural Discovery Primitive: Traverses graph hierarchy to find parent/child nodes.
    
    Logic:
    - Given a URN, finds all immediate parent and child nodes
    - Uses PARENT_OF relationships for hierarchy
    - Returns structural context (e.g., if Section URN provided, finds Chapter and Sub-sections)
    """
    
    def __init__(self, driver, database: str = "neo4j"):
        """
        Initialize Structural Discovery Primitive.
        
        Args:
            driver: Neo4j driver instance
            database: Database name
        """
        self.driver = driver
        self.database = database
    
    def discover_structure(self, urn: str) -> Dict:
        """
        Discover structural context (parents and children) for a given URN.
        
        Args:
            urn: Canonical identifier (e.g., "urn:lex:lk:act:5:2025!sec24")
        
        Returns:
            Dictionary with:
            {
                'urn': str,
                'parents': List[Dict],  # Immediate parent nodes
                'children': List[Dict], # Immediate child nodes
                'siblings': List[Dict], # Sibling nodes (same parent)
                'status': str,          # 'found', 'not_found', 'error'
                'reasoning': str        # Explanation
            }
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Find parents (nodes that have this node as child)
                parent_query = """
                MATCH (parent:Provision)-[:PARENT_OF]->(child:Provision {urn: $urn})
                RETURN parent.urn AS urn,
                       parent.label AS label,
                       parent.text AS text,
                       labels(parent) AS labels
                ORDER BY parent.urn
                """
                
                parent_result = session.run(parent_query, {'urn': urn})
                parents = []
                for record in parent_result:
                    parents.append({
                        'urn': record['urn'],
                        'label': record.get('label'),
                        'text': record.get('text', '')[:200] if record.get('text') else '',  # Truncate
                        'labels': record.get('labels', [])
                    })
                
                # Find children (nodes that have this node as parent)
                child_query = """
                MATCH (parent:Provision {urn: $urn})-[:PARENT_OF]->(child:Provision)
                RETURN child.urn AS urn,
                       child.label AS label,
                       child.text AS text,
                       labels(child) AS labels
                ORDER BY child.urn
                """
                
                child_result = session.run(child_query, {'urn': urn})
                children = []
                for record in child_result:
                    children.append({
                        'urn': record['urn'],
                        'label': record.get('label'),
                        'text': record.get('text', '')[:200] if record.get('text') else '',  # Truncate
                        'labels': record.get('labels', [])
                    })
                
                # Find siblings (nodes with same parent)
                sibling_query = """
                MATCH (parent:Provision)-[:PARENT_OF]->(sibling:Provision)
                WHERE EXISTS {
                    MATCH (parent)-[:PARENT_OF]->(target:Provision {urn: $urn})
                }
                AND sibling.urn <> $urn
                RETURN sibling.urn AS urn,
                       sibling.label AS label,
                       sibling.text AS text,
                       labels(sibling) AS labels
                ORDER BY sibling.urn
                """
                
                sibling_result = session.run(sibling_query, {'urn': urn})
                siblings = []
                for record in sibling_result:
                    siblings.append({
                        'urn': record['urn'],
                        'label': record.get('label'),
                        'text': record.get('text', '')[:200] if record.get('text') else '',  # Truncate
                        'labels': record.get('labels', [])
                    })
                
                # Check if node exists
                exists_query = """
                MATCH (p:Provision {urn: $urn})
                RETURN p.urn AS urn, p.label AS label, labels(p) AS labels
                LIMIT 1
                """
                
                exists_result = session.run(exists_query, {'urn': urn})
                exists_record = exists_result.single()
                
                if not exists_record:
                    return {
                        'urn': urn,
                        'parents': [],
                        'children': [],
                        'siblings': [],
                        'status': 'not_found',
                        'reasoning': f"Provision {urn} not found in graph"
                    }
                
                return {
                    'urn': urn,
                    'parents': parents,
                    'children': children,
                    'siblings': siblings,
                    'status': 'found',
                    'reasoning': (
                        f"Found {len(parents)} parent(s), {len(children)} child(ren), "
                        f"and {len(siblings)} sibling(s) for {urn}"
                    )
                }
        
        except Exception as e:
            logger.error(f"Error in structural discovery for {urn}: {e}", exc_info=True)
            return {
                'urn': urn,
                'parents': [],
                'children': [],
                'siblings': [],
                'status': 'error',
                'reasoning': f"Error during structural discovery: {str(e)}"
            }
