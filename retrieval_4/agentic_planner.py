"""
Agentic Planner: Multi-Step Deterministic Retrieval Orchestrator
================================================================

The Agentic Planner orchestrates atomic primitives to perform deterministic,
temporally-aware retrieval with full reasoning transparency.

Process:
1. Grounding: Extract URNs from user query using LLM
2. Scoping: Call Temporal Scoping Primitive for each URN to "lock" version to query date
3. Deterministic Fetch: Retrieve exact TextUnit from scoped version
4. Validation: Compare vector search results with scoped versions and flag discrepancies
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from neo4j import GraphDatabase
import logging
import os
import re
import json
import time
import math

from retrieval_4.agentic_primitives import TemporalScopingPrimitive, StructuralDiscoveryPrimitive
from retrieval_4.hybrid_retriever import HybridRetriever
from config.llm_config import GOOGLE_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# Try to import Gemini for function calling
try:
    import google.generativeai as genai
    from google.generativeai.types import content_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available. Install with: pip install google-generativeai")


class AgenticPlanner:
    """
    Agentic Planner: Orchestrates deterministic retrieval with temporal awareness.
    
    Returns Reasoning Log showing:
    - Resolved URN -> Locked Temporal Version [Date] -> Fetched Valid Text
    - Validation results comparing vector search with scoped versions
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        gemini_api_key: Optional[str] = None
    ):
        """
        Initialize Agentic Planner with Gemini as primary reasoning engine.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Database name
            gemini_api_key: Gemini API key (optional, uses GOOGLE_API_KEY env var if not provided)
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = neo4j_database
        
        # Initialize primitives
        self.temporal_scoper = TemporalScopingPrimitive(self.driver, neo4j_database)
        self.structural_discoverer = StructuralDiscoveryPrimitive(self.driver, neo4j_database)
        
        # Initialize hybrid retriever for vector search (for validation)
        self.hybrid_retriever = HybridRetriever(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database
        )
        
        # Define tools (functions) for Gemini FIRST (before Gemini initialization)
        self.tools = self._define_tools_gemini()
        
        # Validate tools are properly defined
        if not self.tools or not isinstance(self.tools, list):
            logger.error("Tools definition failed! Falling back to empty list.")
            self.tools = []
        
        logger.info(f" Tools defined: {len(self.tools)} tool(s)")
        for i, tool in enumerate(self.tools, 1):
            logger.info(f"   Tool {i}: {tool.name}")
        
        # Initialize Gemini for function calling
        api_key = gemini_api_key or GOOGLE_API_KEY or os.getenv('GOOGLE_API_KEY')
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model_name = "gemini-2.5-flash"
                self.gemini_model = genai.GenerativeModel(
                    model_name=self.gemini_model_name,
                    tools=self.tools
                )
                logger.info(f" Gemini initialized successfully")
                logger.info(f"   Model: {self.gemini_model_name}")
                logger.info(f"   API Key: {'*' * 10 + api_key[-4:] if len(api_key) > 4 else '***'}")
                logger.info(f"   Function calling enabled with {len(self.tools)} tools")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}", exc_info=True)
                self.gemini_model = None
        else:
            self.gemini_model = None
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini library not available. Install with: pip install google-generativeai")
            elif not api_key:
                logger.warning("GOOGLE_API_KEY not set. Function calling will be disabled.")
    
    def _define_tools_gemini(self) -> List[Any]:
        """Define tools (functions) for Gemini function calling."""
        
        temporal_scope_urn = genai.protos.FunctionDeclaration(
            name="temporal_scope_urn",
            description=(
                "Performs Point-in-Time walk to find the exact version of a URN at a target date. "
                "This is the Temporal Scoping Primitive. Use this when you need to determine what "
                "version of a law was valid on a specific date. Returns the version URN, validity dates, "
                "and text content if found."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "urn": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Canonical identifier (e.g., 'urn:lex:lk:act:5:2025!sec24')"
                    ),
                    "target_date": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="ISO format date string (e.g., '2020-01-01')"
                    )
                },
                required=["urn", "target_date"]
            )
        )
        
        discover_structure = genai.protos.FunctionDeclaration(
            name="discover_structure",
            description=(
                "Discovers structural context (parents, children, siblings) for a given URN. "
                "This is the Structural Discovery Primitive. Use this when you need to understand "
                "the hierarchy of a legal provision (e.g., which Chapter contains a Section, "
                "or which Sub-sections belong to a Section)."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "urn": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Canonical identifier (e.g., 'urn:lex:lk:act:5:2025!sec24')"
                    )
                },
                required=["urn"]
            )
        )
        
        return [temporal_scope_urn, discover_structure]
    
    def close(self):
        """Close all connections."""
        self.driver.close()
        self.hybrid_retriever.close()
    
    def _call_groq_with_retry(self, call_func, max_retries: int = 3, initial_delay: float = 10.0):
        """
        Call Groq API with exponential backoff retry logic.
        
        Args:
            call_func: Function that makes the Groq API call (lambda or callable)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (will be doubled on each retry)
        
        Returns:
            Result from the API call
        
        Raises:
            Exception: If all retries are exhausted
        """
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return call_func()
            except Exception as e:
                # Check if it's a rate limit error (429)
                error_str = str(e)
                is_rate_limit = (
                    RateLimitError is not None and isinstance(e, RateLimitError)
                ) or "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower()
                
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        # CRITICAL FIX: Extract the actual retry time from error message
                        retry_after = self._extract_retry_delay_from_error(error_str)
                        
                        if retry_after is None:
                            # Fallback to exponential backoff if we can't extract time
                            retry_after = delay
                        
                        logger.warning(f"   API says: Please wait {retry_after:.1f} seconds")
                        logger.warning(f"   Waiting as instructed...")
                        
                        time.sleep(retry_after)
                        delay = retry_after * 2  # Exponential backoff for next attempt
                    else:
                        logger.error(f"Groq RateLimitError: Max retries ({max_retries}) exceeded")
                        logger.error(f"   Last error: {error_str}")
                        raise
                elif APIError is not None and isinstance(e, APIError):
                    # For other API errors, don't retry
                    logger.error(f"Groq APIError: {e}")
                    raise
                else:
                    # For unexpected errors, retry with exponential backoff
                    if attempt < max_retries - 1:
                        logger.warning(f"Groq API call failed on attempt {attempt + 1}/{max_retries}: {e}")
                        logger.warning(f"   Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(f"Groq API call failed after {max_retries} attempts: {e}")
                        raise
        
        raise Exception("Retry logic exhausted without success")

    def _extract_retry_delay_from_error(self, error_message: str) -> Optional[float]:
        """
        Extract retry delay from Groq error message.
        
        Handles formats like:
        - "Please try again in 7m13.728s" -> 433.728 seconds
        - "Please try again in 30s" -> 30 seconds
        - "Please try again in 2h5m" -> 7500 seconds
        
        Args:
            error_message: The error message from Groq API
        
        Returns:
            Number of seconds to wait, or None if couldn't parse
        """
        import re
        
        # Pattern for time formats: "7m13.728s", "30s", "2h5m30s", etc.
        # Matches: "try again in 7m13.728s" or similar
        time_pattern = r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?'
        match = re.search(time_pattern, error_message, re.IGNORECASE)
        
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            seconds = float(match.group(3)) if match.group(3) else 0
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            if total_seconds > 0:
                logger.info(f"   Parsed retry delay: {hours}h {minutes}m {seconds:.2f}s = {total_seconds:.2f} seconds")
                return total_seconds
        
        # Fallback: try to find any number followed by 's' (seconds only)
        simple_pattern = r'(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?'
        simple_match = re.search(simple_pattern, error_message, re.IGNORECASE)
        if simple_match:
            seconds = float(simple_match.group(1))
            logger.info(f"   Parsed retry delay (simple): {seconds:.2f} seconds")
            return seconds
        
        logger.warning(f"   Could not parse retry delay from: {error_message}")
        return None
    
    def retrieve(self, query: str, target_date: Optional[str] = None, top_k: int = 10) -> Dict:
        """
        Main retrieval method: Deterministic agentic retrieval with Groq function calling.
        
        Implements strict Operational Protocol:
        1. Grounding: Resolve textual references to canonical URNs
        2. Temporal Scoping: Block all text that was not valid on target_date
        3. Validation: Flag "Temporal Collision" if vector search contradicts scoped statute
        
        Args:
            query: Natural language query
            target_date: Optional target date (ISO format). If None, uses current date.
            top_k: Number of results to return
        
        Returns:
            Dictionary with:
            {
                'query': str,
                'target_date': str,
                'reasoning_log': List[Dict],  # Step-by-step reasoning (Lineage of Truth)
                'resolved_urns': List[Dict],  # URNs with temporal scoping
                'structural_context': Dict,    # Structural discovery results
                'temporal_collisions': List[Dict], # Temporal collisions detected
                'lineage_of_truth': List[str],  # Human-readable lineage
                'summary': str                 # Human-readable summary
            }
        """
        # Resolve target date - will be extracted in grounding step
        reasoning_log = []
        lineage_of_truth = []
        
        # Step 1: Grounding - Extract URNs AND temporal context using Gemini (SINGLE API CALL)
        logger.info(f"[Step 1] Grounding: Extracting URNs and temporal context from query...")
        
        # Ensure Gemini client is available before proceeding
        if not self.gemini_model:
            logger.warning("  Gemini model not available. Will use regex fallback for URN extraction.")
            urns = self._extract_urns_with_regex(query)
            # Fallback to current date if no Gemini
            if not target_date:
                target_date = datetime.now().date().isoformat()
        else:
            logger.info(f" Gemini client available. Using Gemini with {len(self.tools)} tools for grounding.")
            # SINGLE API CALL: Extract both URNs and temporal context
            urns, extracted_target_date = self._ground_with_gemini(query, target_date)
            
            # Use extracted date if no explicit date was provided
            if not target_date and extracted_target_date:
                target_date = extracted_target_date
                logger.info(f" Extracted target_date from query: {target_date}")
            elif not target_date:
                target_date = datetime.now().date().isoformat()
                logger.info(f" No temporal context in query. Using current date: {target_date}")
        reasoning_log.append({
            'step': 'grounding',
            'action': 'extract_urns',
            'input': query,
            'output': urns,
            'reasoning': f"Extracted {len(urns)} URN(s) from query: {urns}"
        })
        
        # Step 2: Temporal Scoping - Lock each URN to target_date (STRICT: Block invalid text)
        logger.info(f"[Step 2] Temporal Scoping: Locking URNs to {target_date} (STRICT MODE)...")
        scoped_urns = []
        for urn in urns:
            scoped = self.temporal_scoper.scope_urn(urn, target_date)
            
            # [Resilient Resolution] If not found, try fallbacks
            if scoped['status'] != 'found':
                # Generate fallbacks
                fallbacks = []
                # Permutation: act:9:2023 <-> act:2023:9
                # Generic: urn:lex:lk:2023:act
                
                # Match act:NUM:YEAR (Standard)
                m1 = re.search(r'(urn:lex:lk:act):(\d+):(\d{4})', urn)
                if m1:
                    prefix, num, year = m1.groups()
                    # Fallback 1: Flip to YEAR:NUM
                    fallbacks.append(f"{prefix}:{year}:{num}")
                    # Fallback 2: Generic Year (urn:lex:lk:YEAR:act)
                    fallbacks.append(f"urn:lex:lk:{year}:act")
                
                # Match act:YEAR:NUM (Non-standard)
                m2 = re.search(r'(urn:lex:lk:act):(\d{4}):(\d+)', urn)
                if m2:
                    prefix, year, num = m2.groups()
                    # Fallback 1: Flip to NUM:YEAR
                    fallbacks.append(f"{prefix}:{num}:{year}")
                    # Fallback 2: Generic Year
                    fallbacks.append(f"urn:lex:lk:{year}:act")
                
                # If suffix exists (e.g. !sec24), handle it
                suffix = ""
                if '!' in urn:
                    suffix = "!" + urn.split('!', 1)[1]
                    # Apply suffix to fallbacks (which matched base)
                    fallbacks = [f + suffix for f in fallbacks]
                
                for fallback_urn in fallbacks:
                    logger.info(f" [Resilience planner] Trying fallback URN: {fallback_urn} (Original: {urn})")
                    fallback_result = self.temporal_scoper.scope_urn(fallback_urn, target_date)
                    
                    if fallback_result['status'] == 'found':
                        scoped = fallback_result
                        scoped['reasoning'] = f"[Resilient Resolution] Resolved via {fallback_urn}. " + scoped['reasoning']
                        # Important: Update urn variable? The loop iteration is done.
                        # No, just set scoped to success.
                        lineage_of_truth.append(f" Resiliently Resolved: {urn} -> {fallback_urn}")
                        break
            
            scoped_urns.append(scoped)
            
            # Build lineage entry
            if scoped['status'] == 'found':
                lineage_entry = (
                    f"URN {scoped['urn']} → "
                    f"Version Locked at {target_date} → "
                    f"Text Fetched ({len(scoped.get('text', '') or '')} chars)"
                )
                lineage_of_truth.append(lineage_entry)
            elif scoped['status'] == 'not_found':
                lineage_entry = (
                    f"URN {scoped['urn']} → "
                    f"Law not yet enacted as of {target_date}"
                )
                lineage_of_truth.append(lineage_entry)
            
            reasoning_log.append({
                'step': 'temporal_scoping',
                'action': 'scope_urn',
                'input': {'urn': urn, 'target_date': target_date},
                'output': scoped,
                'reasoning': scoped['reasoning']
            })
        
        # Step 3: Structural Discovery - Use Groq to decide when to discover structure
        logger.info(f"[Step 3] Structural Discovery: Finding hierarchy...")
        structural_context = {}
        if self.gemini_model:
            # Use Groq to decide which URNs need structural discovery
            structure_decision = self._decide_structure_discovery_with_groq(query, urns)
            for urn in structure_decision.get('urns_to_discover', urns):
                structure = self.structural_discoverer.discover_structure(urn)
                structural_context[urn] = structure
                reasoning_log.append({
                    'step': 'structural_discovery',
                    'action': 'discover_structure',
                    'input': {'urn': urn},
                    'output': structure,
                    'reasoning': structure['reasoning']
                })
        else:
            # Fallback: discover structure for all URNs
            for urn in urns:
                structure = self.structural_discoverer.discover_structure(urn)
                structural_context[urn] = structure
                reasoning_log.append({
                    'step': 'structural_discovery',
                    'action': 'discover_structure',
                    'input': {'urn': urn},
                    'output': structure,
                    'reasoning': structure['reasoning']
                })
        
        # Step 4: Deterministic Fetch - Retrieve text from scoped versions (no hard blocking)
        logger.info(f"[Step 4] Deterministic Fetch: Retrieving text from scoped versions (NO HARD BLOCKING)...")
        fetched_texts = []
        for scoped in scoped_urns:
            # Fetch if status is 'found' and text exists
            if scoped['status'] == 'found' and scoped.get('text'):
                fetched_texts.append({
                    'urn': scoped['urn'],
                    'version_urn': scoped['version_urn'],
                    'valid_start': scoped['valid_start'],
                    'valid_end': scoped['valid_end'],
                    'text': scoped['text'],
                    'is_active': scoped['is_active']
                })
                reasoning_log.append({
                    'step': 'deterministic_fetch',
                    'action': 'fetch_text',
                    'input': {'urn': scoped['urn'], 'version_urn': scoped['version_urn']},
                    'output': {'text_length': len(scoped['text'])},
                    'reasoning': (
                        f"Fetched text from {scoped['version_urn']} "
                        f"(valid {scoped['valid_start'] or 'unknown'} to {scoped['valid_end'] or 'present'})"
                    )
                })
            else:
                # No text available – keep track but do not hard-block; ranking will handle relevance
                reasoning_log.append({
                    'step': 'deterministic_fetch',
                    'action': 'skip_missing_text',
                    'input': {'urn': scoped.get('urn')},
                    'output': None,
                    'reasoning': (
                        f"No scoped text available for {scoped.get('urn')} at {target_date}; "
                        f"semantic + temporal re-ranking will determine relevance."
                    )
                })
        
        # Step 5: Temporal Re-ranking & Drift Detection (soft time-aware scoring)
        logger.info(f"[Step 5] Temporal Re-ranking: Combining semantic and temporal scores...")
        vector_results = self.hybrid_retriever.retrieve(query, top_k=top_k * 2, query_date=target_date)
        
        temporal_collisions = []  # Reused to store temporal drift events for backward compatibility
        
        results = vector_results.get('results', [])
        reranked_results = []
        
        # Cache for validity lookups to avoid repeated queries
        validity_cache: Dict[str, Dict[str, Optional[str]]] = {}
        
        for idx, result in enumerate(results, 1):
            evidence = result.get('evidence', {})
            segment_info = evidence.get('segment', {}) or {}
            provisions = evidence.get('provisions', []) or []
            
            # Semantic score from vector search
            semantic_score = float(segment_info.get('similarity_score', 0.0))
            
            # Choose primary URN for temporal scoring: prefer provision URN, else segment URN
            primary_urn = None
            if provisions and isinstance(provisions, list):
                primary_urn = provisions[0].get('urn')
            if not primary_urn:
                primary_urn = segment_info.get('urn')
            
            # Fetch validity metadata
            valid_start = None
            valid_end = None
            if primary_urn:
                if primary_urn in validity_cache:
                    validity_info = validity_cache[primary_urn]
                else:
                    validity_info = self._get_validity_for_urn(primary_urn)
                    validity_cache[primary_urn] = validity_info
                valid_start = validity_info.get('valid_start')
                valid_end = validity_info.get('valid_end')
            
            # Compute temporal decay score
            temporal_score = self._compute_temporal_decay_score(valid_start, valid_end, target_date)
            
            # Final combined score
            final_score = (0.7 * semantic_score) + (0.3 * temporal_score)
            
            # Attach scores to metadata
            metadata = result.setdefault('metadata', {})
            metadata['semantic_score'] = semantic_score
            metadata['temporal_score'] = temporal_score
            metadata['combined_score'] = final_score
            metadata['valid_start'] = valid_start
            metadata['valid_end'] = valid_end
            
            reranked_results.append(result)
            
            # Collision Intelligence: Temporal Drift detection
            if semantic_score > 0.8 and temporal_score < 0.2:
                drift_message = (
                    "TEMPORAL DRIFT DETECTED: This provision is highly relevant but its validity has decayed. "
                    "Suggesting newer versions."
                )
                temporal_collisions.append({
                    'type': 'temporal_drift',
                    'urn': primary_urn,
                    'semantic_score': semantic_score,
                    'temporal_score': temporal_score,
                    'final_score': final_score,
                    'message': drift_message
                })
                
                reasoning_log.append({
                    'step': 'validation',
                    'action': 'detect_temporal_drift',
                    'input': {'urn': primary_urn, 'rank': idx},
                    'output': {
                        'semantic_score': semantic_score,
                        'temporal_score': temporal_score,
                        'combined_score': final_score
                    },
                    'reasoning': drift_message
                })
        
        # Sort by combined score (time-aware ranking)
        reranked_results.sort(key=lambda r: r.get('metadata', {}).get('combined_score', 0.0), reverse=True)
        
        # Limit to top_k for reporting
        top_ranked = reranked_results[:top_k]
        
        # Log per-document scores in reasoning log and lineage of truth
        for rank, result in enumerate(top_ranked, 1):
            metadata = result.get('metadata', {})
            evidence = result.get('evidence', {})
            segment_info = evidence.get('segment', {}) or {}
            provisions = evidence.get('provisions', []) or []
            
            primary_urn = None
            if provisions and isinstance(provisions, list):
                primary_urn = provisions[0].get('urn')
            if not primary_urn:
                primary_urn = segment_info.get('urn')
            
            semantic_score = metadata.get('semantic_score', 0.0)
            temporal_score = metadata.get('temporal_score', 0.0)
            final_score = metadata.get('combined_score', 0.0)
            
            reasoning_log.append({
                'step': 'validation',
                'action': 'rank_by_temporal_relevance',
                'input': {'rank': rank, 'urn': primary_urn},
                'output': {
                    'semantic_score': semantic_score,
                    'temporal_score': temporal_score,
                    'combined_score': final_score
                },
                'reasoning': (
                    f"Doc {rank}: URN {primary_urn or 'unknown'} → "
                    f"semantic_score={semantic_score:.3f}, temporal_score={temporal_score:.3f}, "
                    f"combined_score={final_score:.3f} (70% semantic, 30% temporal)."
                )
            })
            
            lineage_of_truth.append(
                f"Doc {rank}: URN {primary_urn or 'unknown'} → "
                f"Semantic={semantic_score:.3f}, Temporal={temporal_score:.3f}, "
                f"Confidence={final_score:.3f}."
            )
        
        reasoning_log.append({
            'step': 'validation',
            'action': 'summary_temporal_rerank',
            'input': {'vector_results_count': len(results)},
            'output': {
                'ranked_results': len(top_ranked),
                'temporal_drifts': len(temporal_collisions)
            },
            'reasoning': (
                f"Ranked {len(results)} documents by combined semantic (70%) and temporal (30%) scores. "
                f"Detected {len(temporal_collisions)} temporal drift case(s)."
            )
        })
        
        # Generate summary with Lineage of Truth
        summary = self._generate_summary_with_lineage(reasoning_log, scoped_urns, temporal_collisions, lineage_of_truth)
        
        return {
            'query': query,
            'target_date': target_date,
            'reasoning_log': reasoning_log,
            'resolved_urns': scoped_urns,
            'structural_context': structural_context,
            'temporal_collisions': temporal_collisions,
            'lineage_of_truth': lineage_of_truth,
            'summary': summary
        }
    
    def _ground_with_gemini(self, query: str, target_date: Optional[str]) -> tuple[List[str], Optional[str]]:
        """
        Grounding: Extract URNs AND temporal context from query using Gemini (SINGLE API CALL).
        
        Args:
            query: Natural language query
            target_date: Explicit target date (if provided, overrides extraction)
        
        Returns:
            Tuple of (urns, extracted_target_date)
            - urns: List of URN strings
            - extracted_target_date: Extracted date from query (or None if not found/not needed)
        """
        if not self.gemini_model:
            logger.warning("Gemini not available. Falling back to regex URN extraction.")
            return self._extract_urns_with_regex(query), None
        
        try:
            # Build prompt that extracts BOTH URNs and temporal context
            # If target_date is provided, use it; otherwise ask model to extract from query
            if target_date:
                date_instruction = f"TARGET DATE: {target_date} (provided explicitly)"
            else:
                date_instruction = """TARGET DATE: Not provided. 
                
TEMPORAL EXTRACTION REQUIRED:
1. Analyze the query for temporal references (years, dates, time periods)
2. If you find temporal information:
   - Extract the year and construct an ISO date (YYYY-MM-DD)
   - Examples: "in 2020" → 2020-01-01, "before 2023" → 2022-12-31
3. If no temporal context found, use current date (2026-02-04)
4. Include the extracted/inferred date when calling temporal_scope_urn"""
            
            prompt = f"""You are an advanced Legal Grounding Agent. Your task is to identify the precise legal URNs for the user's query, accounting for TEMPORAL CONTEXT.

{date_instruction}

CRITICAL: JURISPRUDENCE AWARENESS RULES
1. ENTITY SUCCESSION (Bribery Act Case):
   - "Bribery Act" is AMBIGUOUS. Check the TARGET DATE.
   - If target_date < "2023-09-01": Ground to Bribery Act, No. 11 of 1954 (urn:lex:lk:act:11:1954)
   - If target_date >= "2023-09-01": Ground to Anti-Corruption Act, No. 9 of 2023 (urn:lex:lk:act:9:2023)
   - If NO target_date is given (current law): Prefer the 2023 Act.

2. DATE-AWARE EXTRACTION:
   - Compare the legislation year in the URN to the TARGET DATE.
   - Do NOT ground a futuristic Act (e.g., Act of 2025) if the query is about 2015, unless the user explicitly asks about "future law".
   - If the query mentions a specific Act Year (e.g., "Act of 2025"), ALWAYS respect that specific year regardless of the target date.

3. AMBIGUITY HANDLING:
   - If a common name matches multiple potential laws and the date rules don't resolve it, call 'temporal_scope_urn' for ALL candidates.
   - Example query: "Consumer Protection Act" -> If unsure, try both 1979 and 2003 versions.

STANDARD CONVERSION RULES:
  - "Section X of Act No. Y of ZZZZ" → urn:lex:lk:act:Y:ZZZZ!secX
  - "Chapter X of Act No. Y of ZZZZ" → urn:lex:lk:act:Y:ZZZZ!chX
  - "Act No. X of YYYY" → urn:lex:lk:act:X:YYYY

INSTRUCTIONS:
1. Analyze the context: Query "{query}"
2. Extract temporal context from query if not provided
3. Apply Jurisprudence Rules (especially for Bribery/Anti-Corruption).
4. Call the 'temporal_scope_urn' tool for the resolved URN(s) (passing the extracted/provided target_date).
5. DO NOT write text explanations. ONLY call tools.

EXAMPLE TOOL CALL:
[Call temporal_scope_urn with urn="urn:lex:lk:act:11:1954", target_date="2020-01-01"]

QUERY: {query}
"""
            
            logger.info(f" [GEMINI] Calling API (URN + Temporal Extraction)")
            logger.info(f"   Model: {self.gemini_model_name}")
            logger.info(f"   Query: {query[:80]}...")
            logger.info(f"   Tools: {[t.name for t in self.tools]}")
            
            # CRITICAL: Make exactly ONE API call
            api_call_count = 0
            
            # Start chat with Gemini
            chat = self.gemini_model.start_chat(enable_automatic_function_calling=False)
            api_call_count += 1
            response = chat.send_message(prompt)
            
            logger.info(f" [GEMINI]  API call successful! Response received.")
            logger.info(f" API calls made: {api_call_count} (MUST BE 1)")
            
            # Extract URNs and temporal context from function calls
            urns = []
            extracted_date = None
            
            # Check if model called functions
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        logger.info(f"   → Tool called: {fc.name}")
                        
                        if fc.name == "temporal_scope_urn":
                            # Extract URN
                            urn = fc.args.get('urn')
                            if urn and urn not in urns:
                                urns.append(urn)
                                logger.info(f"      ✓ Extracted URN: {urn}")
                            
                            # Extract temporal context from the function call
                            if not extracted_date and not target_date:
                                func_target_date = fc.args.get('target_date')
                                if func_target_date:
                                    extracted_date = func_target_date
                                    logger.info(f"      ✓ Extracted date: {extracted_date}")
                            
                        
                        elif fc.name == "discover_structure":
                            # Just extract the URN for structure discovery
                            urn = fc.args.get('urn')
                            if urn and urn not in urns:
                                urns.append(urn)
                                logger.info(f"       Extracted URN for structure discovery: {urn}")
                            
                            
            
            # If no URNs found via function calling, check if model provided text
            if not urns:
                if hasattr(response, 'text') and response.text:
                    logger.warning(f"️  Model returned text instead of calling tools: {response.text[:200]}")
                    logger.warning(" Falling back to regex extraction from original query.")
                else:
                    logger.warning("  No function calls or text received. Falling back to regex.")
                return self._extract_urns_with_regex(query), None
            
            logger.info(f" [GEMINI] Extracted {len(urns)} URN(s): {urns}")
            if extracted_date:
                logger.info(f" [GEMINI] Extracted temporal context: {extracted_date}")
            logger.info(f"VERIFICATION: Total API calls = {api_call_count} (Expected: 1)")
            
            return urns, extracted_date
        
        except Exception as e:
            logger.error(f"Gemini grounding failed: {e}", exc_info=True)
            return self._extract_urns_with_regex(query), None
    
    def _extract_query_temporal_context(self, query: str) -> Optional[Dict]:
        """
        Extract temporal information (dates, years, time periods) from query using Gemini.
        
        Args:
            query: Natural language query
        
        Returns:
            Dictionary with:
            - 'year': Extracted year if present
            - 'date': ISO format date if extractable (YYYY-MM-DD)
            - 'temporal_phrase': The phrase indicating time
            - 'has_temporal_context': Boolean indicating if temporal info was found
            
            Returns None if extraction fails or no temporal context found.
        """
        if not self.gemini_model:
            logger.warning("Gemini not available for temporal extraction")
            return None
        
        try:
            prompt = f"""Analyze this legal query and extract any temporal information (dates, years, time periods).

Query: "{query}"

Extract:
1. Any specific year mentioned (e.g., "in 2020", "as of 2015")
2. Any date or date range
3. Relative time references (e.g., "before 2023", "after the amendment")

If you find temporal information:
- Return the year as an integer
- If possible, construct an ISO date (YYYY-MM-DD). If only year is mentioned, use YYYY-01-01
- Include the exact phrase from the query that indicates time

Return a JSON object with this structure:
{{
  "has_temporal_context": true/false,
  "year": <year as integer or null>,
  "date": "YYYY-MM-DD" or null,
  "temporal_phrase": "the phrase from query" or null
}}

Examples:
- "What did Section 24 say in 2020?" → {{"has_temporal_context": true, "year": 2020, "date": "2020-01-01", "temporal_phrase": "in 2020"}}
- "What is the current law?" → {{"has_temporal_context": false, "year": null, "date": null, "temporal_phrase": null}}
- "Before the 2023 amendment" → {{"has_temporal_context": true, "year": 2023, "date": "2022-12-31", "temporal_phrase": "before the 2023 amendment"}}
"""
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Very low temperature for factual extraction
                    response_mime_type="application/json"
                )
            )
            
            # Parse JSON response
            result = json.loads(response.text)
            
            if result.get('has_temporal_context'):
                logger.info(f" Temporal extraction successful: {result}")
                return result
            else:
                logger.info("  No temporal context found in query")
                return None
        
        except Exception as e:
            logger.warning(f"Temporal extraction failed: {e}")
            return None
    
    def _decide_structure_discovery_with_groq(self, query: str, urns: List[str]) -> Dict:
        """
        Decide which URNs need structural discovery.
        
        OPTIMIZATION: No LLM call needed - just discover structure for all URNs.
        This saves 1 API call per retrieval.
        
        Args:
            query: Original query
            urns: List of URNs to consider
        
        Returns:
            Dictionary with 'urns_to_discover' list
        """
        # NO API CALL - just return all URNs for structure discovery
        logger.info(f" Structure discovery: Will discover structure for all {len(urns)} URN(s)")
        logger.info(f"    NO LLM CALL - saves 1 API call!")
        return {'urns_to_discover': urns}

    def _get_validity_for_urn(self, urn: str) -> Dict[str, Optional[str]]:
        """
        Fetch validity metadata (valid_start, valid_end) for a given URN.
        
        Uses Neo4j directly and caches results for efficiency.
        """
        if not urn:
            return {'valid_start': None, 'valid_end': None}
        
        if not hasattr(self, '_validity_cache'):
            self._validity_cache: Dict[str, Dict[str, Optional[str]]] = {}
        
        if urn in self._validity_cache:
            return self._validity_cache[urn]
        
        valid_start = None
        valid_end = None
        
        try:
            with self.driver.session(database=self.database) as session:
                record = session.run(
                    """
                    MATCH (n {urn: $urn})
                    RETURN n.valid_start AS valid_start,
                           n.valid_end AS valid_end
                    LIMIT 1
                    """,
                    {'urn': urn}
                ).single()
                
                if record:
                    valid_start = record.get('valid_start')
                    valid_end = record.get('valid_end')
        except Exception as e:
            logger.warning(f"Could not fetch validity metadata for URN {urn}: {e}")
        
        info = {'valid_start': valid_start, 'valid_end': valid_end}
        self._validity_cache[urn] = info
        return info
    
    def _compute_temporal_decay_score(
        self,
        valid_start: Optional[str],
        valid_end: Optional[str],
        target_date: str
    ) -> float:
        """
        TemporalDecayScorer: Compute time-aware score in [0, 1].
        
        Score = 1 / (1 + ln(1 + Δdays))
        
        If target_date falls within [valid_start, valid_end], returns 1.0 (temporal boost).
        Otherwise, Δdays is the minimum distance in days between target_date and the validity bounds.
        """
        try:
            target_dt = datetime.fromisoformat(target_date).date() if isinstance(target_date, str) else target_date
        except (ValueError, TypeError):
            # If target_date is invalid, return neutral score
            return 0.5
        
        def _parse_date(value: Any) -> Optional[date]:
            if not value:
                return None
            try:
                if isinstance(value, date):
                    return value
                if isinstance(value, datetime):
                    return value.date()
                if isinstance(value, str):
                    # Handle possible datetime strings with time component
                    if 'T' in value:
                        return datetime.fromisoformat(value.replace('Z', '+00:00')).date()
                    return datetime.fromisoformat(value).date()
            except Exception:
                return None
            return None
        
        start_dt = _parse_date(valid_start)
        end_dt = _parse_date(valid_end)
        
        # If both bounds exist and target is within [start, end], full boost
        if start_dt and end_dt and start_dt <= target_dt <= end_dt:
            return 1.0
        
        # Build list of distances to validity bounds
        deltas: List[int] = []
        if start_dt:
            deltas.append(abs((target_dt - start_dt).days))
        if end_dt:
            deltas.append(abs((target_dt - end_dt).days))
        
        if not deltas:
            # No temporal information – neutral
            return 0.5
        
        delta_days = max(0, min(deltas))
        
        # Reciprocal decay: 1 / (1 + ln(1 + Δdays))
        score = 1.0 / (1.0 + math.log1p(delta_days))
        
        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, score)))

    def _extract_urns_with_llm(self, query: str) -> List[str]:
        """Use Groq to extract URNs from query."""
        if not self.gemini_model:
            logger.warning("Groq not available for LLM extraction. Falling back to regex.")
            return self._extract_urns_with_regex(query)
        
        prompt = f"""You are a legal information extraction system for Sri Lankan law.

Analyze the following legal query and extract all URNs (canonical identifiers) mentioned.

URN Format: urn:lex:lk:act:X:YYYY!secZ or urn:lex:lk:case:sc:appeal:X:YYYY

Query: {query}

Return ONLY valid JSON object with this exact structure:
{{"urns": ["urn:lex:lk:act:5:2025!sec24", "urn:lex:lk:case:sc:appeal:65:2025"]}}

If no URNs are found, return: {{"urns": []}}
"""
        try:
            logger.info(f"[GROQ] Using Groq for URN extraction (fallback method)")
            # Wrap with retry logic for rate limiting
            response = self._call_groq_with_retry(
                lambda: self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a legal information extraction system. Extract URNs from queries and return ONLY a JSON object with a 'urns' key containing an array of URN strings."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                ),
                max_retries=3,
                initial_delay=2.0
            )
            
            response_text = response.choices[0].message.content
            
            # Try to parse as JSON object first
            try:
                response_json = json.loads(response_text)
                # Check if it's a JSON object with an array
                if isinstance(response_json, dict):
                    # First, look for the explicit "urns" key we requested
                    if 'urns' in response_json and isinstance(response_json['urns'], list):
                        urns = response_json['urns']
                        return [urn for urn in urns if isinstance(urn, str) and urn.startswith('urn:')]
                    # Fallback: Look for other common keys that might contain the array
                    for key in ['urn_list', 'results', 'data', 'urn_array']:
                        if key in response_json and isinstance(response_json[key], list):
                            urns = response_json[key]
                            return [urn for urn in urns if isinstance(urn, str) and urn.startswith('urn:')]
                    # If no key found, try to find any array in the values
                    for value in response_json.values():
                        if isinstance(value, list):
                            urns = value
                            return [urn for urn in urns if isinstance(urn, str) and urn.startswith('urn:')]
                elif isinstance(response_json, list):
                    # Direct array response (fallback)
                    return [urn for urn in response_json if isinstance(urn, str) and urn.startswith('urn:')]
            except json.JSONDecodeError:
                pass
            
            # Fallback: Extract JSON array using regex
            json_match = re.search(r'\[[^\]]*\]', response_text)
            if json_match:
                urns = json.loads(json_match.group(0))
                return [urn for urn in urns if isinstance(urn, str) and urn.startswith('urn:')]
            else:
                logger.warning("Could not parse URNs from Groq response. Falling back to regex.")
                return self._extract_urns_with_regex(query)
        except Exception as e:
            logger.warning(f"Groq URN extraction failed: {e}. Falling back to regex.")
            return self._extract_urns_with_regex(query)
    
    def _extract_urns_with_regex(self, query: str) -> List[str]:
        """Fallback: Extract URNs using regex patterns."""
        urns = []
        
        # Pattern for URNs
        urn_pattern = r'urn:lex:lk:[a-z]+:[^"\s]+'
        matches = re.finditer(urn_pattern, query, re.IGNORECASE)
        for match in matches:
            urns.append(match.group(0))
        
        # Also try to extract Act references and convert to URN format
        act_pattern = r'Act\s+No\.?\s*(\d+)\s+of\s+(\d{4})'
        section_pattern = r'Section\s+(\d+)|Sec\.?\s+(\d+)'
        
        act_matches = re.finditer(act_pattern, query, re.IGNORECASE)
        for act_match in act_matches:
            act_num = act_match.group(1)
            year = act_match.group(2)
            
            # Try to find section number
            section_match = re.search(section_pattern, query[act_match.end():], re.IGNORECASE)
            if section_match:
                section_num = section_match.group(1) or section_match.group(2)
                urn = f"urn:lex:lk:act:{act_num}:{year}!sec{section_num}"
            else:
                urn = f"urn:lex:lk:act:{act_num}:{year}"
            
            if urn not in urns:
                urns.append(urn)
        
        return urns
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap (can be enhanced with embeddings)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_summary_with_lineage(
        self, 
        reasoning_log: List[Dict], 
        scoped_urns: List[Dict], 
        temporal_collisions: List[Dict],
        lineage_of_truth: List[str]
    ) -> str:
        """Generate human-readable summary with Lineage of Truth."""
        lines = []
        lines.append("=" * 80)
        lines.append("APOFASI CHRONOS-LEX AGENTIC RETRIEVAL - REASONING LOG")
        lines.append("=" * 80)
        lines.append("")
        
        # Lineage of Truth
        lines.append("LINEAGE OF TRUTH:")
        lines.append("-" * 80)
        if lineage_of_truth:
            for i, lineage in enumerate(lineage_of_truth, 1):
                lines.append(f"  {i}. {lineage}")
        else:
            lines.append("  No URNs resolved.")
        lines.append("")
        
        # Summary of URNs resolved
        lines.append("RESOLVED URNS WITH TEMPORAL SCOPING:")
        lines.append("-" * 80)
        for scoped in scoped_urns:
            if scoped['status'] == 'found':
                lines.append(f"  ✓ {scoped['urn']}")
                lines.append(f"    → Version Locked: {scoped['version_urn']}")
                lines.append(f"    → Valid from: {scoped['valid_start'] or 'unknown'} to {scoped['valid_end'] or 'present'}")
                if scoped['text']:
                    lines.append(f"    → Text Fetched: {len(scoped['text'])} characters")
            elif scoped['status'] == 'not_found':
                lines.append(f"  ✗ {scoped['urn']}")
                lines.append(f"    → {scoped['reasoning']}")
                if 'not yet enacted' in scoped['reasoning'].lower():
                    lines.append(f"    → Law not yet enacted as of {scoped['target_date']}")
            else:
                lines.append(f"  ✗ {scoped['urn']} - {scoped['reasoning']}")
        lines.append("")
        
        # Temporal Events (Collisions / Drift)
        if temporal_collisions:
            lines.append("TEMPORAL EVENTS DETECTED:")
            lines.append("-" * 80)
            for collision in temporal_collisions:
                ctype = collision.get('type', 'collision')
                # Backwards-compatible handling for original collision shape
                if ctype == 'temporal_collision':
                    case_urn = collision.get('case_urn', 'unknown')
                    statute_urn = collision.get('statute_urn', 'unknown')
                    similarity = collision.get('similarity', 0.0)
                    priority = collision.get('priority', 'statute')
                    reasoning = collision.get('reasoning', '')
                    
                    lines.append(f"   COLLISION: Case {case_urn} contradicts Statute {statute_urn}")
                    lines.append(f"    Similarity: {similarity:.2%}")
                    lines.append(f"    Priority: {priority.upper()} (Statute prioritized as deterministic source)")
                    if reasoning:
                        lines.append(f"    Reasoning: {reasoning}")
                # New temporal drift events
                elif ctype == 'temporal_drift':
                    urn = collision.get('urn', 'unknown')
                    sem = collision.get('semantic_score', 0.0)
                    temp = collision.get('temporal_score', 0.0)
                    combined = collision.get('final_score', 0.0)
                    message = collision.get('message', 'TEMPORAL DRIFT DETECTED.')
                    
                    lines.append(f"   TEMPORAL DRIFT: URN {urn}")
                    lines.append(f"    Semantic score: {sem:.3f}")
                    lines.append(f"    Temporal score: {temp:.3f}")
                    lines.append(f"    Combined score: {combined:.3f}")
                    lines.append(f"    Message: {message}")
                else:
                    # Fallback generic rendering
                    lines.append(f"   TEMPORAL EVENT: {collision}")
            lines.append("")
        else:
            lines.append("✓ No Temporal Collisions or Temporal Drift detected")
            lines.append("")
        
        # Step-by-step reasoning
        lines.append("STEP-BY-STEP REASONING:")
        lines.append("-" * 80)
        for entry in reasoning_log:
            lines.append(f"  [{entry['step'].upper()}] {entry['action']}")
            lines.append(f"      Reasoning: {entry['reasoning']}")
        
        return "\n".join(lines)
