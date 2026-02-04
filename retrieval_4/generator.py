"""
Legal Answer Generator using Gemini
====================================

Generates natural language legal answers based on AgenticPlanner retrieval results.
Uses Gemini API with grounding to verified legal sources (URNs).
"""

from typing import Dict, List, Optional
import logging
import os
import json

logger = logging.getLogger(__name__)

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google.generativeai not available. Install with: pip install google-generativeai")


class LegalGenerator:
    """
    Generates legal answers grounded in verified lineage from AgenticPlanner.
    
    Features:
    - Cites URNs in square brackets [urn:lex:lk...]
    - Explains temporal drift when documents are not primary authority
    - Handles zero-result cases gracefully
    - Uses Gemini for generation with strict grounding
    - EXACTLY 1 API call per generation
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize Legal Generator with Gemini.
        
        Args:
            gemini_api_key: Gemini API key (optional, uses GOOGLE_API_KEY env var if not provided)
            model: Gemini model name (default: gemini-2.5-flash)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini library not available. Install with: pip install google-generativeai")
        
        api_key = gemini_api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Please provide API key.")
        
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model_name=model)
        
        logger.info(f"LegalGenerator initialized with Gemini model: {model}")
    
    def generate(self, retrieval_result: Dict, query: str) -> Dict:
        """
        Generate legal answer from AgenticPlanner retrieval result.
        
        Args:
            retrieval_result: Output from AgenticPlanner.retrieve()
            query: Original user query
        
        Returns:
            Dictionary with:
            - 'answer': Generated legal answer with URN citations
            - 'sources': List of URNs cited
            - 'temporal_warnings': List of temporal drift explanations
            - 'metadata': Generation metadata
        """
        # Extract lineage of truth
        lineage = retrieval_result.get('lineage_of_truth', [])
        temporal_collisions = retrieval_result.get('temporal_collisions', [])
        target_date = retrieval_result.get('target_date')
        
        # Check if we have valid results
        if not lineage or len(lineage) == 0:
            return self._generate_no_results_response(query, target_date)
        
        # Build context from lineage
        context = self._build_context(lineage, temporal_collisions)
        
        # Generate answer using Gemini (SINGLE API CALL)
        answer = self._generate_with_gemini(query, context, target_date)
        
        # Extract cited URNs
        cited_urns = self._extract_cited_urns(answer)
        
        # Build temporal warnings
        temporal_warnings = self._build_temporal_warnings(temporal_collisions)
        
        return {
            'answer': answer,
            'sources': cited_urns,
            'temporal_warnings': temporal_warnings,
            'metadata': {
                'model': self.model_name,
                'lineage_count': len(lineage),
                'temporal_drift_count': len(temporal_collisions),
                'target_date': target_date
            }
        }
    
    def _generate_no_results_response(self, query: str, target_date: Optional[str]) -> Dict:
        """Generate response when no valid provisions were found."""
        date_str = f" for the requested date ({target_date})" if target_date else ""
        answer = f"No valid legal provisions were in force{date_str}."
        
        return {
            'answer': answer,
            'sources': [],
            'temporal_warnings': [],
            'metadata': {
                'model': self.model_name,
                'lineage_count': 0,
                'temporal_drift_count': 0,
                'target_date': target_date
            }
        }
    
    def _build_context(self, lineage: List, temporal_collisions: List[Dict]) -> str:
        """
        Build context string from lineage and temporal collisions.
        
        Format:
        PRIMARY AUTHORITY:
        [urn:...] Text content
        
        TEMPORAL DRIFT (Not Primary Authority):
        [urn:...] Text content (Reason: ...)
        """
        context_parts = []
        
        # Primary authority (lineage of truth)
        # Lineage items are strings in format: "URN: text content"
        if lineage:
            context_parts.append("PRIMARY AUTHORITY (Verified Lineage):")
            for item in lineage:
                # Handle both string and dict formats
                if isinstance(item, str):
                    # String format: just use as-is
                    context_parts.append(f"\n{item}")
                elif isinstance(item, dict):
                    # Dict format: extract urn, text, label
                    urn = item.get('urn', 'unknown')
                    text = item.get('text', 'No text available')[:1000]  # Truncate long text
                    label = item.get('label', '')
                    
                    context_parts.append(f"\n[{urn}] {label}")
                    context_parts.append(f"Text: {text}")
        
        # Temporal drift (collisions)
        if temporal_collisions:
            context_parts.append("\n\nTEMPORAL DRIFT (Not Primary Authority):")
            for collision in temporal_collisions:
                # Handle both old and new formats
                if 'case_urn' in collision:
                    # Old format
                    urn = collision.get('case_urn', 'unknown')
                    reason = collision.get('reason', 'Unknown reason')
                else:
                    # New format (temporal_drift)
                    urn = collision.get('urn', 'unknown')
                    reason = collision.get('drift_reason', collision.get('message', 'Unknown reason'))
                
                context_parts.append(f"\n[{urn}] - Reason: {reason}")
        
        return "\n".join(context_parts)
    
    def _extract_temporal_context(self, query: str) -> Optional[Dict]:
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
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Very low temperature for factual extraction
                    response_mime_type="application/json"
                )
            )
            
            # Parse JSON response
            import json
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
    
    def _generate_with_gemini(self, query: str, context: str, target_date: Optional[str]) -> str:
        """
        Generate answer using Gemini with strict grounding instructions.
        
        System prompt enforces:
        - Apofasi Legal Analyst persona
        - URN citation in square brackets
        - Temporal drift explanations
        - Grounding to provided context only
        - Enhanced explanations and structure
        
        CRITICAL: Makes EXACTLY 1 API call.
        """
        system_instruction = """You are the Apofasi Legal Analyst. Your goal is to provide comprehensive, well-explained legal answers based ONLY on the provided verified lineage.

CRITICAL RULES FOR CITATION:
1. CITATION MASKING: NEVER display raw URN strings (e.g., "urn:lex:lk:act...") in the natural language text.
   - BAD: "According to urn:lex:lk:act:19:1994 section 24..."
   - GOOD: "According to Section 24 of the CIABOC Act, No. 19 of 1994..."

2. URN REFERENCES: Use URNs only as hidden identifiers or in structured citation brackets at the end of paragraphs.
   - Format: [Source: Human Readable Title | urn:...]
   - Example: [Source: Bribery Act | urn:lex:lk:act:11:1954]

CRITICAL RULES FOR TEMPORAL REASONING:
1. TEMPORAL DRIFT: If a document is marked as "Temporal Drift", you MUST explicitly explain why it is excluded.
   - Format: "Note: This document (Human Name) was excluded from the primary ruling because it post-dates the query year (Target Year)."
   - Example: "Note: The Anti-Corruption Act of 2023 was excluded from the primary ruling because it post-dates the query year (2020)."

2. TEMPORAL CONTEXT EXPLANATION: Always explain the temporal context of your answer:
   - If answering about a specific date/year, explain what was in force at that time
   - If laws have changed since the query date, mention this
   - Explain WHY the temporal context matters for this specific question

GENERAL RULES:
1. GROUNDING: Base your answer ONLY on the provided context. Do not invent or assume information.
2. ACCURACY: If the context doesn't contain enough information to answer, say so clearly.
3. DETAILED EXPLANATIONS: Provide thorough explanations, not just brief statements. Explain the legal reasoning, implications, and context.
4. STRUCTURE: Use clear sections with headings for better readability.

ANSWER STRUCTURE (use markdown headings):

## Direct Answer
[Provide a clear, concise answer to the specific question asked - 2-3 sentences]

## Legal Analysis
[Provide detailed explanation of the relevant legal provisions - 4-6 sentences minimum]
- Explain WHAT the law says
- Explain WHY it matters
- Explain HOW it applies to the question
- Include relevant context and implications

## Temporal Context
[If temporal information is relevant, explain the time-based considerations - 2-4 sentences]
- What was in force at the query date
- Any changes or amendments since then
- Why the temporal aspect matters

## Citations
[List all sources used with proper formatting]
- [Source: Human Readable Title | urn:...]

FORMAT REQUIREMENTS:
- Use markdown headings (##) for sections
- Use bullet points for clarity where appropriate
- Write in complete, well-formed sentences
- Aim for substantive explanations (minimum 200 words for complex queries)
- Use natural language for all legal references
"""

        date_context = f"\nTarget Date: {target_date}" if target_date else ""
        
        user_prompt = f"""Query: {query}{date_context}

VERIFIED LEGAL CONTEXT:
{context}

Please provide a comprehensive legal answer based on the above verified context. 
Follow the structured format with sections: Direct Answer, Legal Analysis, Temporal Context (if relevant), and Citations.
Provide detailed explanations - don't just state facts, explain their significance and reasoning.
Remember: NO RAW URNs in the natural language text."""

        try:
            logger.info(f" [GEMINI] Generating answer for query: {query[:80]}...")
            
            # CRITICAL: Make exactly ONE API call
            api_call_count = 0
            
            # Create model with system instruction
            model_with_system = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction
            )
            
            api_call_count += 1
            response = model_with_system.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more factual responses
                    max_output_tokens=3072  # Increased for more detailed answers
                )
            )
            
            answer = response.text
            logger.info(f"[GEMINI] Answer generated ({len(answer)} chars)")
            logger.info(f"API calls made: {api_call_count} (MUST BE 1)")
            logger.info(f"VERIFICATION: Total API calls = {api_call_count} (Expected: 1)")
            
            return answer
        
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"
    
    def _extract_cited_urns(self, answer: str) -> List[Dict[str, str]]:
        """
        Extract citations from the answer.
        
        Looks for patterns like [Source: Title | urn:...]
        Returns list of dicts with 'display_title' and 'raw_urn'.
        """
        import re
        
        # Pattern: [Source: Title | urn:...]
        citation_pattern = r'\[Source:\s*([^|]+?)\s*\|\s*(urn:[^\]]+)\]'
        matches = re.finditer(citation_pattern, answer)
        
        citations = []
        seen_urns = set()
        
        for match in matches:
            title = match.group(1).strip()
            urn = match.group(2).strip()
            
            if urn not in seen_urns:
                citations.append({
                    'display_title': title,
                    'raw_urn': urn
                })
                seen_urns.add(urn)
        
        # Fallback: look for just URNs if new format isn't used perfectly
        if not citations:
            urn_pattern = r'\[(urn:lex:[^\]]+)\]'
            simple_matches = re.findall(urn_pattern, answer)
            for urn in simple_matches:
                if urn not in seen_urns:
                    citations.append({
                        'display_title': 'Legal Document', # Generic fallback title
                        'raw_urn': urn
                    })
                    seen_urns.add(urn)
                    
        return citations
    
    def _build_temporal_warnings(self, temporal_collisions: List[Dict]) -> List[str]:
        """
        Build human-readable temporal warnings from collisions.
        
        Returns:
            List of warning strings
        """
        warnings = []
        
        for collision in temporal_collisions:
            # Handle both old and new formats
            if 'case_urn' in collision:
                # Old format
                urn = collision.get('case_urn', 'unknown')
                reason = collision.get('reason', 'Unknown reason')
            else:
                # New format (temporal_drift)
                urn = collision.get('urn', 'unknown')
                reason = collision.get('drift_reason', 'Unknown reason')
            
            # Format: "URN: Reason" -> "Document X: Reason" (if we could resolve title)
            # For now keep simple but clean
            warnings.append(f"Temporal Exclusion ({urn}): {reason}")
        
        return warnings
