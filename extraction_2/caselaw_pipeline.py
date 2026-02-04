"""
CaseLawPipeline: Neuro-Symbolic Case Law Processing
====================================================

4-Stage Pipeline:
A. Deterministic Header Extraction (Regex)
B. Semantic Segmentation
C. Citation Resolution (Neuro-Symbolic Bridge)
D. Vectorization (Nomic Embeddings)

"""

import re
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

try:
    import google.genai as genai  # pyright: ignore[reportMissingImports]
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google.genai not available. Install with: pip install google-genai")

try:
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence_transformers not available. Install with: pip install sentence-transformers")

try:
    from pypdf import PdfReader  # pyright: ignore[reportMissingImports]
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logging.warning("pypdf not available. Install with: pip install pypdf")

logger = logging.getLogger(__name__)


class CaseLawPipeline:
    """
    Pipeline for processing Sri Lankan Supreme Court Judgments.
    
    Transforms raw PDF text into semantically segmented and vectorized JSON.
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None, nomic_model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize the CaseLawPipeline.
        
        Args:
            gemini_api_key: Google Gemini API key (or set GOOGLE_API_KEY env var)
            nomic_model_name: Name of the Nomic embedding model
        """
        # Initialize Gemini (new google.genai API)
        if GEMINI_AVAILABLE:
            import os
            api_key = gemini_api_key or os.getenv('GOOGLE_API_KEY')
            if api_key:
                try:
                    self.gemini_client = genai.Client(api_key=api_key)
                    # Use gemini-2.0-flash as per migration guide examples
                    self.gemini_model_name = 'gemini-2.5-flash-lite'
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {e}")
                    self.gemini_client = None
                    self.gemini_model_name = None
            else:
                logger.warning("Gemini API key not provided. Stage B (Semantic Segmentation) will fail.")
                self.gemini_client = None
                self.gemini_model_name = None
        else:
            self.gemini_client = None
            self.gemini_model_name = None
            logger.warning("Gemini not available. Stage B will be skipped.")
        
        # Initialize Nomic embedding model (load once for performance)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading Nomic embedding model: {nomic_model_name}")
                self.embedding_model = SentenceTransformer(nomic_model_name, trust_remote_code=True)
                logger.info("Nomic model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Nomic model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logger.warning("SentenceTransformers not available. Stage D (Vectorization) will be skipped.")
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF or TXT file through all 4 stages.
        
        Args:
            pdf_path: Path to the PDF or TXT file (supports both formats)
        
        Returns:
            Dictionary with the complete case law structure
        """
        # Check if it's a .txt file
        if pdf_path.lower().endswith('.txt'):
            logger.info(f"Processing TXT file: {pdf_path}")
            try:
                with open(pdf_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                raise ValueError(f"Failed to read TXT file: {pdf_path}. Error: {e}")
        else:
            logger.info(f"Processing PDF: {pdf_path}")
            # Extract text from PDF
            if not PYPDF_AVAILABLE:
                raise ImportError("pypdf is required. Install with: pip install pypdf")
            
            text = self._extract_text_from_pdf(pdf_path)
        
        # Ensure text is a string (not None)
        if text is None:
            raise ValueError(f"PDF extraction returned None: {pdf_path}")
        
        # Ensure text is actually a string type
        if not isinstance(text, str):
            logger.warning(f"PDF extraction returned non-string type: {type(text)}. Converting to string...")
            text = str(text) if text is not None else ""
        
        if not text or len(text.strip()) < 100:
            raise ValueError(f"PDF appears to be empty or unreadable: {pdf_path}")
        
        # Stage A: Deterministic Header Extraction
        logger.info("Stage A: Extracting headers...")
        try:
            header_data = self._extract_headers(text)
        except Exception as e:
            logger.error(f"Error in _extract_headers: {e}")
            logger.error(f"Text type: {type(text)}, Text length: {len(text) if text else 0}")
            logger.error(f"First 200 chars: {repr(text[:200]) if text else 'None'}")
            raise
        
        # Stage B: Semantic Segmentation
        logger.info("Stage B: Semantic segmentation with Gemini...")
        segmented_content = self._semantic_segmentation(text)
        
        # Stage C: Citation Resolution
        logger.info("Stage C: Resolving citations...")
        citations = self._resolve_citations(segmented_content)
        
        # Stage D: Vectorization
        logger.info("Stage D: Generating embeddings...")
        vectors = self._vectorize_content(segmented_content)
        
        # Construct URN
        urn = self._construct_urn(header_data)
        
            # Build final output (matching exact schema)
        result = {
            "urn": urn,
            "meta": {
                "case_number": header_data.get("case_number"),
                "date": header_data.get("date"),
                "decision_date": header_data.get("decision_date") or header_data.get("date"),  # For Neo4j temporal queries
                "court": header_data.get("court", "Supreme Court"),
                "parties": header_data.get("parties", {})
            },
            "content": {
                "facts": segmented_content.get("facts", ""),
                "ratio": segmented_content.get("ratio_decidendi", ""),  # Map ratio_decidendi to "ratio" for schema
                "ratio_decidendi": segmented_content.get("ratio_decidendi", ""),  # Also keep original key for clarity
                "legal_issues": segmented_content.get("legal_issues", ""),
                "arguments": segmented_content.get("arguments", ""),
                "obiter_dicta": segmented_content.get("obiter_dicta", ""),
                "outcome": segmented_content.get("outcome", "")
            },
            "vectors": vectors,
            "edges": {
                "statutes": citations
            }
        }
        
        logger.info(f"Processing complete. URN: {urn}")
        return result
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pypdf.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text as string
        
        Raises:
            ValueError: If PDF is empty or unreadable
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    # Handle None return value from extract_text()
                    if page_text is not None and isinstance(page_text, str):
                        text += page_text + "\n"
                    else:
                        logger.warning(f"Page {page_num} returned None or non-string text, skipping...")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
                    continue
            
            # Ensure we have valid text
            if not text or len(text.strip()) < 10:
                raise ValueError(f"PDF extraction failed: extracted text is empty or too short")
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise ValueError(f"Failed to extract text from PDF: {pdf_path}. Error: {e}")
    
    def _extract_headers(self, text: str) -> Dict:
        """
        Stage A: Deterministic Header Extraction using "Top & Tail" strategy.
        
        Top (first 3000 chars): Case Number, Parties, Court
        Tail (last 3000 chars): Decision Date (Sri Lankan footer format)
        
        Args:
            text: Full text of the judgment
        
        Returns:
            Dictionary with extracted header information
        """
        # Safety check: ensure text is a string
        if not isinstance(text, str):
            logger.error(f"_extract_headers received non-string type: {type(text)}")
            return {}
        
        if not text:
            logger.warning("_extract_headers received empty text")
            return {}
        
        header_data = {}
        text_len = len(text)
        
        # TOP: First 3000 characters for case number, parties, court
        top_section = text[:3000] if text_len > 3000 else text
        # Ensure top_section is a string
        if not isinstance(top_section, str):
            logger.error(f"top_section is not a string: {type(top_section)}")
            top_section = str(top_section) if top_section is not None else ""
        
        # Extract case number from top
        # Pattern 1: SC/APPEAL/65/2025
        case_pattern_1 = r'SC/APPEAL/(\d+)/(\d{4})'
        try:
            match = re.search(case_pattern_1, top_section, re.IGNORECASE)
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error in case_pattern_1: {e}, top_section type: {type(top_section)}")
            match = None
        if match:
            header_data["case_number"] = f"SC/APPEAL/{match.group(1)}/{match.group(2)}"
            header_data["case_year"] = match.group(2)
            header_data["case_num"] = match.group(1)
        
        # Pattern 2: 2024 1 SLR 123
        if not header_data.get("case_number"):
            case_pattern_2 = r'(\d{4})\s+(\d+)\s+SLR\s+(\d+)'
            try:
                match = re.search(case_pattern_2, top_section)
            except TypeError as e:
                logger.error(f"Regex error on case_pattern_2: {e}, top_section type: {type(top_section)}")
                match = None
            if match:
                header_data["case_number"] = f"{match.group(1)} {match.group(2)} SLR {match.group(3)}"
                header_data["case_year"] = match.group(1)
        
        # Extract parties from top
        parties = {}
        try:
            petitioner_pattern = r'(?:Petitioner|Appellant)[:\s]+([A-Z][^v\n]+?)(?:\s+v\.?\s+|$)'
            match = re.search(petitioner_pattern, top_section, re.IGNORECASE)
            if match:
                parties["petitioner"] = match.group(1).strip()
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error extracting petitioner: {e}")
        
        try:
            respondent_pattern = r'v\.?\s+(?:Respondent[:\s]+)?([A-Z][^\n]+?)(?:\n|$)'
            match = re.search(respondent_pattern, top_section, re.IGNORECASE)
            if match:
                parties["respondent"] = match.group(1).strip()
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error extracting respondent: {e}")
        
        if parties:
            header_data["parties"] = parties
        
        # Extract court from top
        try:
            if re.search(r'\bSC\b|\bSupreme\s+Court\b', top_section, re.IGNORECASE):
                header_data["court"] = "Supreme Court"
            else:
                court_pattern = r'(Supreme\s+Court|Court\s+of\s+Appeal|High\s+Court|District\s+Court)'
                match = re.search(court_pattern, top_section, re.IGNORECASE)
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error extracting court: {e}, defaulting to Supreme Court")
            header_data["court"] = "Supreme Court"
            match = None
            if match:
                header_data["court"] = match.group(1)
            else:
                header_data["court"] = "Supreme Court"  # Default
        
        # TAIL: Last 3000 characters for decision date (Sri Lankan footer format)
        tail_section = text[-3000:] if text_len > 3000 else text
        # Ensure tail_section is a string
        if not isinstance(tail_section, str):
            logger.error(f"tail_section is not a string: {type(tail_section)}")
            tail_section = str(tail_section) if tail_section is not None else ""
        
        # Sri Lankan footer format: "Decided on (this) [Day] day of [Month] [Year]"
        # Example: "Decided on this 10th day of October 2025"
        sri_lankan_footer_pattern = r'Decided\s+on\s+(?:this\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+(\w+)\s+(\d{4})'
        try:
            match = re.search(sri_lankan_footer_pattern, tail_section, re.IGNORECASE)
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error in sri_lankan_footer_pattern: {e}, tail_section type: {type(tail_section)}")
            match = None
        if match:
            date_str = self._parse_date(match.group(1), match.group(2), match.group(3))
            header_data["date"] = date_str
            header_data["decision_date"] = date_str  # Also store as decision_date for Neo4j
        
        # Fallback patterns for date (in tail section)
        if not header_data.get("date"):
            try:
                # Pattern: "Decided on: 10th October, 2025"
                date_pattern_1 = r'Decided\s+on[:\s]+(\d{1,2})(?:st|nd|rd|th)?\s+(\w+),?\s+(\d{4})'
                match = re.search(date_pattern_1, tail_section, re.IGNORECASE)
                if match:
                    date_str = self._parse_date(match.group(1), match.group(2), match.group(3))
                    header_data["date"] = date_str
                    header_data["decision_date"] = date_str
            except (TypeError, AttributeError) as e:
                logger.warning(f"Error in date_pattern_1: {e}")
        
        if not header_data.get("date"):
            try:
                # Pattern: "Date of Judgment: 10-10-2025"
                date_pattern_2 = r'Date\s+of\s+Judgment[:\s]+(\d{1,2})[-/](\d{1,2})[-/](\d{4})'
                match = re.search(date_pattern_2, tail_section, re.IGNORECASE)
                if match:
                    date_str = f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}"
                    header_data["date"] = date_str
                    header_data["decision_date"] = date_str
            except (TypeError, AttributeError) as e:
                logger.warning(f"Error in date_pattern_2: {e}")
        
        if not header_data.get("date"):
            try:
                # Pattern: ISO format "2025-10-10" (in tail)
                date_pattern_3 = r'(\d{4}-\d{2}-\d{2})'
                match = re.search(date_pattern_3, tail_section)
                if match:
                    header_data["date"] = match.group(1)
                    header_data["decision_date"] = match.group(1)
            except (TypeError, AttributeError) as e:
                logger.warning(f"Error in date_pattern_3: {e}")
        
        return header_data
    
    def _parse_date(self, day: str, month: str, year: str) -> str:
        """
        Parse date components into ISO format (YYYY-MM-DD).
        
        Args:
            day: Day as string
            month: Month name
            year: Year as string
        
        Returns:
            ISO format date string
        """
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        month_num = month_map.get(month.lower(), '01')
        day_num = day.zfill(2)
        
        return f"{year}-{month_num}-{day_num}"
    
    def _semantic_segmentation(self, text: str) -> Dict:
        """
        Stage B: Semantic Segmentation using Gemini 1.5 Flash.
        
        Segments the judgment into:
        - facts: The narrative history
        - legal_issues: Questions of law
        - arguments: Counsel submissions
        - ratio_decidendi: The binding reasoning (Crucial)
        - obiter_dicta: Non-binding remarks
        - outcome: Final order
        
        Args:
            text: Full text of the judgment
        
        Returns:
            Dictionary with segmented content
        """
        if not self.gemini_client or not self.gemini_model_name:
            logger.warning("Gemini model not available. Returning empty segmentation.")
            return {
                "facts": "",
                "legal_issues": "",
                "arguments": "",
                "ratio_decidendi": "",
                "obiter_dicta": "",
                "outcome": ""
            }
        
        # Handle long documents: if very long, send strategic sections but keep single JSON response
        # Gemini 2.5 Flash Lite has large context window, but we'll optimize for very long docs
        # Strategy: Include beginning (facts), middle (arguments/ratio), and end (outcome/order)
        max_text_length = 500000  # 500k chars should be safe for gemini-2.5-flash-lite
        text_to_analyze = text
        
        if len(text) > max_text_length:
            logger.warning(f"Document is very long ({len(text)} chars). Using strategic sections for analysis.")
            
            # Strategy to capture all critical parts:
            # 1. First 20% (facts, parties, legal issues)
            # 2. Middle section (arguments, ratio decidendi) 
            # 3. Last 20% (outcome, order, signatures) - CRITICAL: Don't skip this!
            
            text_len = len(text)
            first_section_end = int(text_len * 0.2)  # First 20%
            last_section_start = int(text_len * 0.8)  # Last 20% starts here
            middle_section_start = int(text_len * 0.3)  # Middle section start
            middle_section_end = int(text_len * 0.7)  # Middle section end
            
            # Reserve space for separators (markers between sections)
            separator_len = 50  # Approximate length of separator text
            
            # Allocate space: prioritize first and last sections (they contain critical info)
            # Reserve ~30% for first, ~30% for last, ~40% for middle
            available_length = max_text_length - (2 * separator_len)  # Reserve for 2 separators
            first_section_len = min(first_section_end, int(available_length * 0.3))
            last_section_len = min(text_len - last_section_start, int(available_length * 0.3))
            middle_section_len = min(
                middle_section_end - middle_section_start,
                available_length - first_section_len - last_section_len
            )
            
            # Build composite text: beginning + middle + end
            # This ensures we capture Outcome/Order at the end
            text_to_analyze = (
                text[:first_section_len] +  # First section (facts, parties)
                "\n\n[... MIDDLE SECTION ...]\n\n" +
                text[middle_section_start:middle_section_start + middle_section_len] +  # Middle (ratio)
                "\n\n[... FINAL SECTION (OUTCOME/ORDER) ...]\n\n" +
                text[last_section_start:]  # Last section (outcome, order) - CRITICAL!
            )
            
            logger.info(f"Using composite text for long document: "
                       f"first {first_section_len} chars (0-{first_section_len}), "
                       f"middle {middle_section_len} chars ({middle_section_start}-{middle_section_start + middle_section_len}), "
                       f"last {last_section_len} chars ({last_section_start}-{text_len})")
        
        # Construct prompt for Gemini - single JSON for entire document
        prompt = f"""You are a legal expert analyzing a Sri Lankan Supreme Court judgment. 
Extract and segment the following judgment text into structured components.

IMPORTANT: You MUST return ONLY valid JSON. No markdown, no explanations, just JSON.

Text to analyze:
{text_to_analyze}

Extract the following segments from the ENTIRE document:
1. "facts": The narrative history and background of the case
2. "legal_issues": Questions of law that the court must decide
3. "arguments": Key submissions made by counsel for both parties
4. "ratio_decidendi": The binding legal reasoning and ratio (CRUCIAL - this is the precedent)
5. "obiter_dicta": Non-binding remarks and observations
6. "outcome": The final order/disposition (e.g., "Appeal Allowed", "Application Dismissed")

Return ONLY a JSON object with these exact keys:
{{
  "facts": "...",
  "legal_issues": "...",
  "arguments": "...",
  "ratio_decidendi": "...",
  "obiter_dicta": "...",
  "outcome": "..."
}}

Ensure all text is properly escaped for JSON. Return ONLY the JSON object, nothing else."""

        # Retry logic for rate limits (429 errors)
        max_retries = 10
        retry_delay = 20  # Start with 5 seconds (increased from 2)
        
        logger.info(f"Calling Gemini API for semantic segmentation (model: {self.gemini_model_name})...")
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Use new google.genai API (per migration guide)
                # https://ai.google.dev/gemini-api/docs/migrate
                logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt
                )
                elapsed = time.time() - start_time
                logger.info(f"Gemini API call successful (took {elapsed:.1f}s)")
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                elapsed = time.time() - start_time
                
                # Check if it's a rate limit error (429)
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}, {elapsed:.1f}s elapsed). Waiting {wait_time}s before retry...")
                        logger.warning(f"Your quota may be exhausted. Check: https://ai.dev/rate-limit")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts ({elapsed:.1f}s total).")
                        logger.error(f"Quota appears to be exhausted. Please check: https://ai.dev/rate-limit")
                        logger.error(f"Free tier limits: ~15 requests/minute, ~1500 requests/day")
                        raise
                else:
                    # Not a rate limit error, re-raise immediately
                    logger.error(f"Non-rate-limit error after {elapsed:.1f}s: {e}")
                    raise
        
        try:
            
            # Extract JSON from response
            # According to migration guide, response has .text attribute
            # Or access via response.candidates[0].content.parts[0].text
            if hasattr(response, 'text') and response.text:
                response_text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts:
                        response_text = response.candidates[0].content.parts[0].text.strip()
                    else:
                        response_text = str(response.candidates[0].content).strip()
                else:
                    response_text = str(response.candidates[0]).strip()
            else:
                # Fallback: try to get text from response object
                response_text = str(response).strip()
                logger.warning(f"Unexpected response structure: {type(response)}")
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            segmented = json.loads(response_text)
            
            # Ensure all required keys exist
            required_keys = ["facts", "legal_issues", "arguments", "ratio_decidendi", "obiter_dicta", "outcome"]
            for key in required_keys:
                if key not in segmented:
                    segmented[key] = ""
            
            return segmented
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            response_text_safe = response.text[:500] if 'response' in locals() and hasattr(response, 'text') else "N/A"
            logger.error(f"Response text: {response_text_safe}")
            # Return empty structure
            return {
                "facts": "",
                "legal_issues": "",
                "arguments": "",
                "ratio_decidendi": "",
                "obiter_dicta": "",
                "outcome": ""
            }
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                logger.error(f"Gemini API quota exceeded after retries. Please check: https://ai.dev/rate-limit")
                logger.error(f"Error: {e}")
            else:
                logger.error(f"Gemini API error: {e}")
            
            # Return empty segmentation on error
            return {
                "facts": "",
                "legal_issues": "",
                "arguments": "",
                "ratio_decidendi": "",
                "obiter_dicta": "",
                "outcome": ""
            }
    
    def _resolve_citations(self, segmented_content: Dict) -> List[str]:
        """
        Stage C: Citation Resolution (Neuro-Symbolic Bridge).
        
        Scans ratio_decidendi and facts for mentions of Acts and sections.
        Stores as string references.
        
        Args:
            segmented_content: Dictionary with segmented content
        
        Returns:
            List of statute citations as strings
        """
        citations = []
        
        # Combine ratio and facts for citation extraction
        text_to_scan = segmented_content.get("ratio_decidendi", "") + " " + segmented_content.get("facts", "")
        
        # Pattern 1: "Section X of [Act Name]"
        pattern_1 = r'Section\s+(\d+)\s+of\s+(?:the\s+)?([A-Z][a-zA-Z\s]+?Act)'
        matches = re.finditer(pattern_1, text_to_scan, re.IGNORECASE)
        for match in matches:
            citation = f"{match.group(2)}, Sec {match.group(1)}"
            if citation not in citations:
                citations.append(citation)
        
        # Pattern 2: "[Act Name], Section X"
        pattern_2 = r'([A-Z][a-zA-Z\s]+?Act),?\s+Section\s+(\d+)'
        matches = re.finditer(pattern_2, text_to_scan, re.IGNORECASE)
        for match in matches:
            citation = f"{match.group(1)}, Sec {match.group(2)}"
            if citation not in citations:
                citations.append(citation)
        
        # Hardcoded mapping for common Sri Lankan Acts (Symbolic Lookup)
        sri_lankan_acts_mapping = {
            "Act No. 19 of 1990": "High Court of the Provinces (Special Provisions) Act",
            "Act No. 15 of 1979": "Code of Criminal Procedure Act",
            "Act No. 13 of 1979": "Judicature Act",
            "Act No. 2 of 2003": "Prevention of Terrorism Act",
            "Act No. 11 of 1972": "Ceylon (Constitution) Order in Council",
            "Act No. 1 of 1978": "Constitution of the Democratic Socialist Republic of Sri Lanka",
            "Act No. 7 of 2007": "Proceeds of Crime Act",
            "Act No. 9 of 2003": "Bribery Act",
            "Act No. 15 of 1987": "Civil Procedure Code",
            "Act No. 38 of 2005": "Penal Code",
            "Act No. 22 of 1995": "Evidence Ordinance",
        }
        
        # Pattern 3: "Act No. X of YYYY" - map to full name using symbolic lookup
        pattern_3 = r'Act\s+No\.?\s+(\d+)\s+of\s+(\d{4})'
        matches = re.finditer(pattern_3, text_to_scan, re.IGNORECASE)
        for match in matches:
            act_ref = f"Act No. {match.group(1)} of {match.group(2)}"
            full_name = sri_lankan_acts_mapping.get(act_ref)
            
            # Try to extract section number if mentioned nearby
            section_match = re.search(r'Section\s+(\d+)', text_to_scan[max(0, match.start()-100):match.end()+100], re.IGNORECASE)
            
            if full_name:
                if section_match:
                    citation = f"{full_name}, Sec {section_match.group(1)}"
                else:
                    citation = full_name
            else:
                # Keep original if not in mapping
                if section_match:
                    citation = f"{act_ref}, Sec {section_match.group(1)}"
                else:
                    citation = act_ref
            
            if citation not in citations:
                citations.append(citation)
        
        # Pattern 4: "Chapter X" (for old laws)
        pattern_4 = r'Chapter\s+(\d+)'
        matches = re.finditer(pattern_4, text_to_scan, re.IGNORECASE)
        for match in matches:
            citation = f"Chapter {match.group(1)}"
            if citation not in citations:
                citations.append(citation)
        
        return citations
    
    def _vectorize_content(self, segmented_content: Dict) -> Dict:
        """
        Stage D: Vectorization using Nomic Embeddings.
        
        Generates separate embeddings for:
        - facts (for factual retrieval)
        - ratio_decidendi (for legal retrieval)
        
        Prefixes text with "search_document: " as required by Nomic.
        
        Args:
            segmented_content: Dictionary with segmented content
        
        Returns:
            Dictionary with embedding vectors
        """
        vectors = {
            "facts_embedding": None,
            "ratio_embedding": None
        }
        
        if not self.embedding_model:
            logger.warning("Embedding model not available. Returning None vectors.")
            return vectors
        
        try:
            # Embed facts
            facts_text = segmented_content.get("facts", "")
            if facts_text:
                # Prefix with "search_document: " as required by Nomic
                facts_prefixed = f"search_document: {facts_text}"
                facts_embedding = self.embedding_model.encode(facts_prefixed, normalize_embeddings=True)
                vectors["facts_embedding"] = facts_embedding.tolist()
            
            # Embed ratio
            ratio_text = segmented_content.get("ratio_decidendi", "")
            if ratio_text:
                # Prefix with "search_document: " as required by Nomic
                ratio_prefixed = f"search_document: {ratio_text}"
                ratio_embedding = self.embedding_model.encode(ratio_prefixed, normalize_embeddings=True)
                vectors["ratio_embedding"] = ratio_embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
        
        return vectors
    
    def _construct_urn(self, header_data: Dict) -> str:
        """
        Construct URN for the case.
        
        Format: urn:lex:lk:case:sc:appeal:65:2025
        
        Args:
            header_data: Dictionary with header information
        
        Returns:
            URN string
        """
        case_number = header_data.get("case_number", "")
        
        if not case_number:
            # Fallback URN
            date = header_data.get("date", datetime.now().strftime("%Y-%m-%d"))
            return f"urn:lex:lk:case:sc:unknown:{date.replace('-', '')}"
        
        # Parse case number
        # SC/APPEAL/65/2025 -> urn:lex:lk:case:sc:appeal:65:2025
        if "SC/APPEAL" in case_number.upper():
            match = re.search(r'SC/APPEAL/(\d+)/(\d{4})', case_number, re.IGNORECASE)
            if match:
                return f"urn:lex:lk:case:sc:appeal:{match.group(1)}:{match.group(2)}"
        
        # Fallback: use case number as-is
        safe_case_num = re.sub(r'[^a-zA-Z0-9]', ':', case_number.lower())
        return f"urn:lex:lk:case:sc:{safe_case_num}"
    
    def process_pdf_to_json(self, pdf_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process PDF and optionally save to JSON file.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save JSON output
        
        Returns:
            Dictionary with complete case law structure
        """
        result = self.process_pdf(pdf_path)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved output to: {output_path}")
        
        return result


if __name__ == "__main__":
    # Example usage
    import os
    
    # Initialize pipeline
    api_key = os.getenv('GOOGLE_API_KEY')
    pipeline = CaseLawPipeline(gemini_api_key=api_key)
    
    # Process a PDF
    pdf_path = "data/raw/cases/sample_judgment.pdf"
    if os.path.exists(pdf_path):
        result = pipeline.process_pdf_to_json(pdf_path, "output.json")
        print(json.dumps(result, indent=2))
    else:
        print(f"PDF not found: {pdf_path}")
