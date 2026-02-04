"""
Extract temporal metadata (dates) from legal documents.
Handles cases, statutes, and gazettes with various date formats.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dateutil import parser as date_parser


def parse_legal_date(date_str: str) -> Optional[datetime]:
    """
    Parse various legal date formats.
    Handles formats like:
    - "15th March 2010"
    - "2010-03-15"
    - "March 15, 2010"
    - "15/03/2010"
    """
    if not date_str or not date_str.strip():
        return None
    
    date_str = date_str.strip()
    
    # Try dateutil parser first (handles most formats)
    try:
        return date_parser.parse(date_str, dayfirst=True)
    except (ValueError, TypeError):
        pass
    
    # Try common legal formats
    patterns = [
        r"(\d{1,2})[st|nd|rd|th]?\s+(\w+)\s+(\d{4})",  # "15th March 2010"
        r"(\d{4})-(\d{1,2})-(\d{1,2})",  # "2010-03-15"
        r"(\d{1,2})/(\d{1,2})/(\d{4})",  # "15/03/2010"
        r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",  # "March 15, 2010"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                return date_parser.parse(match.group(0), dayfirst=True)
            except (ValueError, TypeError):
                continue
    
    return None


def extract_case_dates(text: str) -> Dict[str, Optional[datetime]]:
    """
    DEPRECATED: Use extraction_2.caselaw_pipeline.CaseLawPipeline instead.
    Date extraction is now part of Stage A of the pipeline.
    """
    """
    Extract temporal dates from case law documents.
    
    Returns:
        {
            'decision_date': datetime or None,
            'hearing_date': datetime or None,
            'filing_date': datetime or None
        }
    """
    dates = {
        'decision_date': None,
        'hearing_date': None,
        'filing_date': None
    }
    
    # Patterns for decision date
    decision_patterns = [
        r"decided\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"judgment\s+(?:dated|on)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"decided\s+(?:on\s+)?(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        r"date[:\s]+(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
    ]
    
    for pattern in decision_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['decision_date'] = parsed
                break
    
    # Patterns for hearing date
    hearing_patterns = [
        r"heard\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"hearing\s+(?:date|on)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
    ]
    
    for pattern in hearing_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['hearing_date'] = parsed
                break
    
    # Patterns for filing date
    filing_patterns = [
        r"filed\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"filing\s+(?:date|on)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
    ]
    
    for pattern in filing_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['filing_date'] = parsed
                break
    
    return dates


def extract_statute_dates(text: str) -> Dict[str, Optional[datetime]]:
    """
    Extract temporal dates from statute documents.
    
    Returns:
        {
            'enactment_date': datetime or None,
            'effective_date': datetime or None,
            'amendment_dates': List[datetime]
        }
    """
    dates = {
        'enactment_date': None,
        'effective_date': None,
        'amendment_dates': []
    }
    
    # Patterns for enactment date
    enactment_patterns = [
        r"enacted\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"enactment\s+(?:date|on)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"act\s+no\.?\s+\d+[,\s]+(\d{4})",  # "Act No. 15, 2010"
    ]
    
    for pattern in enactment_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['enactment_date'] = parsed
                break
    
    # Patterns for effective date
    effective_patterns = [
        r"effective\s+(?:from|on|as\s+of)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"comes\s+into\s+effect\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"shall\s+come\s+into\s+force\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
    ]
    
    for pattern in effective_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['effective_date'] = parsed
                break
    
    # Patterns for amendments
    amendment_patterns = [
        r"amended\s+(?:by|on)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"as\s+amended\s+(?:by|on)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
    ]
    
    for pattern in amendment_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['amendment_dates'].append(parsed)
    
    return dates


def extract_gazette_dates(text: str) -> Dict[str, Optional[datetime]]:
    """
    Extract temporal dates from gazette documents.
    
    Returns:
        {
            'publication_date': datetime or None,
            'effective_date': datetime or None
        }
    """
    dates = {
        'publication_date': None,
        'effective_date': None
    }
    
    # Patterns for publication date
    publication_patterns = [
        r"published\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"gazette\s+(?:dated|of)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"extraordinary\s+gazette\s+no\.?\s+\d+[,\s]+(\d{4})",
    ]
    
    for pattern in publication_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['publication_date'] = parsed
                break
    
    # Patterns for effective date (same as statute)
    effective_patterns = [
        r"effective\s+(?:from|on|as\s+of)\s+(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
        r"comes\s+into\s+effect\s+(?:on\s+)?(?:the\s+)?(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
    ]
    
    for pattern in effective_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = parse_legal_date(match.group(1))
            if parsed:
                dates['effective_date'] = parsed
                break
    
    return dates


def extract_all_temporal_metadata(text: str, doc_type: str) -> Dict:
    """
    Extract all temporal metadata based on document type.
    
    Args:
        text: Document text
        doc_type: 'case', 'statute', or 'gazette'
    
    Returns:
        Dictionary with temporal metadata
    """
    doc_type = doc_type.lower()
    
    if doc_type == 'case':
        dates = extract_case_dates(text)
        return {
            **dates,
            'effective_from': dates.get('decision_date'),
            'effective_to': None,  # Will be set if overruled
            'status': 'active'  # Default, will be updated if overruled
        }
    elif doc_type == 'statute':
        dates = extract_statute_dates(text)
        return {
            **dates,
            'last_amended': dates['amendment_dates'][-1] if dates['amendment_dates'] else None,
            'version': '1.0',  # Default, will be updated based on amendments
            'status': 'active'  # Default, will be updated if repealed
        }
    elif doc_type == 'gazette':
        return extract_gazette_dates(text)
    else:
        return {}


def format_date_for_storage(date: Optional[datetime]) -> Optional[str]:
    """Format datetime for storage (ISO format)."""
    if date is None:
        return None
    return date.isoformat()
