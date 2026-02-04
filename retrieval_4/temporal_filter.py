"""
Core temporal filtering logic for legal document retrieval.
Filters documents by effective dates, excludes overruled cases, resolves versions.
"""

from typing import List, Dict, Optional
from datetime import datetime, date


def filter_by_effective_date(query_date: str, documents: List[Dict]) -> List[Dict]:
    """
    Filter documents to only those effective at the query date.
    
    Args:
        query_date: Date in ISO format (YYYY-MM-DD)
        documents: List of document dicts with effective_from/effective_to
    
    Returns:
        Filtered list of documents
    """
    query_dt = datetime.fromisoformat(query_date).date() if isinstance(query_date, str) else query_date
    
    filtered = []
    for doc in documents:
        effective_from = doc.get('effective_from')
        effective_to = doc.get('effective_to')
        
        # Convert to date if string
        if isinstance(effective_from, str):
            effective_from = datetime.fromisoformat(effective_from).date()
        if isinstance(effective_to, str):
            effective_to = datetime.fromisoformat(effective_to).date()
        
        # Check if document was effective at query date
        if effective_from and effective_from > query_dt:
            continue  # Not yet effective
        
        if effective_to and effective_to < query_dt:
            continue  # No longer effective
        
        filtered.append(doc)
    
    return filtered


def get_active_law_at_date(query_date: str, documents: List[Dict]) -> List[Dict]:
    """
    Get all active law at a specific date.
    Alias for filter_by_effective_date with status check.
    
    Args:
        query_date: Date in ISO format
        documents: List of document dicts
    
    Returns:
        List of active documents
    """
    # First filter by effective date
    active = filter_by_effective_date(query_date, documents)
    
    # Then filter by status
    active = [doc for doc in active if doc.get('status') == 'active']
    
    return active


def exclude_overruled_cases(query_date: str, cases: List[Dict], overruled_relationships: List[Dict] = None) -> List[Dict]:
    """
    Exclude cases that were overruled before or at the query date.
    
    Args:
        query_date: Date in ISO format
        cases: List of case dicts
        overruled_relationships: List of OVERRULES relationship dicts (optional)
    
    Returns:
        List of cases excluding overruled ones
    """
    query_dt = datetime.fromisoformat(query_date).date() if isinstance(query_date, str) else query_date
    
    # Build set of overruled case IDs
    overruled_case_ids = set()
    
    if overruled_relationships:
        for rel in overruled_relationships:
            overruled_date = rel.get('overruled_date')
            if overruled_date:
                if isinstance(overruled_date, str):
                    overruled_date = datetime.fromisoformat(overruled_date).date()
                
                if overruled_date <= query_dt:
                    overruled_case_ids.add(rel.get('target_case_id'))
    
    # Filter out overruled cases
    filtered = []
    for case in cases:
        case_id = case.get('case_id') or case.get('id')
        
        # Check status
        if case.get('status') == 'overruled':
            # Check if overruled before query date
            effective_to = case.get('effective_to')
            if effective_to:
                if isinstance(effective_to, str):
                    effective_to = datetime.fromisoformat(effective_to).date()
                if effective_to <= query_dt:
                    continue  # Was overruled before query date
        
        # Check overruled relationships
        if case_id in overruled_case_ids:
            continue
        
        filtered.append(case)
    
    return filtered


def get_relevant_statute_versions(query_date: str, statute_id: str, versions: List[Dict]) -> Optional[Dict]:
    """
    Get the statute version that was effective at the query date.
    
    Args:
        query_date: Date in ISO format
        statute_id: Statute identifier
        versions: List of version dicts with effective_from/effective_to
    
    Returns:
        Version dict or None
    """
    query_dt = datetime.fromisoformat(query_date).date() if isinstance(query_date, str) else query_date
    
    # Filter versions effective at query date
    relevant_versions = []
    for version in versions:
        if version.get('statute_id') != statute_id:
            continue
        
        effective_from = version.get('effective_from')
        effective_to = version.get('effective_to')
        
        if isinstance(effective_from, str):
            effective_from = datetime.fromisoformat(effective_from).date()
        if isinstance(effective_to, str):
            effective_to = datetime.fromisoformat(effective_to).date()
        
        if effective_from and effective_from > query_dt:
            continue
        
        if effective_to and effective_to < query_dt:
            continue
        
        relevant_versions.append(version)
    
    # Return most recent version
    if relevant_versions:
        # Sort by effective_from descending
        relevant_versions.sort(
            key=lambda v: datetime.fromisoformat(str(v.get('effective_from', ''))).date() if isinstance(v.get('effective_from'), str) else v.get('effective_from', date.min),
            reverse=True
        )
        return relevant_versions[0]
    
    return None


def resolve_query_date(user_query: str, default_date: Optional[str] = None) -> str:
    """
    Resolve query date from user query or use default.
    
    Args:
        user_query: User query text
        default_date: Default date if not specified (defaults to today)
    
    Returns:
        Date in ISO format
    """
    import re
    from datetime import datetime
    
    # Try to extract date from query
    # Pattern: "as of 2015", "in 2010", "2015-01-01"
    date_patterns = [
        r"as\s+of\s+(\d{4}(?:-\d{2}-\d{2})?)",
        r"in\s+(\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4})",
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                # Try to parse
                if len(date_str) == 4:  # Just year
                    parsed_date = datetime(int(date_str), 1, 1)
                else:
                    from dateutil import parser
                    parsed_date = parser.parse(date_str)
                
                return parsed_date.date().isoformat()
            except (ValueError, TypeError):
                continue
    
    # Use default date (today if not provided)
    if default_date:
        return default_date
    
    return datetime.now().date().isoformat()


def filter_chunks_by_temporal_context(
    chunks: List[Dict],
    query_date: str,
    doc_type: Optional[str] = None,
    exclude_overruled: bool = True
) -> List[Dict]:
    """
    Filter chunks by temporal context.
    
    Args:
        chunks: List of chunk dicts
        query_date: Query date
        doc_type: Filter by document type
        exclude_overruled: Exclude chunks from overruled/repealed documents
    
    Returns:
        Filtered chunks
    """
    # Filter by effective date
    filtered = filter_by_effective_date(query_date, chunks)
    
    # Filter by document type
    if doc_type:
        filtered = [chunk for chunk in filtered if chunk.get('source_doc_type') == doc_type]
    
    # Filter by status
    if exclude_overruled:
        filtered = [chunk for chunk in filtered if chunk.get('status') != 'overruled' and chunk.get('status') != 'repealed']
    
    return filtered


def get_temporal_relevance_score(document: Dict, query_date: str) -> float:
    """
    Calculate temporal relevance score for a document.
    More recent documents get higher scores.
    
    Args:
        document: Document dict
        query_date: Query date
    
    Returns:
        Relevance score (0-1)
    """
    query_dt = datetime.fromisoformat(query_date).date() if isinstance(query_date, str) else query_date
    
    effective_from = document.get('effective_from')
    if not effective_from:
        return 0.5  # Default score if no date
    
    if isinstance(effective_from, str):
        effective_from = datetime.fromisoformat(effective_from).date()
    
    # Calculate days difference
    days_diff = (query_dt - effective_from).days
    
    # Normalize: more recent = higher score
    # Assume max relevance window of 10 years
    max_days = 3650
    if days_diff < 0:
        return 0.0  # Not yet effective
    elif days_diff > max_days:
        return 0.1  # Very old
    else:
        # Linear decay: 1.0 at 0 days, 0.1 at max_days
        score = 1.0 - (0.9 * days_diff / max_days)
        return max(0.1, min(1.0, score))
