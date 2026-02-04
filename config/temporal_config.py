from datetime import datetime

# Default query date (current date)
TEMPORAL_CONFIG = {
    # Default date for queries when not specified
    'default_query_date': datetime.now().date().isoformat(),
    
    # Date format specifications
    'date_formats': [
        '%Y-%m-%d',  # ISO format
        '%d/%m/%Y',   # DD/MM/YYYY
        '%m/%d/%Y',   # MM/DD/YYYY
        '%d %B %Y',   # DD Month YYYY
        '%B %d, %Y',  # Month DD, YYYY
        '%Y',         # Just year
    ],
    
    # Temporal relevance weights
    'temporal_weights': {
        'recency_weight': 0.1,  # Weight for recency boost
        'active_boost': 1.2,    # Boost for active documents
        'overruled_penalty': 0.5,  # Penalty for overruled documents
        'amendment_boost': 1.1,    # Boost for recently amended statutes
    },
    
    # Statute versioning rules
    'versioning': {
        'initial_version': '1.0',
        'version_increment': 0.1,  # Increment for minor amendments
        'major_version_increment': 1.0,  # Increment for major amendments
    },
    
    # Temporal query defaults
    'default_top_k': 10,
    'max_temporal_window_days': 3650,  # 10 years for relevance calculation
    
    # Temporal filtering defaults
    'exclude_overruled_by_default': True,
    'exclude_repealed_by_default': True,
    
    # Chronological ordering
    'order_by': 'relevance_then_recency',  # 'relevance', 'recency', 'relevance_then_recency'
    
    # Vector search weights
    'hybrid_search_weights': {
        'vector_weight': 0.6,
        'graph_weight': 0.4,
    },
}

# Legal date patterns for extraction
LEGAL_DATE_PATTERNS = [
    r'\d{1,2}[st|nd|rd|th]?\s+\w+\s+\d{4}',  # "15th March 2010"
    r'\d{4}-\d{2}-\d{2}',  # "2010-03-15"
    r'\d{1,2}/\d{1,2}/\d{4}',  # "15/03/2010"
    r'\w+\s+\d{1,2},?\s+\d{4}',  # "March 15, 2010"
]

# Temporal relationship types
TEMPORAL_RELATIONSHIPS = [
    'OVERRULES',
    'AFFIRMS',
    'FOLLOWS',
    'INTERPRETS',
    'AMENDS',
    'REPEALS',
]

# Document status values
DOCUMENT_STATUS = {
    'ACTIVE': 'active',
    'OVERRULED': 'overruled',
    'AMENDED': 'amended',
    'REPEALED': 'repealed',
}
