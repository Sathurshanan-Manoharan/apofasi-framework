"""
Extraction Module for Apofasi Legal RAG System
==============================================

This module contains:
- StatutoryExtraction: Deterministic statute parser
- CaseLawPipeline: Neuro-symbolic case law processor
- Legacy functions: Deprecated but kept for backward compatibility
"""

from extraction_2.statutory_extraction import (
    SriLankanStatuteParser,
    extract_statute_structure,
    build_statute_structure,
    get_section_boundaries,
    build_lrmoo_structure,
    identify_repealed_sections
)

try:
    from extraction_2.caselaw_pipeline import CaseLawPipeline
    __all__ = [
        'SriLankanStatuteParser',
        'extract_statute_structure',
        'build_statute_structure',
        'get_section_boundaries',
        'build_lrmoo_structure',
        'identify_repealed_sections',
        'CaseLawPipeline'
    ]
except ImportError:
    __all__ = [
        'SriLankanStatuteParser',
        'extract_statute_structure',
        'build_statute_structure',
        'get_section_boundaries',
        'build_lrmoo_structure',
        'identify_repealed_sections'
    ]
