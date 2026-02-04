"""
StatutoryExtraction Module for Apofasi
========================================

This module implements a deterministic state machine to extract hierarchical
structure from Sri Lankan Acts (PDFs) and output JSON representing the statute
as a hierarchical graph.

Design Rules:
- Hierarchy: Act > Part > Chapter > Section
- URN Format: Akoma Ntoso-compliant (urn:lex:lk:act:YYYY:N!component)
- Output: JSON schema with provisions, hierarchy, properties, and edges
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Provision:
    """Represents a single provision node in the statute graph."""
    urn: str
    type: str = "Provision"
    subtype: str = "Section"
    content: Dict = field(default_factory=dict)
    hierarchy: Dict = field(default_factory=dict)
    properties: Dict = field(default_factory=dict)
    edges: Dict = field(default_factory=dict)


class SriLankanStatuteParser:
    """
    Deterministic State Machine for parsing Sri Lankan Acts.
    
    Implements:
    - Rule A: Hierarchy stack logic (Act > Part > Chapter > Section)
    - Rule B: Pattern recognition with regex
    - Rule C: Canonical URN construction (Akoma Ntoso)
    """
    
    def __init__(self, source_file: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            source_file: Name of the source file (for metadata)
        """
        self.source_file = source_file
        self.provisions: List[Provision] = []
        
        # State machine context stack
        self.context_stack = {
            'part_id': None,
            'chapter_id': None,
            'schedule_id': None
        }
        
        # Document metadata
        self.doc_type = None  # 'act' or 'cap'
        self.doc_id = None    # e.g., '2023:9' for Act No. 9 of 2023
        self.act_number = None
        self.act_year = None
        self.valid_from = None
        
        # Current section tracking
        self.current_section = None
        self.current_section_lines = []
        self.current_section_number = None
        self.current_marginal_note = None
        self.pending_marginal_note_lines = []
        self.in_marginal_note_zone = False
        self.section_content_ended = False  # Track when section content has ended
        
        # Regex patterns (compiled for efficiency)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns used for pattern recognition."""
        # Work Identification
        self.act_pattern = re.compile(
            r'Act\s*,\s*No\.?\s*(\d+)\s+of\s+(\d{4})',
            re.IGNORECASE
        )
        self.chapter_pattern = re.compile(
            r'Chapter\s+(\d+)',
            re.IGNORECASE
        )
        
        # Hierarchy markers
        self.part_pattern = re.compile(
            r'^PART\s+([IVX]+|[A-Z0-9]+)',
            re.IGNORECASE | re.MULTILINE
        )
        self.chapter_marker_pattern = re.compile(
            r'^CHAPTER\s+([IVX]+|[A-Z0-9]+)',
            re.IGNORECASE | re.MULTILINE
        )
        self.schedule_pattern = re.compile(
            r'^SCHEDULE\s+([IVX]+|[A-Z])',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Section detection
        self.section_start_pattern = re.compile(
            r'^(\d+)\.\s*$',
            re.MULTILINE
        )
        self.section_start_with_text_pattern = re.compile(
            r'^(\d+)\.\s+(.+)$',
            re.MULTILINE
        )
        
        # Marginal note detection (appears after section, often on separate lines)
        # Marginal notes are typically short, title-case lines that don't start with numbers
        self.marginal_note_candidate_pattern = re.compile(
            r'^[A-Z][a-zA-Z\s]{1,80}$'  # Title case, reasonable length
        )
        
        # Page number/header patterns (these separate sections from marginal notes)
        self.page_header_pattern = re.compile(
            r'^\d+\s*$|^Anti-Corruption Act|^PARLIAMENT|^Printed|^Price|^Postage',
            re.IGNORECASE
        )
        
        # Definition detection
        self.interpretation_pattern = re.compile(
            r'Interpretation',
            re.IGNORECASE
        )
        self.definition_text_pattern = re.compile(
            r'In this Act, unless the context otherwise requires',
            re.IGNORECASE
        )
        
        # Date extraction
        self.certified_date_pattern = re.compile(
            r'\[?Certified\s+on\s+(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(\w+),?\s+(\d{4})\]?',
            re.IGNORECASE
        )
        
        # Reference detection (for edges)
        self.section_ref_pattern = re.compile(
            r'Section\s+(\d+)(?:\([^)]+\))?',
            re.IGNORECASE
        )
    
    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral to integer."""
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        result = 0
        prev_value = 0
        
        for char in roman.upper():
            value = roman_map.get(char, 0)
            if value > prev_value:
                result += value - 2 * prev_value
            else:
                result += value
            prev_value = value
        
        return result
    
    def _int_to_roman(self, num: int) -> str:
        """Convert integer to Roman numeral."""
        val = [
            1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1
        ]
        syms = [
            "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"
        ]
        roman = ""
        i = 0
        while num > 0:
            for _ in range(num // val[i]):
                roman += syms[i]
                num -= val[i]
            i += 1
        return roman
    
    def _extract_work_identification(self, text: str) -> bool:
        """
        Extract work identification (Act or Chapter).
        Implements Rule B.1: Work Identification.
        
        Returns:
            True if work identification found, False otherwise
        """
        if not text or len(text.strip()) < 50:
            logger.warning(f"Text too short or empty for work identification (length: {len(text) if text else 0})")
            return False
        
        # Try Act pattern first (more flexible patterns)
        # Pattern 1: "Act, No. X of YYYY" (standard format)
        act_match = self.act_pattern.search(text)
        if act_match:
            self.act_number = act_match.group(1)
            self.act_year = act_match.group(2)
            self.doc_type = 'act'
            self.doc_id = f"{self.act_year}:{self.act_number}"
            logger.info(f"Identified Act: No. {self.act_number} of {self.act_year}")
            return True
        
        # Pattern 2: "Act No. X of YYYY" (without comma)
        act_pattern_2 = re.compile(r'Act\s+No\.?\s*(\d+)\s+of\s+(\d{4})', re.IGNORECASE)
        act_match_2 = act_pattern_2.search(text)
        if act_match_2:
            self.act_number = act_match_2.group(1)
            self.act_year = act_match_2.group(2)
            self.doc_type = 'act'
            self.doc_id = f"{self.act_year}:{self.act_number}"
            logger.info(f"Identified Act (pattern 2): No. {self.act_number} of {self.act_year}")
            return True
        
        # Pattern 3: "Act Number X of YYYY"
        act_pattern_3 = re.compile(r'Act\s+Number\s+(\d+)\s+of\s+(\d{4})', re.IGNORECASE)
        act_match_3 = act_pattern_3.search(text)
        if act_match_3:
            self.act_number = act_match_3.group(1)
            self.act_year = act_match_3.group(2)
            self.doc_type = 'act'
            self.doc_id = f"{self.act_year}:{self.act_number}"
            logger.info(f"Identified Act (pattern 3): No. {self.act_number} of {self.act_year}")
            return True
        
        # Try Chapter pattern
        chapter_match = self.chapter_pattern.search(text)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            self.doc_type = 'cap'
            self.doc_id = chapter_num
            logger.info(f"Identified Chapter: {chapter_num}")
            return True
        
        # Log first 500 chars for debugging
        logger.warning(f"Could not identify Act or Chapter. First 500 chars: {text[:500]}")
        return False
    
    def _extract_enactment_date(self, text: str) -> Optional[str]:
        """
        Extract enactment/certification date from text.
        
        Returns:
            Date string in YYYY-MM-DD format, or None
        """
        date_match = self.certified_date_pattern.search(text)
        if date_match:
            day = date_match.group(1).zfill(2)
            month_name = date_match.group(2).lower()
            year = date_match.group(3)
            
            month_map = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04',
                'may': '05', 'june': '06', 'july': '07', 'august': '08',
                'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            
            month = month_map.get(month_name, '01')
            return f"{year}-{month}-{day}"
        
        # Fallback: use year from Act identification
        if self.act_year:
            return f"{self.act_year}-01-01"
        
        return None
    
    def _generate_urn(self, component_type: str, component_id: str) -> str:
        """
        Generate Akoma Ntoso-compliant URN.
        Implements Rule C: Canonical URN Construction.
        
        Args:
            component_type: 'sec', 'sched', 'part', 'chapter'
            component_id: The identifier (e.g., '12', 'I', 'II')
        
        Returns:
            URN string (e.g., 'urn:lex:lk:act:2023:9!sec12')
        """
        if not self.doc_type or not self.doc_id:
            return f"urn:lex:lk:unknown:unknown!{component_type}{component_id}"
        
        if self.doc_type == 'act':
            doc_identifier = f"act:{self.doc_id}"
        else:  # cap
            doc_identifier = f"cap:{self.doc_id}"
        
        # Normalize component ID
        if component_type == 'sec':
            # Ensure numeric sections are just numbers
            comp_id = component_id if component_id.isdigit() else component_id
        elif component_type == 'sched':
            # Schedules use Roman numerals
            comp_id = component_id.upper()
        else:
            comp_id = component_id
        
        return f"urn:lex:lk:{doc_identifier}!{component_type}{comp_id}"
    
    def _is_definition_node(self, section_text: str, marginal_note: Optional[str]) -> bool:
        """
        Detect if a section is a definition node.
        Implements Rule B.3: Definition Detection.
        
        Args:
            section_text: The text of the section
            marginal_note: The marginal note (if any)
        
        Returns:
            True if this is a definition node
        """
        # Check marginal note
        if marginal_note and self.interpretation_pattern.search(marginal_note):
            return True
        
        # Check section text
        if self.definition_text_pattern.search(section_text):
            return True
        
        return False
    
    def _extract_references(self, text: str) -> List[str]:
        """
        Extract section references from text for edge construction.
        
        Args:
            text: The text to search for references
        
        Returns:
            List of URNs that this provision refers to
        """
        references = []
        matches = self.section_ref_pattern.finditer(text)
        
        for match in matches:
            section_num = match.group(1)
            ref_urn = self._generate_urn('sec', section_num)
            if ref_urn not in references:
                references.append(ref_urn)
        
        return references
    
    def _finalize_current_section(self):
        """Finalize and save the current section being processed."""
        if not self.current_section_number:
            return
        
        # Apply any pending marginal notes
        if self.pending_marginal_note_lines and not self.current_marginal_note:
            self.current_marginal_note = ' '.join(self.pending_marginal_note_lines)
            self.pending_marginal_note_lines = []
        
        section_text = '\n'.join(self.current_section_lines).strip()
        if not section_text:
            # Reset state even if no text
            self.current_section = None
            self.current_section_lines = []
            self.current_section_number = None
            self.current_marginal_note = None
            self.pending_marginal_note_lines = []
            self.in_marginal_note_zone = False
            return
        
        # Generate URN
        urn = self._generate_urn('sec', self.current_section_number)
        
        # Determine if definition node
        is_definition = self._is_definition_node(section_text, self.current_marginal_note)
        
        # Extract references
        references = self._extract_references(section_text)
        
        # Create provision
        provision = Provision(
            urn=urn,
            type="Provision",
            subtype="Section",
            content={
                "text": section_text,
                "label": f"Section {self.current_section_number}",
                "marginal_note": self.current_marginal_note if self.current_marginal_note else None
            },
            hierarchy={
                "part_id": self.context_stack['part_id'],
                "chapter_id": self.context_stack['chapter_id'],
                "schedule_id": self.context_stack['schedule_id']
            },
            properties={
                "is_definition_node": is_definition,
                "valid_from": self.valid_from,
                "source_file": self.source_file
            },
            edges={
                "refers_to": references
            }
        )
        
        # Remove None values from hierarchy
        provision.hierarchy = {k: v for k, v in provision.hierarchy.items() if v is not None}
        
        # Remove None from content
        if provision.content.get("marginal_note") is None:
            provision.content.pop("marginal_note", None)
        
        self.provisions.append(provision)
        
        # Reset current section
        self.current_section = None
        self.current_section_lines = []
        self.current_section_number = None
        self.current_marginal_note = None
        self.pending_marginal_note_lines = []
        self.in_marginal_note_zone = False
        self.section_content_ended = False
    
    def _is_section_end(self, line_stripped: str) -> bool:
        """
        Determine if the current line indicates the end of a section.
        
        Args:
            line_stripped: Stripped line to check
        
        Returns:
            True if this line indicates section end
        """
        # New section number
        if self.section_start_pattern.match(line_stripped) or \
           self.section_start_with_text_pattern.match(line_stripped):
            return True
        
        # Part, Chapter, Schedule markers
        if self.part_pattern.match(line_stripped) or \
           self.chapter_marker_pattern.match(line_stripped) or \
           self.schedule_pattern.match(line_stripped):
            return True
        
        return False
    
    def _process_line(self, line: str, line_num: int):
        """
        Process a single line of text using the state machine.
        Implements Rule A: Hierarchy Stack Logic.
        
        Args:
            line: The line to process
            line_num: Line number (for debugging)
        """
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            if self.current_section_number:
                self.current_section_lines.append(line)
            return
        
        # Check for Part (clears previous Parts)
        part_match = self.part_pattern.match(line_stripped)
        if part_match:
            self._finalize_current_section()
            part_id = f"PART {part_match.group(1)}"
            self.context_stack['part_id'] = part_id
            self.context_stack['chapter_id'] = None  # Part clears chapters
            self.in_marginal_note_zone = False
            return
        
        # Check for Chapter
        chapter_match = self.chapter_marker_pattern.match(line_stripped)
        if chapter_match:
            self._finalize_current_section()
            chapter_id = f"CHAPTER {chapter_match.group(1)}"
            self.context_stack['chapter_id'] = chapter_id
            self.in_marginal_note_zone = False
            return
        
        # Check for Schedule
        schedule_match = self.schedule_pattern.match(line_stripped)
        if schedule_match:
            self._finalize_current_section()
            schedule_id = f"SCHEDULE {schedule_match.group(1)}"
            self.context_stack['schedule_id'] = schedule_id
            self.in_marginal_note_zone = False
            # TODO: Handle schedule content separately if needed
            return
        
        # Check for Section start (Rule B.2: Section Detection)
        section_match = self.section_start_pattern.match(line_stripped)
        section_with_text_match = self.section_start_with_text_pattern.match(line_stripped)
        
        if section_match or section_with_text_match:
            # Finalize previous section
            self._finalize_current_section()
            
            # Start new section
            if section_match:
                self.current_section_number = section_match.group(1)
                self.current_section_lines = [line]
            else:  # section_with_text_match
                self.current_section_number = section_with_text_match.group(1)
                section_text = section_with_text_match.group(2)
                # If the text after section number looks like a marginal note (short, title case)
                if len(section_text) < 80 and not section_text.startswith('('):
                    self.current_marginal_note = section_text.strip()
                else:
                    self.current_section_lines = [line]
            
            self.in_marginal_note_zone = False
            self.section_content_ended = False
            return
        
        # If we're in a section, check if this line ends the section
        if self.current_section_number:
            # Check if this is a new section start
            if self._is_section_end(line_stripped):
                # This is the start of a new section - finalize current one first
                self._finalize_current_section()
                # Then process this line as a new section (recursive call)
                self._process_line(line, line_num)
                return
            
            # Check if section content has ended (we see a potential marginal note or page header)
            # Marginal notes appear after section content, before page numbers
            if not self.section_content_ended:
                # Check if this looks like the end of section content
                # If we see a page header or a marginal note candidate, section content has ended
                if self.page_header_pattern.match(line_stripped):
                    # Page header - section content has ended, but this is not a marginal note
                    self.section_content_ended = True
                    return
                elif self.marginal_note_candidate_pattern.match(line_stripped):
                    # This might be the start of a marginal note
                    self.section_content_ended = True
                    self.in_marginal_note_zone = True
                    self.pending_marginal_note_lines.append(line_stripped)
                    return
                else:
                    # Still in section content
                    self.current_section_lines.append(line)
                    return
            
            # Section content has ended - we're looking for marginal notes or page headers
            if self.in_marginal_note_zone:
                # Collecting marginal note lines
                if self.marginal_note_candidate_pattern.match(line_stripped):
                    self.pending_marginal_note_lines.append(line_stripped)
                elif self.page_header_pattern.match(line_stripped):
                    # Page header - marginal note zone has ended
                    if self.pending_marginal_note_lines:
                        self.current_marginal_note = ' '.join(self.pending_marginal_note_lines)
                        self.pending_marginal_note_lines = []
                    self.in_marginal_note_zone = False
                else:
                    # Not a marginal note candidate and not a page header
                    # This might be the start of a new section
                    if self._is_section_end(line_stripped):
                        # Apply any collected marginal notes before finalizing
                        if self.pending_marginal_note_lines:
                            self.current_marginal_note = ' '.join(self.pending_marginal_note_lines)
                            self.pending_marginal_note_lines = []
                        self._finalize_current_section()
                        self._process_line(line, line_num)
                        return
            elif self.page_header_pattern.match(line_stripped):
                # Just a page header, skip it
                return
            else:
                # Unexpected line after section content - might be new section
                if self._is_section_end(line_stripped):
                    self._finalize_current_section()
                    self._process_line(line, line_num)
                    return
            return
        
        # Not in a section - might be preamble or other content
        # Check if this looks like the start of a section
        if self._is_section_end(line_stripped):
            self._process_line(line, line_num)
    
    def parse(self, text: str) -> List[Dict]:
        """
        Parse raw text and extract hierarchical statute structure.
        
        Args:
            text: Raw text string from Sri Lankan Act PDF
        
        Returns:
            List of provision dictionaries matching the JSON schema
        """
        # Reset state
        self.provisions = []
        self.context_stack = {
            'part_id': None,
            'chapter_id': None,
            'schedule_id': None
        }
        self.current_section = None
        self.current_section_lines = []
        self.current_section_number = None
        self.current_marginal_note = None
        self.pending_marginal_note_lines = []
        self.in_marginal_note_zone = False
        self.section_content_ended = False
        
        # Extract work identification
        if not self._extract_work_identification(text):
            raise ValueError("Could not identify Act or Chapter in text")
        
        # Extract enactment date
        self.valid_from = self._extract_enactment_date(text)
        
        # Process line by line
        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            try:
                self._process_line(line, line_num)
            except Exception as e:
                # Log error but continue processing
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
        
        # Finalize last section
        self._finalize_current_section()
        
        # Convert provisions to dictionaries
        result = []
        for provision in self.provisions:
            prov_dict = {
                "urn": provision.urn,
                "type": provision.type,
                "subtype": provision.subtype,
                "content": provision.content,
                "hierarchy": provision.hierarchy,
                "properties": provision.properties,
                "edges": provision.edges
            }
            result.append(prov_dict)
        
        return result
    
    def parse_to_json(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Parse text and return JSON string.
        
        Args:
            text: Raw text string from Sri Lankan Act PDF
            output_path: Optional path to save JSON file
        
        Returns:
            JSON string representation of the provisions
        """
        provisions = self.parse(text)
        
        output = {
            "metadata": {
                "doc_type": self.doc_type,
                "doc_id": self.doc_id,
                "act_number": self.act_number,
                "act_year": self.act_year,
                "valid_from": self.valid_from,
                "source_file": self.source_file
            },
            "provisions": provisions
        }
        
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str


def extract_statute_structure(text: str, source_file: Optional[str] = None) -> Dict:
    """
    Convenience function to extract statute structure.
    
    Args:
        text: Raw text string from Sri Lankan Act PDF
        source_file: Name of the source file
    
    Returns:
        Dictionary with metadata and provisions
    """
    parser = SriLankanStatuteParser(source_file=source_file)
    provisions = parser.parse(text)
    
    return {
        "metadata": {
            "doc_type": parser.doc_type,
            "doc_id": parser.doc_id,
            "act_number": parser.act_number,
            "act_year": parser.act_year,
            "valid_from": parser.valid_from,
            "source_file": parser.source_file
        },
        "provisions": provisions
    }


# ============================================
# BACKWARD COMPATIBILITY ADAPTERS
# ============================================
# These functions provide compatibility with the old extract_statute_structure.py
# They adapt the new deterministic parser output to the old LRMoo format

def get_section_boundaries(provisions: List[Dict]) -> List[Tuple[int, int]]:
    """
    Get section boundaries for chunking (backward compatibility).
    
    Args:
        provisions: List of provision dictionaries from parse()
    
    Returns:
        List of (start_pos, end_pos) tuples
    """
    # Note: The new parser doesn't track character positions the same way
    # This is a simplified adapter that returns dummy positions
    # For actual chunking, use the provision text directly
    boundaries = []
    for i, prov in enumerate(provisions):
        # Estimate positions based on text length
        start = sum(len(p.get('content', {}).get('text', '')) for p in provisions[:i])
        end = start + len(prov.get('content', {}).get('text', ''))
        boundaries.append((start, end))
    return boundaries


def build_statute_structure(text: str) -> Dict:
    """
    Legacy function - extracts statute structure using new deterministic parser.
    
    Args:
        text: Raw text string from Sri Lankan Act PDF
    
    Returns:
        Dictionary with 'sections' list (for backward compatibility)
    """
    parser = SriLankanStatuteParser()
    provisions = parser.parse(text)
    
    # Convert to old format with SectionExpression-like objects
    sections = []
    for prov in provisions:
        # Extract section number from URN (e.g., "urn:lex:lk:act:2023:9!sec12" -> "12")
        urn = prov['urn']
        section_match = re.search(r'!sec(\d+)', urn)
        section_number = section_match.group(1) if section_match else "unknown"
        
        sections.append({
            'section_number': section_number,
            'text': prov['content']['text'],
            'start_pos': 0,  # Not tracked in new parser
            'end_pos': len(prov['content']['text']),
            'level': 1,
            'parent_section': None
        })
    
    return {
        'sections': sections,
        'metadata': {
            'title': f"Act No. {parser.act_number} of {parser.act_year}" if parser.act_number else "Unknown",
            'year': parser.act_year or 'unknown',
            'act_number': parser.act_number or 'unknown'
        }
    }


def identify_repealed_sections(new_act_text: str, new_act_metadata: Dict) -> List[Dict]:
    """
    Identify which sections from old acts are repealed by this new act.
    (Backward compatibility adapter)
    
    Args:
        new_act_text: Text of the new act
        new_act_metadata: Metadata dictionary
    
    Returns:
        List of dicts with 'target_act', 'target_sections', 'repeal_type'
    """
    repeals = []
    
    # Pattern: "The Bribery Act is hereby repealed"
    full_repeal_pattern = re.compile(
        r'(?:The\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act(?:\s+No\.?\s+\d+\s+of\s+\d{4})?\s+is\s+hereby\s+repealed',
        re.IGNORECASE
    )
    matches = full_repeal_pattern.finditer(new_act_text)
    
    for match in matches:
        act_name = match.group(1)
        repeals.append({
            'target_act': act_name,
            'target_sections': [],  # Empty means entire act
            'repeal_type': 'full',
            'context': match.group(0)
        })
    
    # Pattern: "Section 24 of the Bribery Act is repealed"
    section_repeal_pattern = re.compile(
        r'Section\s+(\d+(?:\([^)]+\))?)\s+of\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Act\s+is\s+repealed',
        re.IGNORECASE
    )
    matches = section_repeal_pattern.finditer(new_act_text)
    
    for match in matches:
        section_num = match.group(1)
        act_name = match.group(2)
        repeals.append({
            'target_act': act_name,
            'target_sections': [section_num],
            'repeal_type': 'partial',
            'context': match.group(0)
        })
    
    return repeals


def build_lrmoo_structure(text: str, statute_metadata: Dict) -> Dict:
    """
    Build LRMoo-compatible structure using new deterministic parser.
    (Backward compatibility adapter for lrmoo_builder.py)
    
    Args:
        text: Full statute text
        statute_metadata: Dict with 'title', 'year', 'act_number', etc.
    
    Returns:
        {
            'work': {...},  # Work node data
            'expressions': [...],  # List of Expression node data
            'event': {...},  # Event node data
            'sections': [...]  # Section objects for reference
        }
    """
    parser = SriLankanStatuteParser(source_file=statute_metadata.get('source_file'))
    provisions = parser.parse(text)
    
    # Parse enactment date
    enactment_date = parser.valid_from or statute_metadata.get('enactment_date')
    
    # Create Work data
    work_id = f"work_{statute_metadata.get('title', 'unknown').lower().replace(' ', '_')}"
    work_data = {
        'work_id': work_id,
        'title': statute_metadata.get('title', 'Unknown'),
        'jurisdiction': 'Sri Lanka',
        'legal_domain': statute_metadata.get('legal_domain', 'General'),
        'original_enactment_year': statute_metadata.get('year', parser.act_year or 'unknown')
    }
    
    # Create Event data
    event_id = f"event_{statute_metadata.get('year', parser.act_year or 'unknown')}_{statute_metadata.get('act_number', parser.act_number or 'unknown')}"
    event_data = {
        'event_id': event_id,
        'event_type': 'enactment',
        'act_number': statute_metadata.get('act_number', parser.act_number or ''),
        'act_year': str(statute_metadata.get('year', parser.act_year or '')),
        'event_date': enactment_date,
        'description': f"{statute_metadata.get('title', 'Unknown')} - Act No. {statute_metadata.get('act_number', parser.act_number or '')} of {statute_metadata.get('year', parser.act_year or '')}"
    }
    
    # Create Expression data for each provision
    expressions = []
    section_objects = []  # For backward compatibility
    
    for prov in provisions:
        # Extract section number from URN
        urn = prov['urn']
        section_match = re.search(r'!sec(\d+)', urn)
        section_number = section_match.group(1) if section_match else "unknown"
        
        expression_id = f"expr_{work_id}_s{section_number}_v{statute_metadata.get('year', parser.act_year or 'unknown')}"
        
        expression_data = {
            'expression_id': expression_id,
            'work_id': work_id,
            'section_number': section_number,
            'version': str(statute_metadata.get('year', parser.act_year or 'unknown')),
            'start_date': enactment_date,
            'end_date': None,  # NULL for active sections
            'status': 'active',
            'text': prov['content']['text'],
            'level': 1,  # Default level
            'parent_section': None
        }
        
        expressions.append(expression_data)
        
        # Create SectionExpression-like object for backward compatibility
        section_objects.append({
            'section_number': section_number,
            'text': prov['content']['text'],
            'start_pos': 0,
            'end_pos': len(prov['content']['text']),
            'level': 1,
            'parent_section': None
        })
    
    return {
        'work': work_data,
        'expressions': expressions,
        'event': event_data,
        'sections': section_objects  # Keep for backward compatibility
    }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    ANTI-CORRUPTION ACT, No. 9 OF 2023
    [Certified on 08th of August, 2023]
    
    PART I
    CHAPTER I
    
    1. (1) This Act may be cited as the Anti-Corruption Act, No. 9 of 2023.
    
    2. In this Act, unless the context otherwise requires, "Commission" means...
    """
    
    parser = SriLankanStatuteParser(source_file="sample.txt")
    result = parser.parse_to_json(sample_text)
    print(result)
