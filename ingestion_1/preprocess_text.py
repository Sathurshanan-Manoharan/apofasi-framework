import os
import re
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ingestion_1.utils import normalize_text

def basic_clean(text: str) -> str:
    """Remove page numbers, extra spaces, weird artifacts."""
    text = normalize_text(text)

    # Remove common page patterns
    text = re.sub(r"\n?\s*Page\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?\s*\d+\s*\n", "\n", text)

    # Remove multiple newlines
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Remove multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def chunk_text(text: str, max_chars=2000, preserve_sections: bool = False, section_boundaries: Optional[List[Tuple[int, int]]] = None):
    """
    Split text into chunks for extraction/embeddings.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        preserve_sections: If True, don't split within section boundaries
        section_boundaries: List of (start_pos, end_pos) tuples for section boundaries
    """
    chunks = []
    
    if preserve_sections and section_boundaries:
        # Chunk respecting section boundaries
        for start_pos, end_pos in section_boundaries:
            section_text = text[start_pos:end_pos]
            
            # If section is small enough, keep as one chunk
            if len(section_text) <= max_chars:
                chunks.append(section_text.strip())
            else:
                # Split section into sub-chunks
                current = ""
                for line in section_text.split("\n"):
                    if len(current) + len(line) >= max_chars:
                        if current.strip():
                            chunks.append(current.strip())
                        current = ""
                    current += line + "\n"
                
                if current.strip():
                    chunks.append(current.strip())
    else:
        # Standard chunking (for cases or statutes without structure)
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) >= max_chars:
                if current.strip():
                    chunks.append(current.strip())
                current = ""
            current += line + "\n"

        if current.strip():
            chunks.append(current.strip())

    return chunks


def add_temporal_metadata_header(chunk: str, metadata: Dict) -> str:
    """
    Add temporal metadata as a header to the chunk.
    This metadata will be preserved in embeddings.
    """
    header_parts = []
    
    # Add document type
    if 'doc_type' in metadata:
        header_parts.append(f"[DOC_TYPE: {metadata['doc_type']}]")
    
    # Add dates
    date_fields = ['decision_date', 'enactment_date', 'effective_date', 'publication_date']
    for field in date_fields:
        if field in metadata and metadata[field]:
            date_str = metadata[field].isoformat() if hasattr(metadata[field], 'isoformat') else str(metadata[field])
            header_parts.append(f"[{field.upper()}: {date_str}]")
    
    # Add status
    if 'status' in metadata:
        header_parts.append(f"[STATUS: {metadata['status']}]")
    
    # Add section info for statutes
    if 'section_number' in metadata:
        header_parts.append(f"[SECTION: {metadata['section_number']}]")
    
    if header_parts:
        header = " ".join(header_parts) + "\n\n"
        return header + chunk
    return chunk


def process_clean_and_chunk(
    input_path: str, 
    output_dir: str, 
    chunk_size=2000,
    doc_type: str = "case",
    temporal_metadata: Optional[Dict] = None,
    preserve_sections: bool = False,
    section_boundaries: Optional[List[Tuple[int, int]]] = None
):
    """
    Clean raw text file and create multiple chunk files with temporal metadata.
    
    Args:
        input_path: Path to input text file
        output_dir: Output directory for chunks
        chunk_size: Maximum characters per chunk
        doc_type: 'case', 'statute', or 'gazette'
        temporal_metadata: Dictionary with temporal metadata to add to chunks
        preserve_sections: If True, preserve section boundaries (for statutes)
        section_boundaries: List of (start_pos, end_pos) for section boundaries
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = basic_clean(raw)
    
    # Chunk with section preservation if needed
    chunks = chunk_text(
        cleaned, 
        max_chars=chunk_size,
        preserve_sections=preserve_sections,
        section_boundaries=section_boundaries
    )

    base = os.path.basename(input_path).replace(".txt", "")
    
    # Prepare metadata for chunks
    if temporal_metadata is None:
        temporal_metadata = {}
    temporal_metadata['doc_type'] = doc_type

    output_paths = []
    metadata_list = []
    
    for i, chunk in enumerate(chunks):
        # Add section metadata if available
        chunk_metadata = temporal_metadata.copy()
        if preserve_sections and section_boundaries and i < len(section_boundaries):
            # Try to extract section number from chunk
            section_match = re.search(r"(?:Section|S\.)\s*(\d+(?:\([^)]+\))?)", chunk, re.IGNORECASE)
            if section_match:
                chunk_metadata['section_number'] = section_match.group(1)
        
        # Add temporal metadata header to chunk
        chunk_with_metadata = add_temporal_metadata_header(chunk, chunk_metadata)
        
        # Save chunk
        out = f"{output_dir}/{base}_chunk_{i+1}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(chunk_with_metadata)
        output_paths.append(out)
        
        # Save metadata separately 
        metadata_file = f"{output_dir}/{base}_chunk_{i+1}_metadata.json"
        # Convert datetime objects to strings for JSON
        json_metadata = {}
        for k, v in chunk_metadata.items():
            if hasattr(v, 'isoformat'):
                json_metadata[k] = v.isoformat()
            elif isinstance(v, list):
                json_metadata[k] = [item.isoformat() if hasattr(item, 'isoformat') else item for item in v]
            else:
                json_metadata[k] = v
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(json_metadata, f, indent=2)
        
        metadata_list.append(json_metadata)

    print(f"Created {len(output_paths)} chunks with temporal metadata.")
    return output_paths, metadata_list
