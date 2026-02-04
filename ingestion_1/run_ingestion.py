import argparse
from pathlib import Path

from ingestion_1.load_pdf import process_pdf_folder
from ingestion_1.preprocess_text import process_clean_and_chunk
from extraction_2.extract_temporal_metadata import extract_all_temporal_metadata
from extraction_2.statutory_extraction import build_statute_structure, get_section_boundaries
from config.paths import DATA_RAW_CASES, DATA_PROCESSED_CASES, DATA_RAW_STATUTES, DATA_PROCESSED_STATUTES


def run_cases():
    print("=== INGESTION: CASES ===")

    # Step 1: Convert PDFs to raw text
    process_pdf_folder(
        input_dir=DATA_RAW_CASES,
        output_dir=DATA_RAW_CASES
    )

    # Step 2: Extract temporal metadata and chunk with temporal awareness
    raw_files = Path(DATA_RAW_CASES).glob("*.txt")
    for f in raw_files:
        print(f"\nProcessing -> {f.name}")
        
        # Read text for temporal metadata extraction
        with open(f, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Extract temporal metadata
        temporal_metadata = extract_all_temporal_metadata(text, 'case')
        print(f"  Extracted temporal metadata: decision_date={temporal_metadata.get('decision_date')}")
        
        # Clean & chunk with temporal metadata
        process_clean_and_chunk(
            input_path=str(f),
            output_dir=DATA_PROCESSED_CASES,
            chunk_size=2000,
            doc_type='case',
            temporal_metadata=temporal_metadata,
            preserve_sections=False
        )

    print("\n[OK] Completed ingestion for CASES.\n")


def run_statutes():
    print("=== INGESTION: STATUTES ===")

    # Step 1: Convert PDFs -> raw text
    process_pdf_folder(
        input_dir=DATA_RAW_STATUTES,
        output_dir=DATA_RAW_STATUTES
    )

    # Step 2: Extract structure and temporal metadata, then chunk
    raw_files = Path(DATA_RAW_STATUTES).glob("*.txt")
    for f in raw_files:
        print(f"\nProcessing -> {f.name}")
        
        # Read text
        with open(f, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Extract statute structure
        structure = build_statute_structure(text)
        section_boundaries = get_section_boundaries(structure['sections'])
        print(f"  Found {len(structure['sections'])} sections")
        
        # Extract temporal metadata
        temporal_metadata = extract_all_temporal_metadata(text, 'statute')
        print(f"  Extracted temporal metadata: enactment_date={temporal_metadata.get('enactment_date')}")
        
        # Clean & chunk with structure preservation
        process_clean_and_chunk(
            input_path=str(f),
            output_dir=DATA_PROCESSED_STATUTES,
            chunk_size=2000,
            doc_type='statute',
            temporal_metadata=temporal_metadata,
            preserve_sections=True,
            section_boundaries=section_boundaries
        )

    print("\n[OK] Completed ingestion for STATUTES.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Pipeline Runner")
    parser.add_argument(
        "--source",
        required=True,
        choices=["cases", "statutes"],
        help="Choose which data source to ingest"
    )

    args = parser.parse_args()

    if args.source == "cases":
        run_cases()
    elif args.source == "statutes":
        run_statutes()
