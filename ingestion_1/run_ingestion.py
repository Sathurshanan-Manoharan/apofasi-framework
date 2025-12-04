import argparse
from pathlib import Path

from ingestion_1.load_pdf import process_pdf_folder
from ingestion_1.preprocess_text import process_clean_and_chunk
from config.paths import DATA_RAW_CASES, DATA_PROCESSED_CASES


def run_cases():
    print("=== INGESTION: CASES ===")

    # Step 1: Convert PDFs → raw text
    process_pdf_folder(
        input_dir=DATA_RAW_CASES,
        output_dir=DATA_RAW_CASES
    )

    # Step 2: Clean & chunk each raw text file
    raw_files = Path(DATA_RAW_CASES).glob("*.txt")
    for f in raw_files:
        print(f"\nCleaning + Chunking → {f.name}")
        process_clean_and_chunk(
            input_path=str(f),
            output_dir=DATA_PROCESSED_CASES,
            chunk_size=2000
        )

    print("\n✔ Completed ingestion for CASES.\n")


def run_statutes():
    print("=== INGESTION: STATUTES ===")
    # SAME LOGIC — we'll fill later
    pass


def run_gazettes():
    print("=== INGESTION: GAZETTES ===")
    # SAME LOGIC — fill later
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Pipeline Runner")
    parser.add_argument(
        "--source",
        required=True,
        choices=["cases", "statutes", "gazettes"],
        help="Choose which data source to ingest"
    )

    args = parser.parse_args()

    if args.source == "cases":
        run_cases()
    elif args.source == "statutes":
        run_statutes()
    elif args.source == "gazettes":
        run_gazettes()
