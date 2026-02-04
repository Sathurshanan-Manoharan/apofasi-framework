import os
import fitz  # PyMuPDF
from pathlib import Path
from ingestion_1.utils import clean_filename

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()
    return text


def save_raw_text(output_path: str, text: str):
    """Write extracted text to a .txt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def process_pdf_folder(input_dir: str, output_dir: str):
    """
    Loops through PDFs inside input_dir,
    extracts text, and writes to output_dir.
    """
    input_dir = Path(input_dir)
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in: {input_dir}")
        return

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")

        text = extract_text_from_pdf(str(pdf))
        cleaned_name = clean_filename(pdf.stem)
        output_path = f"{output_dir}/{cleaned_name}.txt"
        save_raw_text(output_path, text)

        print(f"Saved -> {output_path}")


if __name__ == "__main__":
    # EXAMPLE USAGE
    process_pdf_folder(
        input_dir="data/raw/cases_pdfs",
        output_dir="data/raw/cases"
    )
