import os
import re

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


def chunk_text(text: str, max_chars=2000):
    """Split text into chunks for extraction/embeddings."""
    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) >= max_chars:
            chunks.append(current.strip())
            current = ""
        current += line + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def process_clean_and_chunk(input_path: str, output_dir: str, chunk_size=2000):
    """Clean raw text file and create multiple chunk files."""
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = basic_clean(raw)
    chunks = chunk_text(cleaned, max_chars=chunk_size)

    base = os.path.basename(input_path).replace(".txt", "")

    output_paths = []
    for i, chunk in enumerate(chunks):
        out = f"{output_dir}/{base}_chunk_{i+1}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(chunk)
        output_paths.append(out)

    print(f"Created {len(output_paths)} chunks.")
    return output_paths
