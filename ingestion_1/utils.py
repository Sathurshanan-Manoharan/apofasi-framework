import re
import unicodedata

def clean_filename(name: str) -> str:
    """Remove invalid filename characters."""
    name = name.lower()
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
    return name.strip("_")


def normalize_text(text: str) -> str:
    """Normalize unicode inconsistencies."""
    return unicodedata.normalize("NFKC", text)


def run_ocr(image_bytes) -> str:
    """
    Placeholder for OCR extraction.
    Implement later using pytesseract or easyocr.
    """
    print("OCR requested but not implemented.")
    return ""
