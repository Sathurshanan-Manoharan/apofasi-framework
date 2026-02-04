"""
Test script for CaseLawPipeline
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from extraction_2.caselaw_pipeline import CaseLawPipeline
import os

def test_pipeline():
    """Test the CaseLawPipeline with a sample PDF."""
    
    print("=" * 70)
    print("TESTING CASELAW PIPELINE")
    print("=" * 70)
    
    # Initialize pipeline
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("[WARN] GOOGLE_API_KEY not set. Stage B (Semantic Segmentation) will fail.")
        print("       Set it with: export GOOGLE_API_KEY=your_key")
    
    print("\n[1/4] Initializing pipeline...")
    pipeline = CaseLawPipeline(gemini_api_key=api_key)
    print("[OK] Pipeline initialized")
    
    # Check for test PDF
    test_pdf = Path("data/raw/cases")
    pdf_files = list(test_pdf.glob("*.pdf")) if test_pdf.exists() else []
    
    if not pdf_files:
        print(f"\n[INFO] No PDF files found in {test_pdf}")
        print("       Testing with text extraction only...")
        
        # Test with a sample text file if available
        txt_files = list(test_pdf.glob("*.txt")) if test_pdf.exists() else []
        if txt_files:
            print(f"\n[2/4] Testing header extraction with text file...")
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                text = f.read()
            
            headers = pipeline._extract_headers(text)
            print(f"  Case Number: {headers.get('case_number', 'N/A')}")
            print(f"  Date: {headers.get('date', 'N/A')}")
            print(f"  Court: {headers.get('court', 'N/A')}")
            print(f"  Parties: {headers.get('parties', {})}")
            
            print("\n[OK] Header extraction test complete")
        else:
            print("[SKIP] No test files available")
    else:
        # Test with first PDF
        pdf_path = pdf_files[0]
        print(f"\n[2/4] Processing PDF: {pdf_path.name}")
        
        try:
            result = pipeline.process_pdf(str(pdf_path))
            
            print("\n[3/4] Processing Results:")
            print(f"  URN: {result['urn']}")
            print(f"  Case Number: {result['meta']['case_number']}")
            print(f"  Date: {result['meta']['date']}")
            print(f"  Facts length: {len(result['content']['facts'])} chars")
            print(f"  Ratio length: {len(result['content']['ratio'])} chars")
            print(f"  Citations found: {len(result['edges']['statutes'])}")
            print(f"  Facts embedding: {'Yes' if result['vectors']['facts_embedding'] else 'No'}")
            print(f"  Ratio embedding: {'Yes' if result['vectors']['ratio_embedding'] else 'No'}")
            
            # Save output
            output_path = f"test_output_{pdf_path.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n[4/4] Output saved to: {output_path}")
            print("\n[OK] Pipeline test PASSED!")
            
            return True
            
        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
