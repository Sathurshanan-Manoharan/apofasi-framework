import argparse
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from modeling_3.graph_schema.lrmoo_schema import LRMooSchemaInitializer
from modeling_3.neo4j_ops import TemporalNeo4jOps
from modeling_3.vector_ops import TemporalVectorOps
from retrieval_4.temporal_reconstruction import TemporalReconstruction
from retrieval_4.hybrid_search import TemporalHybridSearch
from retrieval_4.query_router import TemporalQueryRouter
from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def setup_schema():
    """Initialize LRMoo Neo4j schema."""
    print("\n" + "=" * 70)
    print("STEP 1: Initializing LRMoo Schema")
    print("=" * 70)
    
    neo4j_ops = TemporalNeo4jOps(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        initializer = LRMooSchemaInitializer(neo4j_ops)
        initializer.initialize_all()
    finally:
        neo4j_ops.close()

def clear_database():
    """Clear all Neo4j data (keeps schema definitions/config in code)."""
    print("\n" + "=" * 70)
    print("STEP 0: Clearing Neo4j Database")
    print("=" * 70)
    neo4j_ops = TemporalNeo4jOps(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        # Use session directly to properly consume result
        with neo4j_ops.driver.session() as session:
            result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
            # Consume the result properly
            record = result.single()
            deleted_count = record["deleted"] if record else 0
            # Consume summary to avoid buffer issues
            result.consume()
            print(f"[OK] Neo4j cleared. Deleted {deleted_count} nodes.")
    except Exception as e:
        print(f"[WARN] Error during clear (may be empty DB): {e}")
        import traceback
        traceback.print_exc()
    finally:
        neo4j_ops.close()

def run_ingestion(source: str = "statutes"):
    """Run ingestion pipeline."""
    print("\n" + "=" * 70)
    print(f"STEP 2: Running Ingestion for {source.upper()}")
    print("=" * 70)
    
    from ingestion_1.run_ingestion import run_statutes
    
    if source == "statutes":
        run_statutes()
    elif source == "cases":
        # Case law now uses NEW pipeline (PDF -> Gemini -> Nomic -> Neo4j)
        # Legacy chunking in ingestion_1 is deprecated for cases
        print("[INFO] Case law ingestion is handled by the NEW CaseLawPipeline (PDF-based).")
        print("[INFO] Use --cases-new flag with --model to process case PDFs.")
    else:
        print(f"Unknown source: {source}")

def run_caselaw_new_pipeline(pdf_glob: str = "data/raw/cases/*.pdf"):
    """
    Run the NEW case law pipeline (PDF -> Gemini segmentation -> Nomic embeddings -> Neo4j).
    This bypasses the legacy chunk-based ingestion.
    """
    print("\n" + "=" * 70)
    print("STEP 3B: Processing Case Law PDFs (NEW Pipeline)")
    print("=" * 70)

    from glob import glob
    from config.neo4j_config import NEO4J_DATABASE
    from extraction_2.caselaw_pipeline import CaseLawPipeline
    from modeling_3.caselaw_ingestor import CaseLawIngestor

    pdfs = sorted(glob(pdf_glob))
    print(f"Found {len(pdfs)} case PDFs via: {pdf_glob}")
    if not pdfs:
        print("[WARN] No case PDFs found. Skipping case-law pipeline.")
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[WARN] GOOGLE_API_KEY is not set. Gemini segmentation will fail.")

    # Warn about rate limits and quota
    print(f"\n[INFO] Processing {len(pdfs)} PDFs. Each PDF requires 1 Gemini API call.")
    print(f"[INFO] Free tier limits: ~15 requests/minute, ~1500 requests/day")
    print(f"[INFO] Adding 5s delay between PDFs to respect rate limits.")
    if len(pdfs) > 10:
        print(f"[WARN] Processing {len(pdfs)} PDFs may take a while and could hit rate limits.")
    print(f"[INFO] If you hit quota limits, wait 1 hour and retry, or check: https://ai.dev/rate-limit\n")

    pipeline = CaseLawPipeline(gemini_api_key=api_key)
    ingestor = CaseLawIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)

    try:
        for i, pdf_path in enumerate(pdfs):
            print(f"\n=== Case PDF ({i+1}/{len(pdfs)}): {pdf_path} ===")
            try:
                case_json = pipeline.process_pdf(pdf_path)
                stats = ingestor.ingest_case(case_json)
                print(f"  Segments: {stats.get('segments_created')} | Statute links: {stats.get('statute_links')} | Warnings: {len(stats.get('warnings', []))}")
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                    print(f"  [ERROR] Quota exceeded. Skipping remaining PDFs.")
                    print(f"  [INFO] Check your quota: https://ai.dev/rate-limit")
                    print(f"  [INFO] Wait and retry later, or upgrade your plan.")
                    break  # Stop processing more PDFs
                else:
                    print(f"  [ERROR] Failed to process PDF: {e}")
                    continue  # Try next PDF
            
            # Add delay between PDFs to avoid rate limits (5 seconds for free tier)
            if i < len(pdfs) - 1:  # Don't delay after last PDF
                import time
                delay = 30  # Increased to 5 seconds to respect free tier limits
                print(f"  [INFO] Waiting {delay}s before next PDF (to avoid rate limits)...")
                time.sleep(delay)
    finally:
        ingestor.close()

def run_modeling(doc_type: str = "statute", use_new_caselaw: bool = False, caselaw_pdf_glob: str = "data/raw/cases/*.pdf"):
    """Run graph building: new GraphIngestor for statutes; NEW CaseLawPipeline+CaseLawIngestor for cases if requested."""
    print("\n" + "=" * 70)
    print(f"STEP 3: Building Graph for {doc_type.upper()}")
    print("=" * 70)
    
    if doc_type == "statute":
        # Use new GraphIngestor for statutes
        from modeling_3.graph_ingestor import GraphIngestor
        from extraction_2.statutory_extraction import extract_statute_structure
        from pathlib import Path
        from config.paths import DATA_RAW_STATUTES
        from config.neo4j_config import NEO4J_DATABASE
        
        ingestor = GraphIngestor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
        
        try:
            raw_dir = Path(DATA_RAW_STATUTES)
            statute_files = list(raw_dir.glob("*.txt"))
            
            print(f"\nProcessing {len(statute_files)} statute files...")
            
            total_stats = {
                'nodes_created': 0,
                'nodes_updated': 0,
                'hierarchy_links': 0,
                'citation_links': 0,
                'ghost_nodes_created': 0
            }
            
            for statute_file in statute_files:
                print(f"\n=== Processing: {statute_file.name} ===")
                
                # Read statute text
                with open(statute_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Extract structure using StatutoryExtraction
                json_data = extract_statute_structure(text, source_file=statute_file.name)
                
                # Ingest into Neo4j
                stats = ingestor.ingest_statutes(json_data)
                
                # Accumulate stats
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)
                
                print(f"  Created {stats['nodes_created']} nodes, {stats['hierarchy_links']} hierarchy links")
            
            print(f"\n{'='*70}")
            print("Total Statistics:")
            print(f"  Nodes Created: {total_stats['nodes_created']}")
            print(f"  Nodes Updated: {total_stats['nodes_updated']}")
            print(f"  Hierarchy Links: {total_stats['hierarchy_links']}")
            print(f"  Citation Links: {total_stats['citation_links']}")
            print(f"  Ghost Nodes: {total_stats['ghost_nodes_created']}")
            print(f"{'='*70}")
            
        finally:
            ingestor.close()
    
    elif doc_type == "case":
        # Case law now only uses the new pipeline (PDF -> Gemini -> Nomic -> Neo4j)
        if not use_new_caselaw:
            print("[WARN] Legacy case law path removed. Using NEW pipeline. Pass --cases-new flag.")
        run_caselaw_new_pipeline(pdf_glob=caselaw_pdf_glob)
    else:
        print(f"Unknown doc_type: {doc_type}")


def run_temporal_demo():
    """Run example temporal reconstruction queries."""
    print("\n" + "=" * 70)
    print("STEP 4: Running Temporal Reconstruction Queries")
    print("=" * 70)
    
    neo4j_ops = TemporalNeo4jOps(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    reconstruction = TemporalReconstruction(neo4j_ops)
    
    try:
        # Query 1: Law as of 1980
        print("\n--- Query 1: Bribery Act in 1980 ---")
        try:
            law_1980 = reconstruction.reconstruct_law("Bribery Act", "1980-01-01")
            print(f"Total sections: {law_1980['total_sections']}")
            if law_1980['sections']:
                print("Sample sections:")
                for section in law_1980['sections'][:5]:
                    print(f"  Section {section['section_number']}: v{section['version']}")
        except Exception as e:
            print(f"  [INFO] Query 1 failed or no data: {e}")
        
        # Query 2: Law as of today
        print("\n--- Query 2: Bribery Act Today ---")
        try:
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            law_today = reconstruction.reconstruct_law("Bribery Act", today)
            print(f"Total sections: {law_today['total_sections']}")
            if law_today['sections']:
                print("Sample sections:")
                for section in law_today['sections'][:5]:
                    print(f"  Section {section['section_number']}: v{section['version']}")
        except Exception as e:
            print(f"  [INFO] Query 2 failed or no data: {e}")

    finally:
        neo4j_ops.close()


def run_query(query_text: str):
    """Run a natural language query."""
    print("\n" + "=" * 70)
    print(f"STEP 5: Running Query: '{query_text}'")
    print("=" * 70)
    
    neo4j_ops = TemporalNeo4jOps(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    vector_ops = TemporalVectorOps()
    
    try:
        hybrid_search = TemporalHybridSearch(neo4j_ops, vector_ops)
        router = TemporalQueryRouter(hybrid_search)
        
        results = router.route_query(query_text)
        
        print("\n=== Search Results ===")
        print(f"Found {results['result_count']} results (Query Date: {results['query_date']})")
        
        for i, result in enumerate(results['results'][:5]):
            print(f"\n{i+1}. [Score: {result.get('score', 0):.4f}] {result.get('source_doc_type', 'Unknown')}")
            print(f"   Text: {result.get('text', '')[:200]}...")
            if 'effective_from' in result:
                print(f"   Effective: {result['effective_from']} to {result.get('effective_to', 'Present')}")
                
    finally:
        neo4j_ops.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Apofasi Temporal RAG Pipeline (LRMoo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline for statutes
  python pipeline_manager.py --full --source statutes

  # Full pipeline for case law using NEW pipeline (PDF -> Gemini -> Nomic -> Neo4j)
  python pipeline_manager.py --full --source cases --cases-new
  
  # Run a natural language query
  python pipeline_manager.py --query "What is the penalty for bribery?"
  
  # Just initialize schema
  python pipeline_manager.py --schema-only
  
  # Ingest only
  python pipeline_manager.py --ingest
  
  # Build graph only
  python pipeline_manager.py --model
  
  # Run temporal demo queries
  python pipeline_manager.py --test
        """
    )
    
    # Pipeline steps
    parser.add_argument("--clear-db", action="store_true", help="Clear all Neo4j data before running")
    parser.add_argument("--schema-only", action="store_true", help="Only initialize Neo4j schema")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion pipeline")
    parser.add_argument("--model", action="store_true", help="Run graph modeling (LRMoo)")
    parser.add_argument("--test", action="store_true", help="Run temporal demo queries")
    parser.add_argument("--full", action="store_true", help="Run complete pipeline (schema + ingest + model)")
    parser.add_argument("--query", type=str, help="Run a natural language query")
    
    # Options
    parser.add_argument("--source", choices=["statutes", "cases"], default="statutes", help="Data source to process")
    parser.add_argument("--skip-schema", action="store_true", help="Skip schema initialization")
    parser.add_argument("--cases-new", action="store_true", help="Use NEW CaseLawPipeline + CaseLawIngestor for cases (PDF-based)")
    parser.add_argument("--cases-pdf-glob", type=str, default="data/raw/cases/*.pdf", help="Glob for case PDFs (used with --cases-new)")
    
    args = parser.parse_args()
    
    # If no flags, show help
    if not any([args.schema_only, args.ingest, args.model, args.test, args.full, args.query]):
        parser.print_help()
        return
    
    try:
        if args.clear_db:
            clear_database()

        # Schema initialization
        if args.schema_only:
            setup_schema()
            return
        
        if args.full or (args.ingest and not args.skip_schema):
            setup_schema()
        
        # Ingestion
        if args.full or args.ingest:
            # NEW case pipeline does not use ingestion_1 for cases (it reads PDFs directly)
            if args.source == "cases" and args.cases_new:
                print("\n[INFO] Skipping legacy case ingestion (chunking). Using NEW case pipeline on PDFs.")
            else:
                run_ingestion(source=args.source)
        
        # Modeling
        if args.full or args.model:
            run_modeling(
                doc_type="statute" if args.source == "statutes" else "case",
                use_new_caselaw=args.cases_new,
                caselaw_pdf_glob=args.cases_pdf_glob
            )
        
        # Testing
        if args.test:
            run_temporal_demo()
            
        # Semantic Search
        if args.query:
            run_query(args.query)
        
        print("\n" + "=" * 70)
        print("[OK] Pipeline completed successfully!")
        print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\n[WARN] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FAIL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
