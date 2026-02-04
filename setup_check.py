"""
Setup verification script for Apofasi.
Checks that all dependencies and configuration are correct.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  [FAIL] Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False


def check_dependencies():
    """Check that required packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        ('neo4j', 'Neo4j driver'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('transformers', 'Transformers (for NER)'),
        ('langchain', 'LangChain'),
        ('torch', 'PyTorch'),
        ('fitz', 'PyMuPDF'),
        ('tqdm', 'Progress bars'),
    ]
    
    all_installed = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {description}")
        except ImportError:
            print(f"  [FAIL] {description} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_env_file():
    """Check if .env file exists."""
    print("\nChecking configuration...")
    
    env_path = Path('.env')
    template_path = Path('.env.template')
    
    if env_path.exists():
        print("  [OK] .env file exists")
        
        # Check if it has required variables
        with open(env_path) as f:
            content = f.read()
        
        required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
        missing = []
        for var in required_vars:
            if var not in content:
                missing.append(var)
        
        if missing:
            print(f"  [WARN] Missing variables in .env: {', '.join(missing)}")
            return False
        else:
            print("  [OK] All required variables present")
            return True
    else:
        print("  [FAIL] .env file not found")
        if template_path.exists():
            print("    Run: cp .env.template .env")
            print("    Then edit .env with your Neo4j credentials")
        return False


def check_neo4j_connection():
    """Check Neo4j connection."""
    print("\nChecking Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        from dotenv import load_dotenv
        
        load_dotenv()
        
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', '')
        
        if not password or password == 'your_password_here':
            print("  [WARN] Neo4j password not configured in .env")
            return False
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Try to connect
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        
        driver.close()
        print(f"  [OK] Connected to Neo4j at {uri}")
        return True
    
    except Exception as e:
        print(f"  [FAIL] Cannot connect to Neo4j: {e}")
        print("    Make sure Neo4j is running and credentials are correct")
        return False


def check_data_directories():
    """Check that data directories exist."""
    print("\nChecking data directories...")
    
    directories = [
        'data/raw/statutes',
        'data/raw/cases',
        'data/processed/statutes',
        'data/processed/cases',
    ]
    
    all_exist = True
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            # Count files
            files = list(path.glob('*'))
            print(f"  [OK] {dir_path} ({len(files)} files)")
        else:
            print(f"  [WARN] {dir_path} - does not exist")
            all_exist = False
    
    return all_exist


def check_models():
    """Check if models can be loaded."""
    print("\nChecking models...")
    
    # Check embedding model
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model (this may take a moment)...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print(f"  [OK] Embedding model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    except Exception as e:
        print(f"  [FAIL] Cannot load embedding model: {e}")
        return False
    
    # Check NER model
    try:
        from transformers import pipeline
        print("  Loading NER model (this may take a moment)...")
        ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        print("  [OK] NER model loaded")
    except Exception as e:
        print(f"  [WARN] Cannot load NER model: {e}")
        print("    This is optional - regex fallback will be used")
    
    return True


def main():
    """Run all checks."""
    print("=" * 70)
    print("APOFASI SETUP VERIFICATION")
    print("=" * 70)
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Configuration", check_env_file),
        ("Neo4j connection", check_neo4j_connection),
        ("Data directories", check_data_directories),
        ("Models", check_models),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n  [FAIL] Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n[OK] Setup complete! You're ready to run Apofasi.")
        print("\nNext steps:")
        print("  1. python modeling_3/graph_schema/init_schema.py")
        print("  2. python main.py --full --source statutes")
    else:
        print("\n[WARN] Some checks failed. Please fix the issues above.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
