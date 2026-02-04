
from neo4j import GraphDatabase
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from config.model_config import VECTOR_INDEX_NAME, VECTOR_DIMENSION


class SchemaInitializer:
    """Initialize Neo4j schema."""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        """Close connection."""
        self.driver.close()
    
    def _execute_query(self, query: str, description: str):
        """Execute a query and print status."""
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query)
            print(f"[OK] {description}")
        except Exception as e:
            print(f"[WARN] {description} - {str(e)}")
    
    def create_constraints(self):
        """Create uniqueness constraints."""
        print("\n=== Creating Constraints ===")
        
        constraints = [
            ("CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:Statute) REQUIRE s.statute_id IS UNIQUE",
             "Statute ID uniqueness constraint"),
            ("CREATE CONSTRAINT version_id_unique IF NOT EXISTS FOR (v:StatuteVersion) REQUIRE v.version_id IS UNIQUE",
             "StatuteVersion ID uniqueness constraint"),
            ("CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (sec:Section) REQUIRE sec.section_id IS UNIQUE",
             "Section ID uniqueness constraint"),
            ("CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE",
             "Case ID uniqueness constraint"),
            ("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
             "Chunk ID uniqueness constraint"),
        ]
        
        for query, description in constraints:
            self._execute_query(query, description)
    
    def create_indexes(self):
        """Create property indexes."""
        print("\n=== Creating Indexes ===")
        
        indexes = [
            # Temporal indexes
            ("CREATE INDEX statute_dates IF NOT EXISTS FOR (s:Statute) ON (s.year, s.repealed_date)",
             "Statute temporal index"),
            ("CREATE INDEX version_dates IF NOT EXISTS FOR (v:StatuteVersion) ON (v.effective_from, v.effective_to)",
             "StatuteVersion temporal index"),
            ("CREATE INDEX case_dates IF NOT EXISTS FOR (c:Case) ON (c.decision_date, c.overruled_date)",
             "Case temporal index"),
            ("CREATE INDEX chunk_dates IF NOT EXISTS FOR (ch:Chunk) ON (ch.effective_from, ch.effective_to)",
             "Chunk temporal index"),
            
            # Text search indexes
            ("CREATE INDEX statute_name IF NOT EXISTS FOR (s:Statute) ON (s.name)",
             "Statute name index"),
            ("CREATE INDEX case_name IF NOT EXISTS FOR (c:Case) ON (c.name)",
             "Case name index"),
            ("CREATE INDEX section_number IF NOT EXISTS FOR (sec:Section) ON (sec.section_number)",
             "Section number index"),
            
            # Status indexes
            ("CREATE INDEX statute_status IF NOT EXISTS FOR (s:Statute) ON (s.status)",
             "Statute status index"),
            ("CREATE INDEX case_status IF NOT EXISTS FOR (c:Case) ON (c.status)",
             "Case status index"),
            ("CREATE INDEX chunk_status IF NOT EXISTS FOR (ch:Chunk) ON (ch.status)",
             "Chunk status index"),
        ]
        
        for query, description in indexes:
            self._execute_query(query, description)
    
    def create_vector_index(self):
        """Create vector index for embeddings."""
        print("\n=== Creating Vector Index ===")
        
        # Check Neo4j version first
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS version")
                version = result.single()['version']
                print(f"  Neo4j version: {version}")
                
                # Extract major.minor version
                major_minor = '.'.join(version.split('.')[:2])
                version_float = float(major_minor)
                
                if version_float < 5.11:
                    print(f"[WARN] Vector indexes require Neo4j 5.11+, you have {version}")
                    print("  Skipping vector index creation")
                    return
        except Exception as e:
            print(f"[WARN] Could not determine Neo4j version: {e}")
            print("  Attempting vector index creation anyway...")
        
        # Create vector index
        query = f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (ch:Chunk) ON (ch.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {VECTOR_DIMENSION},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        
        self._execute_query(query, f"Vector index '{VECTOR_INDEX_NAME}' ({VECTOR_DIMENSION} dimensions)")
    
    def verify_schema(self):
        """Verify schema was created successfully."""
        print("\n=== Verifying Schema ===")
        
        try:
            with self.driver.session(database=self.database) as session:
                # Check constraints
                result = session.run("SHOW CONSTRAINTS")
                constraints = [record['name'] for record in result]
                print(f"  Constraints: {len(constraints)} created")
                
                # Check indexes
                result = session.run("SHOW INDEXES")
                indexes = [record['name'] for record in result]
                print(f"  Indexes: {len(indexes)} created")
                
                # Check if vector index exists
                vector_index_exists = any(VECTOR_INDEX_NAME in idx for idx in indexes)
                if vector_index_exists:
                    print(f"  [OK] Vector index '{VECTOR_INDEX_NAME}' exists")
                else:
                    print(f"  [WARN] Vector index '{VECTOR_INDEX_NAME}' not found")
        
        except Exception as e:
            print(f"[WARN] Verification failed: {e}")
    
    def initialize_all(self):
        """Initialize complete schema."""
        print("=" * 60)
        print("Initializing Neo4j Schema for Apofasi")
        print("=" * 60)
        
        self.create_constraints()
        self.create_indexes()
        self.create_vector_index()
        self.verify_schema()
        
        print("\n" + "=" * 60)
        print("Schema initialization complete!")
        print("=" * 60)


def main():
    """Main entry point."""
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    initializer = SchemaInitializer(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    
    try:
        initializer.initialize_all()
    finally:
        initializer.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
