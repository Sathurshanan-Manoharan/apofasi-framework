from typing import Dict, List
from modeling_3.neo4j_ops import TemporalNeo4jOps
from config.model_config import VECTOR_DIMENSION

# Index and constraint names
WORK_ID_CONSTRAINT = "work_id_unique"
EXPRESSION_ID_CONSTRAINT = "expression_id_unique"
EVENT_ID_CONSTRAINT = "event_id_unique"
CHUNK_ID_CONSTRAINT = "chunk_id_unique"

VECTOR_INDEX_NAME = "chunk_embeddings"


class LRMooSchemaInitializer:
    """Initialize LRMoo-inspired graph schema."""
    
    def __init__(self, neo4j_ops: TemporalNeo4jOps):
        self.neo4j_ops = neo4j_ops
    
    def _execute_query(self, query: str, description: str):
        """Execute a schema query with error handling."""
        try:
            self.neo4j_ops._execute_write(query)
            print(f"[OK] {description}")
        except Exception as e:
            print(f"[WARN] {description} - {str(e)}")
    
    def create_constraints(self):
        """Create uniqueness constraints."""
        print("\n=== Creating Constraints ===")
        
        constraints = [
            (f"CREATE CONSTRAINT {WORK_ID_CONSTRAINT} IF NOT EXISTS FOR (w:Work) REQUIRE w.work_id IS UNIQUE",
             "Work ID uniqueness constraint"),
            
            (f"CREATE CONSTRAINT {EXPRESSION_ID_CONSTRAINT} IF NOT EXISTS FOR (e:Expression) REQUIRE e.expression_id IS UNIQUE",
             "Expression ID uniqueness constraint"),
            
            (f"CREATE CONSTRAINT {EVENT_ID_CONSTRAINT} IF NOT EXISTS FOR (ev:Event) REQUIRE ev.event_id IS UNIQUE",
             "Event ID uniqueness constraint"),
            
            (f"CREATE CONSTRAINT {CHUNK_ID_CONSTRAINT} IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
             "Chunk ID uniqueness constraint"),
            
            ("CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE",
             "Case ID uniqueness constraint"),
        ]
        
        for query, description in constraints:
            self._execute_query(query, description)
    
    def create_indexes(self):
        """Create indexes for efficient querying."""
        print("\n=== Creating Indexes ===")
        
        indexes = [
            # Work indexes
            ("CREATE INDEX work_title_idx IF NOT EXISTS FOR (w:Work) ON (w.title)",
             "Work title index"),
            
            ("CREATE INDEX work_jurisdiction_idx IF NOT EXISTS FOR (w:Work) ON (w.jurisdiction)",
             "Work jurisdiction index"),
            
            # Expression indexes
            ("CREATE INDEX expr_work_id_idx IF NOT EXISTS FOR (e:Expression) ON (e.work_id)",
             "Expression work_id index"),
            
            ("CREATE INDEX expr_section_idx IF NOT EXISTS FOR (e:Expression) ON (e.section_number)",
             "Expression section_number index"),
            
            ("CREATE INDEX expr_version_idx IF NOT EXISTS FOR (e:Expression) ON (e.version)",
             "Expression version index"),
            
            ("CREATE INDEX expr_start_date_idx IF NOT EXISTS FOR (e:Expression) ON (e.start_date)",
             "Expression start_date index"),
            
            ("CREATE INDEX expr_end_date_idx IF NOT EXISTS FOR (e:Expression) ON (e.end_date)",
             "Expression end_date index"),
            
            ("CREATE INDEX expr_status_idx IF NOT EXISTS FOR (e:Expression) ON (e.status)",
             "Expression status index"),
            
            # Event indexes
            ("CREATE INDEX event_date_idx IF NOT EXISTS FOR (ev:Event) ON (ev.event_date)",
             "Event date index"),
            
            ("CREATE INDEX event_type_idx IF NOT EXISTS FOR (ev:Event) ON (ev.event_type)",
             "Event type index"),
            
            ("CREATE INDEX event_year_idx IF NOT EXISTS FOR (ev:Event) ON (ev.act_year)",
             "Event year index"),
            
            # Chunk indexes
            ("CREATE INDEX chunk_expr_id_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.expression_id)",
             "Chunk expression_id index"),
            
            # Case indexes
            ("CREATE INDEX case_decision_date_idx IF NOT EXISTS FOR (c:Case) ON (c.decision_date)",
             "Case decision_date index"),
            
            ("CREATE INDEX case_status_idx IF NOT EXISTS FOR (c:Case) ON (c.status)",
             "Case status index"),
        ]
        
        for query, description in indexes:
            self._execute_query(query, description)
    
    def create_vector_index(self):
        """Create vector index for chunk embeddings."""
        print("\n=== Creating Vector Index ===")
        
        # Check Neo4j version
        try:
            version_query = "CALL dbms.components() YIELD versions RETURN versions[0] AS version"
            result = self.neo4j_ops._execute_read(version_query)
            if result:
                version = result[0]['version']
                print(f"  Neo4j version: {version}")
        except Exception as e:
            print(f"[WARN] Could not determine Neo4j version: {e}")
        
        # Create vector index
        query = f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (ch:Chunk)
        ON ch.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        
        self._execute_query(query, f"Vector index '{VECTOR_INDEX_NAME}' ({VECTOR_DIMENSION} dimensions)")
    
    def verify_schema(self):
        """Verify schema creation."""
        print("\n=== Verifying Schema ===")
        
        # Count constraints
        constraints_query = "SHOW CONSTRAINTS"
        constraints = self.neo4j_ops._execute_read(constraints_query)
        print(f"  Constraints: {len(constraints)} created")
        
        # Count indexes
        indexes_query = "SHOW INDEXES"
        indexes = self.neo4j_ops._execute_read(indexes_query)
        print(f"  Indexes: {len(indexes)} created")
        
        # Verify vector index
        try:
            vector_check_query = f"""
            SHOW INDEXES
            YIELD name, type
            WHERE name = '{VECTOR_INDEX_NAME}'
            RETURN name, type
            """
            result = self.neo4j_ops._execute_read(vector_check_query)
            if result:
                print(f"  [OK] Vector index '{VECTOR_INDEX_NAME}' exists")
            else:
                print(f"  [WARN] Vector index '{VECTOR_INDEX_NAME}' not found")
        except Exception as e:
            print(f"[WARN] Verification failed: {e}")
    
    def initialize_all(self):
        """Initialize complete schema."""
        print("=" * 60)
        print("Initializing LRMoo-Inspired Schema for Apofasi")
        print("=" * 60)
        
        self.create_constraints()
        self.create_indexes()
        self.create_vector_index()
        self.verify_schema()
        
        print("\n" + "=" * 60)
        print("Schema initialization complete!")
        print("=" * 60)


if __name__ == "__main__":
    from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    neo4j_ops = TemporalNeo4jOps(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    initializer = LRMooSchemaInitializer(neo4j_ops)
    initializer.initialize_all()
    
    neo4j_ops.close()
    print("Connection closed.")
