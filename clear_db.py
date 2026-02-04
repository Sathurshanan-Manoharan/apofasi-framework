"""Clear all data from Neo4j database (keeps schema)."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from modeling_3.neo4j_ops import TemporalNeo4jOps
from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

print("Connecting to Neo4j...")
neo4j_ops = TemporalNeo4jOps(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

print("Clearing all nodes and relationships...")
# Use execute_write properly
query = "MATCH (n) DETACH DELETE n"
neo4j_ops._execute_write(query)

print("Done! Database cleared (schema preserved).")
neo4j_ops.close()
