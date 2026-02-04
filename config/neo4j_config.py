"""
Neo4j database connection settings.
Configure these values for your Neo4j instance.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Database name (for Neo4j 4.0+)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
