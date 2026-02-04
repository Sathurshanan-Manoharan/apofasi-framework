// Temporal-Aware Constraints and Indexes
// Ensures data integrity and query performance

// ============================================
// UNIQUE CONSTRAINTS
// ============================================

// Case constraints
CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE;

// Statute constraints
CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:Statute) REQUIRE s.statute_id IS UNIQUE;
CREATE CONSTRAINT statute_version_unique IF NOT EXISTS FOR (sv:StatuteVersion) REQUIRE (sv.statute_id, sv.version) IS UNIQUE;

// Section constraints
CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (sec:Section) REQUIRE sec.section_id IS UNIQUE;
CREATE CONSTRAINT section_number_unique IF NOT EXISTS FOR (sec:Section) REQUIRE (sec.statute_id, sec.section_number, sec.effective_from) IS UNIQUE;

// Gazette constraints
CREATE CONSTRAINT gazette_id_unique IF NOT EXISTS FOR (g:Gazette) REQUIRE g.gazette_id IS UNIQUE;
CREATE CONSTRAINT gazette_number_unique IF NOT EXISTS FOR (g:Gazette) REQUIRE (g.gazette_number, g.publication_date) IS UNIQUE;

// Chunk constraints
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE;

// ============================================
// TEMPORAL INDEXES
// ============================================

// Indexes on date fields for efficient temporal queries

// Case temporal indexes
CREATE INDEX case_decision_date_idx IF NOT EXISTS FOR (c:Case) ON (c.decision_date);
CREATE INDEX case_effective_from_idx IF NOT EXISTS FOR (c:Case) ON (c.effective_from);
CREATE INDEX case_effective_to_idx IF NOT EXISTS FOR (c:Case) ON (c.effective_to);
CREATE INDEX case_status_idx IF NOT EXISTS FOR (c:Case) ON (c.status);

// Statute temporal indexes
CREATE INDEX statute_enactment_date_idx IF NOT EXISTS FOR (s:Statute) ON (s.enactment_date);
CREATE INDEX statute_effective_date_idx IF NOT EXISTS FOR (s:Statute) ON (s.effective_date);
CREATE INDEX statute_status_idx IF NOT EXISTS FOR (s:Statute) ON (s.status);

// StatuteVersion temporal indexes
CREATE INDEX version_effective_from_idx IF NOT EXISTS FOR (sv:StatuteVersion) ON (sv.effective_from);
CREATE INDEX version_effective_to_idx IF NOT EXISTS FOR (sv:StatuteVersion) ON (sv.effective_to);
CREATE INDEX version_statute_id_idx IF NOT EXISTS FOR (sv:StatuteVersion) ON (sv.statute_id);

// Section temporal indexes
CREATE INDEX section_effective_from_idx IF NOT EXISTS FOR (sec:Section) ON (sec.effective_from);
CREATE INDEX section_effective_to_idx IF NOT EXISTS FOR (sec:Section) ON (sec.effective_to);
CREATE INDEX section_statute_id_idx IF NOT EXISTS FOR (sec:Section) ON (sec.statute_id);

// Gazette temporal indexes
CREATE INDEX gazette_publication_date_idx IF NOT EXISTS FOR (g:Gazette) ON (g.publication_date);
CREATE INDEX gazette_effective_date_idx IF NOT EXISTS FOR (g:Gazette) ON (g.effective_date);

// Chunk temporal indexes
CREATE INDEX chunk_effective_from_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.effective_from);
CREATE INDEX chunk_effective_to_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.effective_to);
CREATE INDEX chunk_source_doc_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.source_doc_id, ch.source_doc_type);

// ============================================
// COMPOSITE INDEXES FOR TEMPORAL QUERIES
// ============================================

// Composite indexes for common temporal query patterns

// Active cases/statutes as of a date
CREATE INDEX case_active_idx IF NOT EXISTS FOR (c:Case) ON (c.status, c.effective_from, c.effective_to);
CREATE INDEX statute_active_idx IF NOT EXISTS FOR (s:Statute) ON (s.status, s.effective_date);

// Version lookup by statute and date range
CREATE INDEX version_lookup_idx IF NOT EXISTS FOR (sv:StatuteVersion) ON (sv.statute_id, sv.effective_from, sv.effective_to);

// Section lookup by statute and date range
CREATE INDEX section_lookup_idx IF NOT EXISTS FOR (sec:Section) ON (sec.statute_id, sec.effective_from, sec.effective_to);

// ============================================
// RELATIONSHIP INDEXES
// ============================================

// Indexes on relationship properties for temporal queries
// Note: Neo4j automatically indexes relationship types, but we can add property indexes

// These will be created programmatically when relationships are added
// Example: CREATE INDEX overrules_date_idx IF NOT EXISTS FOR ()-[r:OVERRULES]-() ON (r.overruled_date);
