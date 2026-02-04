// Temporal-Aware Legal Knowledge Graph Ontology
// Defines node types, relationships, and temporal properties

// ============================================
// NODE TYPES WITH TEMPORAL PROPERTIES
// ============================================

// Case nodes - represent legal cases/judgments
CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE;

// Case properties:
// - case_id: unique identifier
// - name: case name (e.g., "Smith v. Jones")
// - court: court name
// - decision_date: DATE when case was decided
// - effective_from: DATE when case became effective (usually same as decision_date)
// - effective_to: DATE when case was overruled (null if still active)
// - status: 'active' or 'overruled'
// - summary: case summary
// - judges: array of judge names

// Statute nodes - represent legal statutes/acts
CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:Statute) REQUIRE s.statute_id IS UNIQUE;

// Statute properties:
// - statute_id: unique identifier
// - title: statute title
// - act_number: act number
// - act_year: year of enactment
// - enactment_date: DATE when statute was enacted
// - effective_date: DATE when statute became effective
// - last_amended: DATE of last amendment
// - version: version string (e.g., "1.0", "2.1")
// - status: 'active', 'amended', or 'repealed'

// StatuteVersion nodes - represent different versions of a statute
CREATE CONSTRAINT version_id_unique IF NOT EXISTS FOR (sv:StatuteVersion) REQUIRE sv.version_id IS UNIQUE;

// StatuteVersion properties:
// - version_id: unique identifier
// - statute_id: reference to parent statute
// - version: version string
// - effective_from: DATE when this version became effective
// - effective_to: DATE when this version was superseded (null if current)
// - amendment_gazette_id: reference to gazette that created this version
// - content_hash: hash of version content for comparison

// Section nodes - represent sections within statutes
CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (sec:Section) REQUIRE sec.section_id IS UNIQUE;

// Section properties:
// - section_id: unique identifier
// - section_number: section number (e.g., "14", "14(1)")
// - parent_section_id: reference to parent section (for subsections)
// - statute_id: reference to parent statute
// - effective_from: DATE when section became effective
// - effective_to: DATE when section was amended/repealed (null if active)
// - content: section text

// Gazette nodes - represent official gazettes
CREATE CONSTRAINT gazette_id_unique IF NOT EXISTS FOR (g:Gazette) REQUIRE g.gazette_id IS UNIQUE;

// Gazette properties:
// - gazette_id: unique identifier
// - gazette_number: gazette number
// - publication_date: DATE when gazette was published
// - effective_date: DATE when gazette became effective

// Chunk nodes - represent text chunks for vector search
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE;

// Chunk properties:
// - chunk_id: unique identifier
// - source_doc_id: reference to source document
// - source_doc_type: 'case', 'statute', or 'gazette'
// - chunk_index: index within document
// - text: chunk text
// - effective_from: DATE when chunk content was effective
// - effective_to: DATE when chunk content was superseded (null if active)
// - section_number: section number if from statute
// - embedding: vector embedding (stored separately in vector index)

// ============================================
// TEMPORAL RELATIONSHIPS
// ============================================

// Case-to-Case relationships
// OVERRULES: Case A overrules Case B
// Properties:
//   - overruled_date: DATE when target case was overruled
//   - overruling_date: DATE when this case overruled the target

// AFFIRMS: Case A affirms Case B
// Properties:
//   - affirmation_date: DATE when affirmation occurred

// FOLLOWS: Case A follows precedent of Case B
// Properties:
//   - precedence_date: DATE when precedence was established

// Case-to-Statute relationships
// INTERPRETS: Case interprets a Statute/Section
// Properties:
//   - interpretation_date: DATE when interpretation occurred
//   - target_section: section number being interpreted

// Statute-to-Statute relationships
// AMENDS: StatuteVersion A amends StatuteVersion B
// Properties:
//   - amendment_date: DATE when amendment was made
//   - effective_date: DATE when amendment became effective

// REPEALS: StatuteVersion A repeals StatuteVersion B
// Properties:
//   - repeal_date: DATE when repeal was made
//   - effective_date: DATE when repeal became effective

// Statute-to-Gazette relationships
// PUBLISHED_IN: StatuteVersion was published in Gazette
// Properties:
//   - publication_date: DATE of publication

// ============================================
// STRUCTURAL RELATIONSHIPS
// ============================================

// HAS_VERSION: Statute has StatuteVersion
// CONTAINS: StatuteVersion contains Section
// PARENT_OF: Section is parent of another Section (for hierarchy)
// CITES: Document cites another document
// Properties:
//   - citation_date: DATE when citation was made

// ============================================
// VECTOR INDEXES
// ============================================

// Create vector index for chunk embeddings
// This will be created via vector_ops.py using Neo4j's vector index capabilities
