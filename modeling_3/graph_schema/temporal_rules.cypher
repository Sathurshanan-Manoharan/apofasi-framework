// Temporal Query Patterns and Rules
// Common queries for temporal-aware legal retrieval

// ============================================
// TEMPORAL QUERY PATTERNS
// ============================================

// Get all active cases as of a specific date
// Usage: CALL get_active_cases_as_of('2020-01-01')
// Returns: All cases that were active (not overruled) on the given date

// Get all active statutes as of a specific date
// Usage: CALL get_active_statutes_as_of('2020-01-01')
// Returns: All statutes that were active on the given date

// Get statute version effective at a specific date
// Usage: CALL get_statute_version_at('statute_123', '2020-01-01')
// Returns: The version of the statute that was effective on the given date

// Get all cases that overruled a specific case
// Usage: MATCH (c:Case {case_id: 'case_123'})<-[:OVERRULES]-(overruling:Case)
// Returns: All cases that overruled the target case

// Get amendment history of a statute
// Usage: MATCH (s:Statute {statute_id: 'statute_123'})-[:HAS_VERSION]->(v:StatuteVersion)
//        RETURN v ORDER BY v.effective_from
// Returns: All versions of the statute in chronological order

// Get all sections of a statute version
// Usage: MATCH (sv:StatuteVersion {version_id: 'version_123'})-[:CONTAINS]->(sec:Section)
//        RETURN sec ORDER BY sec.section_number
// Returns: All sections in the specified version

// ============================================
// TEMPORAL PRECEDENCE QUERIES
// ============================================

// Find cases that follow a specific case (precedence chain)
// MATCH path = (c:Case {case_id: 'case_123'})<-[:FOLLOWS*]-(following:Case)
// WHERE following.effective_from >= date('2020-01-01')
// RETURN following ORDER BY following.decision_date DESC

// Find all cases that interpret a specific statute section
// MATCH (c:Case)-[r:INTERPRETS]->(sv:StatuteVersion)-[:CONTAINS]->(sec:Section {section_number: '14(1)'})
// WHERE c.effective_from <= date('2020-01-01')
//   AND (c.effective_to IS NULL OR c.effective_to >= date('2020-01-01'))
// RETURN c ORDER BY c.decision_date DESC

// ============================================
// TEMPORAL FILTERING HELPERS
// ============================================

// Check if a case was active at a given date
// (c.effective_from <= query_date) AND (c.effective_to IS NULL OR c.effective_to >= query_date)

// Check if a statute version was effective at a given date
// (sv.effective_from <= query_date) AND (sv.effective_to IS NULL OR sv.effective_to >= query_date)

// Get most recent version of a statute
// MATCH (s:Statute {statute_id: 'statute_123'})-[:HAS_VERSION]->(v:StatuteVersion)
// WHERE v.effective_to IS NULL
// RETURN v ORDER BY v.effective_from DESC LIMIT 1

// ============================================
// TEMPORAL RELATIONSHIP QUERIES
// ============================================

// Find all relationships that were active at a specific date
// For OVERRULES: WHERE r.overruled_date <= query_date
// For AMENDS: WHERE r.amendment_date <= query_date AND r.effective_date <= query_date
// For INTERPRETS: WHERE r.interpretation_date <= query_date

// Get chronological chain of amendments
// MATCH path = (sv1:StatuteVersion)-[:AMENDS*]->(sv2:StatuteVersion)
// WHERE sv1.statute_id = 'statute_123'
// RETURN path ORDER BY sv1.effective_from

// ============================================
// VECTOR SEARCH WITH TEMPORAL FILTERING
// ============================================

// Combine vector similarity with temporal filtering
// 1. Perform vector search to get similar chunks
// 2. Filter chunks by effective_from and effective_to
// 3. Filter out chunks from overruled cases or repealed statutes
// 4. Order by relevance score and recency

// Example pattern:
// CALL db.index.vector.queryNodes('chunk_embeddings', $query_embedding, $top_k)
// YIELD node AS chunk, score
// MATCH (chunk:Chunk)
// WHERE chunk.effective_from <= $query_date
//   AND (chunk.effective_to IS NULL OR chunk.effective_to >= $query_date)
//   AND chunk.source_doc_type = $doc_type
// RETURN chunk, score
// ORDER BY score DESC, chunk.effective_from DESC
// LIMIT $limit
