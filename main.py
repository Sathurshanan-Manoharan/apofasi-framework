"""
Entry Point for Apofasi Temporal RAG System
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os

# Import Apofasi components
from retrieval_4.agentic_planner import AgenticPlanner
from retrieval_4.generator import LegalGenerator
from config.neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for legal query."""
    query: str = Field(..., description="Legal query in natural language. Include temporal references like 'in 2020' or 'before 2023' if needed.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What did Section 24 of Act No. 5 of 2025 say in 2020?"
            }
        }


class QueryResponse(BaseModel):
    """Response model for legal query."""
    answer: str = Field(..., description="Generated legal answer with URN citations")
    lineage: List[str] = Field(..., description="Lineage of truth (verified sources)")
    citations: List[Dict[str, Any]] = Field(..., description="Structured citation information")
    temporal_warnings: List[str] = Field(..., description="Temporal drift warnings")
    metadata: Dict[str, Any] = Field(..., description="Query metadata (date, model, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the verified lineage, Section 24 of Act No. 5 of 2025 [urn:lex:lk:act:5:2025!sec24] was not yet enacted in 2020...",
                "lineage": ["urn:lex:lk:act:5:2025!sec24: Section text..."],
                "citations": [{"urn": "urn:lex:lk:act:5:2025!sec24"}],
                "temporal_warnings": ["urn:lex:lk:act:5:2025!sec24: Law not yet enacted on target date"],
                "metadata": {
                    "target_date": "2020-01-01",
                    "model": "gemini-2.5-flash",
                    "lineage_count": 1
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    services: Dict[str, str]


# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for resource initialization and cleanup.
    
    Startup:
    - Initialize AgenticPlanner with Neo4j connection
    - Initialize LegalGenerator with Gemini
    - Store instances in app.state for request handlers
    
    Shutdown:
    - Close Neo4j connections
    - Clean up resources
    """
    logger.info("Starting Apofasi Temporal RAG Server...")
    
    # Load API keys
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    if not gemini_api_key:
        logger.error("GOOGLE_API_KEY not set!")
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    try:
        # Initialize AgenticPlanner
        logger.info("Initializing AgenticPlanner...")
        planner = AgenticPlanner(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            neo4j_database=NEO4J_DATABASE,
            gemini_api_key=gemini_api_key
        )
        logger.info("AgenticPlanner initialized")
        
        # Initialize LegalGenerator
        logger.info("Initializing LegalGenerator...")
        generator = LegalGenerator(gemini_api_key=gemini_api_key)
        logger.info("LegalGenerator initialized")
        
        # Store in app state
        app.state.planner = planner
        app.state.generator = generator
        
        logger.info("Apofasi Temporal RAG Server ready!")
        
        yield  # Server is running
        
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Apofasi Temporal RAG Server...")
        
        if hasattr(app.state, 'planner'):
            logger.info("Closing AgenticPlanner connections...")
            app.state.planner.close()
            logger.info("AgenticPlanner closed")
        
        logger.info("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Apofasi Temporal RAG API",
    description="Temporal legal research system with deterministic retrieval and answer generation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    """
    services = {
        "planner": "initialized" if hasattr(app.state, 'planner') else "not_initialized",
        "generator": "initialized" if hasattr(app.state, 'generator') else "not_initialized"
    }
    
    return HealthResponse(
        status="healthy" if all(s == "initialized" for s in services.values()) else "degraded",
        services=services
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_legal_system(request: QueryRequest):
    """
    Query the temporal legal research system.
    
    Args:
        request: QueryRequest with query and optional target_date
    
    Returns:
        QueryResponse with answer, lineage, citations, and warnings
    
    Raises:
        HTTPException 503: If rate limit is exceeded
        HTTPException 500: For other errors
    """
    try:
        logger.info(f"Received query: {request.query[:80]}...")
        
        # Step 1: Retrieve using AgenticPlanner (will extract temporal context from query)
        logger.info("Calling AgenticPlanner.retrieve()...")
        retrieval_result = app.state.planner.retrieve(
            query=request.query,
            target_date=None,  # Let planner extract from query
            top_k=5
        )
        logger.info("Retrieval complete")
        
        # Step 2: Generate answer using LegalGenerator
        logger.info("Calling LegalGenerator.generate()...")
        generation_result = app.state.generator.generate(
            retrieval_result=retrieval_result,
            query=request.query
        )
        logger.info("Generation complete")
        
        # Step 3: Build response
        # Map generator citations (display_title, raw_urn) to API response format
        api_citations = []
        for source in generation_result.get('sources', []):
            if isinstance(source, dict):
                api_citations.append({
                    "urn": source.get('raw_urn'),
                    "display_title": source.get('display_title')
                })
            else:
                # Fallback for strings (shouldn't happen with new generator)
                api_citations.append({"urn": str(source)})

        response = QueryResponse(
            answer=generation_result['answer'],
            lineage=retrieval_result.get('lineage_of_truth', []),
            citations=api_citations,
            temporal_warnings=generation_result.get('temporal_warnings', []),
            metadata={
                "target_date": generation_result.get('metadata', {}).get('target_date'),
                "model": generation_result.get('metadata', {}).get('model', 'unknown'),
                "lineage_count": generation_result.get('metadata', {}).get('lineage_count', 0),
                "temporal_drift_count": generation_result.get('metadata', {}).get('temporal_drift_count', 0)
            }
        )
        
        logger.info("Sending response")
        return response
    
    except Exception as e:
        # Check for rate limit errors (429)
        error_str = str(e).lower()
        if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
            logger.warning(f"Rate limit exceeded: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Temporal Reasoning Engine is busy. Please try again in 1 minute."
            )
        
        # Other errors
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Apofasi Temporal RAG Server on port 8000...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
