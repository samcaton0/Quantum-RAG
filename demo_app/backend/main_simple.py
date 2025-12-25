"""
Quantum-RAG Demo API - Simplified version

FastAPI backend for the investor demo application.
Directly loads data from data/ directory (no ChromaDB).
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo_app.backend.api import compare_router
from demo_app.backend.models.schemas import HealthResponse
from demo_app.backend.services.simple_retrieval_service import get_simple_retrieval_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Pre-loads datasets on startup.
    """
    print("Starting Quantum-RAG Demo API (Simple)...")

    # Pre-load retrieval service (loads Wikipedia data)
    service = get_simple_retrieval_service()

    print("API ready!")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Quantum-RAG Demo API",
    description="""
    Backend API for the Quantum-RAG investor demo application.

    ## Features

    - **Compare Methods**: Run Top-K, MMR, and QUBO retrieval side-by-side
    - **Real-time Metrics**: Diversity, relevance, aspect recall

    ## Endpoints

    - `POST /api/compare` - Compare all three methods
    - `GET /api/health` - Health check
    """,
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(compare_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns status of the API and available datasets.
    """
    service = get_simple_retrieval_service()
    available = []
    if service.wikipedia_chunks:
        available.append("wikipedia")

    # Check Gurobi availability
    gurobi_available = True
    try:
        import gurobipy
    except ImportError:
        gurobi_available = False

    return HealthResponse(
        status="healthy",
        orbit_available=gurobi_available,  # Repurposing this field
        datasets_loaded=available,
    )


@app.get("/api/prompts", tags=["prompts"])
async def get_prompts():
    """
    Get list of available prompts from the Wikipedia dataset.
    Users should select from these prompts instead of entering arbitrary queries.
    """
    service = get_simple_retrieval_service()
    prompts = service.get_available_prompts()
    return {"prompts": prompts}


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Quantum-RAG Demo API (Simple)",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
