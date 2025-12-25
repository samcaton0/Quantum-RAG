"""
Compare API - Runs all three retrieval methods and returns comparison.
"""

from fastapi import APIRouter, HTTPException

from ..models.schemas import CompareRequest, CompareResponse
from ..services.simple_retrieval_service import get_simple_retrieval_service

router = APIRouter(prefix="/api", tags=["compare"])


@router.post("/compare", response_model=CompareResponse)
async def compare_methods(request: CompareRequest) -> CompareResponse:
    """
    Run all three retrieval methods (Top-K, MMR, QUBO) and compare results.

    This is the main endpoint for the demo - it shows side-by-side comparison
    of industry standard vs Quantum-RAG.
    """
    try:
        retrieval_service = get_simple_retrieval_service()

        # Run comparison with configurable parameters
        results = await retrieval_service.compare_methods(
            query=request.query,
            dataset=request.dataset,
            k=request.k,
            include_llm=request.include_llm,
            alpha=request.alpha,
            beta=request.beta,
            penalty=request.penalty,
            lambda_param=request.lambda_param,
            solver_preset=request.solver_preset,
        )

        return CompareResponse(
            query=results["query"],
            dataset=results["dataset"],
            topk=results["topk"],
            mmr=results["mmr"],
            qubo=results["qubo"],
            umap_points=results.get("umap_points", []),
            query_point=results.get("query_point"),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
