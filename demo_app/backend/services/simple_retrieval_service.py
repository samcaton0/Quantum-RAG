"""
Simple Retrieval Service - Direct approach like experiments.
No ChromaDB - loads data directly from data/ directory.
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add core to path
CORE_PATH = Path(__file__).parent.parent.parent.parent / "core"
sys.path.insert(0, str(CORE_PATH.parent))

from core.retrieval import NaiveRetrieval, MMRRetrieval, QUBORetrieval
from core.utils import (
    load_wikipedia_dataset,
    filter_chunks_by_prompt,
    get_prompt_embedding,
    compute_aspect_recall,
    compute_intra_list_similarity,
)
from core.embedding import EmbeddingGenerator

from ..models.schemas import (
    RetrievalResult,
    RetrievalMetrics,
    MethodResult,
)
from .llm_service import get_llm_service


class SimpleRetrievalService:
    """
    Simplified retrieval service that works directly with data files.
    """

    def __init__(self):
        """Initialize the retrieval service."""
        self.wikipedia_chunks = None
        self.wikipedia_embeddings = None
        self.embedder = None
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._prompts = []  # Cache of available prompts
        self._load_wikipedia()

    def _load_wikipedia(self):
        """Load Wikipedia dataset on initialization."""
        data_path = Path(__file__).parent.parent.parent.parent / "data" / "wikipedia"
        if data_path.exists():
            print(f"Loading Wikipedia dataset from {data_path}...")
            self.wikipedia_chunks, self.wikipedia_embeddings = load_wikipedia_dataset(str(data_path))
            print(f"Loaded {len(self.wikipedia_chunks)} chunks")

            # Extract and cache all prompts
            self._prompts = [
                {
                    'prompt_id': c['prompt_id'],
                    'text': c['text'],
                    'article_title': c['article_title']
                }
                for c in self.wikipedia_chunks
                if c.get('chunk_type') == 'prompt'
            ]
            print(f"Found {len(self._prompts)} prompts")
        else:
            print(f"WARNING: Wikipedia data not found at {data_path}")

    def get_available_prompts(self) -> List[Dict[str, str]]:
        """Get list of available prompts."""
        return self._prompts

    def _get_embedder(self) -> EmbeddingGenerator:
        """Lazy load embedder."""
        if self.embedder is None:
            self.embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        return self.embedder

    def _find_best_prompt(self, query: str, query_embedding: np.ndarray) -> Optional[str]:
        """
        Find the best matching prompt for a query using semantic similarity.
        """
        if not self.wikipedia_chunks:
            return None

        # Get all prompts with their embeddings
        prompts = [
            c for c in self.wikipedia_chunks
            if c.get('chunk_type') == 'prompt'
        ]

        if not prompts:
            return None

        # Find the most similar prompt
        best_prompt_id = None
        best_similarity = -1.0

        for prompt in prompts:
            prompt_embedding = self.wikipedia_embeddings.get(prompt['chunk_id'])
            if prompt_embedding is None:
                continue

            # Compute cosine similarity
            similarity = float(np.dot(
                query_embedding / np.linalg.norm(query_embedding),
                prompt_embedding / np.linalg.norm(prompt_embedding)
            ))

            if similarity > best_similarity:
                best_similarity = similarity
                best_prompt_id = prompt['prompt_id']

        return best_prompt_id

    def _run_single_method(
        self,
        method: str,
        query: str,
        query_embedding: np.ndarray,
        candidates: List[Dict[str, Any]],
        gold_aspects: set,
        k: int,
        alpha: float = 0.04,
        beta: float = 0.8,
        penalty: float = 10.0,
        lambda_param: float = 0.85,
        is_wikipedia: bool = True,
        include_llm: bool = True,
    ) -> Tuple[str, MethodResult]:
        """Run a single retrieval method."""
        start_time = time.perf_counter()

        # Create strategy
        if method == "topk":
            strategy = NaiveRetrieval()
        elif method == "mmr":
            strategy = MMRRetrieval(lambda_param=lambda_param)
        elif method == "qubo":
            strategy = QUBORetrieval(alpha=alpha, beta=beta, penalty=penalty, solver='gurobi')
        else:
            raise ValueError(f"Unknown method: {method}")

        # For Wikipedia, need to prepare candidates from raw chunks
        # For medical/legal, candidates are already prepared
        if is_wikipedia:
            candidate_results = []
            for cand in candidates:
                chunk_id = cand['chunk_id']
                embedding = self.wikipedia_embeddings.get(chunk_id)
                if embedding is None:
                    continue

                # Compute similarity score
                score = float(np.dot(
                    query_embedding / np.linalg.norm(query_embedding),
                    embedding / np.linalg.norm(embedding)
                ))

                candidate_results.append({
                    'id': chunk_id,
                    'text': cand['text'],
                    'embedding': embedding,
                    'score': score,
                    'metadata': cand
                })

            # Sort by score
            candidate_results.sort(key=lambda x: x['score'], reverse=True)
        else:
            # Candidates already prepared (medical/legal)
            candidate_results = candidates

        # Run retrieval
        results = strategy.retrieve(query_embedding, candidate_results, k=k)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract metadata for results
        retrieval_results = []
        for r in results:
            retrieval_results.append(RetrievalResult(
                rank=r.rank,
                score=r.score,
                text=r.chunk.text,
                source=r.chunk.metadata.get('article_title', 'unknown'),
                chunk_id=r.chunk.id,
                aspect_id=r.chunk.metadata.get('aspect_id'),
                aspect_name=r.chunk.metadata.get('aspect_name'),
                prompt_id=r.chunk.metadata.get('prompt_id'),
            ))

        # Compute metrics
        # 1. Intra-list similarity (diversity)
        results_with_emb = [
            {'embedding': self.wikipedia_embeddings.get(r.chunk.id)}
            for r in results
            if self.wikipedia_embeddings.get(r.chunk.id) is not None
        ]

        try:
            ils = compute_intra_list_similarity(results_with_emb)
        except Exception:
            ils = 0.0

        # 2. Average relevance
        avg_relevance = np.mean([r.score for r in results]) if results else 0.0

        # 3. Aspect recall
        retrieved_meta = [r.chunk.metadata for r in results]
        aspect_recall_pct, aspects_found = compute_aspect_recall(retrieved_meta, gold_aspects)
        total_aspects = len(gold_aspects) if gold_aspects else 5

        # 4. Cluster coverage (count unique aspects)
        unique_aspects = set()
        for r in results:
            aspect_id = r.chunk.metadata.get('aspect_id', -1)
            if aspect_id >= 0:
                unique_aspects.add(aspect_id)

        metrics = RetrievalMetrics(
            latency_ms=latency_ms,
            intra_list_similarity=ils,
            cluster_coverage=len(unique_aspects),
            total_clusters=total_aspects,
            avg_relevance=float(avg_relevance),
            aspect_recall=aspect_recall_pct,
            aspects_found=aspects_found,
            total_aspects=total_aspects,
        )

        # Generate LLM response if requested
        llm_response = None
        if include_llm:
            llm_service = get_llm_service()
            if llm_service.available:
                # Prepare chunks for LLM
                chunks_for_llm = [
                    {
                        'text': res.text,
                        'source': res.source,
                        'aspect_name': res.aspect_name
                    }
                    for res in retrieval_results
                ]
                llm_response = llm_service.generate_response(query, chunks_for_llm, method)

        return method, MethodResult(
            method=method,
            results=retrieval_results,
            metrics=metrics,
            llm_response=llm_response,
        )

    async def compare_methods(
        self,
        query: str,
        dataset: str = "wikipedia",
        k: int = 5,
        include_llm: bool = False,
        alpha: float = 0.04,
        beta: float = 0.8,
        penalty: float = 10.0,
        lambda_param: float = 0.85,
        solver_preset: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Compare all three retrieval methods on Wikipedia dataset.
        Query should be one of the exact prompt texts.
        """
        if dataset != "wikipedia":
            raise ValueError(f"Only 'wikipedia' dataset is supported, got: {dataset}")

        return await self._compare_wikipedia(query, k, alpha, beta, penalty, lambda_param, include_llm)

    async def _compare_wikipedia(
        self,
        query: str,
        k: int,
        alpha: float,
        beta: float,
        penalty: float,
        lambda_param: float,
        include_llm: bool,
    ) -> Dict[str, Any]:
        """Compare methods on Wikipedia dataset (prompt-based)."""
        if not self.wikipedia_chunks:
            raise RuntimeError("Wikipedia dataset not loaded")

        # Get the embedder and embed the query
        embedder = self._get_embedder()
        query_embedding = embedder.embed_query(query)

        # Find the prompt that matches this query exactly (or by similarity)
        # First try exact match
        prompt_id = None
        for prompt in self._prompts:
            if prompt['text'] == query:
                prompt_id = prompt['prompt_id']
                print(f"Using exact prompt match: {prompt['article_title']}")
                break

        # If no exact match, use similarity
        if not prompt_id:
            prompt_id = self._find_best_prompt(query, query_embedding)
            if not prompt_id:
                raise RuntimeError("No prompts found in dataset")

            prompt_text = next(
                (c['text'] for c in self.wikipedia_chunks
                 if c.get('prompt_id') == prompt_id and c.get('chunk_type') == 'prompt'),
                None
            )
            print(f"Query: '{query[:100]}...'")
            print(f"Matched prompt: '{prompt_text[:150]}...'")

        # Filter chunks for this prompt at redundancy_level=4
        candidates, gold_aspects, _, _, _ = filter_chunks_by_prompt(
            self.wikipedia_chunks,
            prompt_id,
            redundancy_level=4
        )

        print(f"Candidates: {len(candidates)}, Gold aspects: {gold_aspects}")

        if len(candidates) < k:
            raise RuntimeError(f"Not enough candidates ({len(candidates)}) for k={k}")

        # Run all methods in parallel
        loop = asyncio.get_event_loop()
        tasks = []
        for method in ["topk", "mmr", "qubo"]:
            task = loop.run_in_executor(
                self._executor,
                self._run_single_method,
                method,
                query,
                query_embedding,
                candidates,
                gold_aspects,
                k,
                alpha,
                beta,
                penalty,
                lambda_param,
                True,  # is_wikipedia=True
                include_llm,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Convert to dict
        method_results = {method: result for method, result in results}

        return {
            "query": query,
            "dataset": "wikipedia",
            "topk": method_results["topk"],
            "mmr": method_results["mmr"],
            "qubo": method_results["qubo"],
            "umap_points": [],
            "query_point": None,
        }


# Global service instance
_service_instance: Optional[SimpleRetrievalService] = None


def get_simple_retrieval_service() -> SimpleRetrievalService:
    """Get or create the global retrieval service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SimpleRetrievalService()
    return _service_instance
