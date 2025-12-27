# Quantum-RAG: Diversity-Aware Retrieval using QUBO Optimization

RAG system that uses Quadratic Unconstrained Binary Optimization (QUBO) to retrieve diverse, relevant documents — outperforming Top-K and MMR baselines on a 171-topic Wikipedia benchmark.

> **Team project.** My contributions: QUBO solver + ORBIT quantum-inspired backend, experimental validation pipeline, and the full demo app (FastAPI + Next.js).

## Setup

**Prerequisites:** Python 3.9+, Node.js 18+, Gurobi ([free academic license](https://www.gurobi.com/academia/))

```bash
git clone <repository-url> && cd Quantum-RAG-Integrated
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optionally pre-build the ChromaDB index for faster queries:
```bash
python data/wikipedia/create_vector_db.py
```

## Running the Demo

```bash
# Backend (http://localhost:8000)
cd demo_app/backend
echo "GEMINI_API_KEY=your_key" > .env
python main_simple.py

# Frontend (http://localhost:3000)
cd demo_app/frontend && npm install && npm run dev
```

## Running Experiments

```bash
python experiments/exp_0_energy_validation.py       # QUBO energy validation
python experiments/exp_1_poisoned_stress_test.py    # Robustness under redundancy
python experiments/exp_2_k_equivalence_analysis.py  # k-equivalence analysis
```

Results (`.png` + `.json`) are saved to `results/`.

## How It Works

Three retrieval methods run in parallel on a 5,600-chunk Wikipedia corpus (BGE-large embeddings, 1024-dim):

| Method | Description |
|--------|-------------|
| **Top-K** | Baseline — top k by cosine similarity |
| **MMR** | Maximal Marginal Relevance (λ=0.85) |
| **QUBO** | Joint relevance + diversity optimization (α=0.04, β=0.8), solved via Gurobi/ORBIT |

## Key Results

- **Diversity**: 15–20% lower intra-list similarity vs Top-K
- **Aspect Coverage**: 40–60% more unique aspects covered
- **Robustness**: Maintains quality under 5× document redundancy
- **Stability**: Quality stable as k increases (unlike MMR)

## Environment Variables

```bash
GEMINI_API_KEY=...          # demo_app/backend/.env
NEXT_PUBLIC_API_URL=http://localhost:8000  # demo_app/frontend/.env.local
```
