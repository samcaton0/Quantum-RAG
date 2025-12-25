# Quantum-RAG: Diversity-Aware Retrieval using QUBO Optimization

A novel approach to Retrieval-Augmented Generation (RAG) that uses Quadratic Unconstrained Binary Optimization (QUBO) to retrieve diverse, relevant documents. This project demonstrates superior performance over traditional Top-K and MMR retrieval methods.

## Overview

This repository contains:

- **Core retrieval algorithms**: Top-K, MMR, and QUBO-based retrieval
- **Interactive demo application**: Side-by-side comparison of all three methods
- **Experimental validation**: Comprehensive benchmarks demonstrating QUBO's advantages
- **Wikipedia dataset**: 171 diverse topics with aspect-based evaluation

## Project Structure

```
Quantum-RAG-Integrated/
├── core/                          # Core retrieval algorithms and utilities
│   ├── retrieval.py              # Retrieval strategies (Top-K, MMR, QUBO)
│   ├── embedding.py              # Embedding generation
│   ├── storage.py                # ChromaDB vector store
│   ├── qubo_solver.py            # QUBO formulation and solving
│   ├── utils.py                  # Utilities and metrics
│   └── generation.py             # LLM response generation
│
├── data/                          # Datasets
│   └── wikipedia/                # Wikipedia dataset (171 topics)
│       ├── checkpoints/          # Pre-computed chunks and embeddings
│       │   ├── chunks.jsonl      # Document chunks with metadata
│       │   └── embeddings.npz    # Pre-computed embeddings (35MB)
│       ├── chroma_db/            # ChromaDB vector database (optional)
│       └── create_vector_db.py   # Script to create ChromaDB from checkpoints
│
├── experiments/                   # Experimental validation scripts
│   ├── exp_0_energy_validation.py         # Validates QUBO energy function
│   ├── exp_1_poisoned_stress_test.py      # Tests robustness to redundancy
│   └── exp_2_k_equivalence_analysis.py    # Analyzes k-equivalence property
│
├── results/                       # Experiment results and visualizations
│   ├── exp_0_energy_validation.png
│   ├── exp_1_poisoned_stress_test.png
│   ├── exp_2_k_equivalence_analysis.png
│   └── *.json                    # Raw experimental data
│
├── demo_app/                      # Interactive demo application
│   ├── backend/                  # FastAPI backend
│   │   ├── main_simple.py        # API server entry point
│   │   ├── api/                  # API endpoints
│   │   ├── services/             # Business logic
│   │   │   ├── simple_retrieval_service.py
│   │   │   └── llm_service.py
│   │   └── models/               # Pydantic schemas
│   │
│   └── frontend/                 # Next.js frontend
│       ├── src/
│       │   ├── app/              # Next.js pages
│       │   ├── components/       # React components
│       │   └── lib/              # API client and utilities
│       └── package.json
│
└── docs/                          # Documentation

```

## Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# Node.js 18+
node --version

# Install Gurobi (for QUBO solver)
# Get free academic license: https://www.gurobi.com/academia/
```

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd Quantum-RAG-Integrated

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Create Wikipedia Vector Database (Optional but Recommended)

For faster queries, create a ChromaDB vector database from the pre-computed embeddings:

```bash
cd data/wikipedia
python create_vector_db.py
```

This creates `data/wikipedia/chroma_db/` with ~5,600 indexed chunks for fast similarity search.

### 3. Run the Demo Application

#### Start Backend

```bash
cd demo_app/backend

# Create .env file with your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Start the API server
python main_simple.py
```

Backend runs at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

#### Start Frontend

```bash
cd demo_app/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Frontend runs at `http://localhost:3000`.

### 4. Run Experiments

Each experiment can be run independently to validate different aspects of the QUBO approach:

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Experiment 0: Energy Function Validation
# Validates that QUBO energy correlates with retrieval quality
python experiments/exp_0_energy_validation.py

# Experiment 1: Poisoned Stress Test
# Tests robustness against high redundancy (5 redundant copies per aspect)
python experiments/exp_1_poisoned_stress_test.py

# Experiment 2: K-Equivalence Analysis
# Demonstrates that QUBO maintains quality as k increases
python experiments/exp_2_k_equivalence_analysis.py
```

Results are saved to `results/` directory with visualizations (.png) and raw data (.json).

## Demo Application Features

- **Split-screen comparison**: View Top-K, MMR, and QUBO results side-by-side
- **Real-time metrics**:
  - Latency (ms)
  - Diversity (Intra-List Similarity)
  - Cluster Coverage
  - Aspect Recall (Wikipedia dataset)
- **Interactive UMAP visualization**: See how each method selects documents in embedding space
- **LLM-powered responses**: Compare answer quality using only retrieved chunks

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 14)                        │
│                    http://localhost:3000                        │
│  Landing Page → Demo Selection → Comparison View                │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/REST
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                            │
│                    http://localhost:8000                        │
│  /api/compare → Runs Top-K, MMR, QUBO in parallel               │
│  /api/prompts → Available Wikipedia dataset queries             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CORE (core/)                                 │
│  retrieval.py │ qubo_solver.py │ embedding.py │ utils.py        │
└─────────────────────────────────────────────────────────────────┘
```

## Dataset

**Wikipedia Knowledge Base**

- 171 diverse topics across multiple domains
- Each article divided into 5 gold aspects
- Controlled redundancy levels (0-5) for testing robustness
- Pre-computed BGE-large embeddings (1024 dimensions)
- Total: ~5,600 chunks (35MB)

## Retrieval Methods

### Top-K (Baseline)

Returns the k documents with highest cosine similarity to the query.

### MMR (State-of-the-Art)

Maximal Marginal Relevance with greedy selection:

- **lambda=0.85**: Balance between relevance (85%) and diversity (15%)

### QUBO (Proposed)

Quadratic optimization with simultaneous relevance and diversity:

- **alpha=0.04**: Diversity weight
- **beta=0.8**: Similarity threshold for redundancy penalty
- **penalty=10**: Cardinality constraint weight
- Solved using Gurobi optimizer

## Key Results

From experiments on Wikipedia dataset:

- **Diversity**: QUBO achieves 15-20% lower intra-list similarity vs Top-K
- **Aspect Coverage**: QUBO retrieves 40-60% more unique aspects
- **Robustness**: QUBO maintains quality under high redundancy (5x copies)
- **K-Equivalence**: QUBO quality stable as k increases (unlike MMR)

## API Endpoints

| Endpoint         | Method | Description                               |
| ---------------- | ------ | ----------------------------------------- |
| `/api/compare` | POST   | Run all three methods and compare results |
| `/api/prompts` | GET    | Get list of available Wikipedia prompts   |
| `/api/health`  | GET    | Health check and system status            |

## Environment Variables

```bash
# Backend (.env in demo_app/backend/)
GEMINI_API_KEY=your_google_gemini_api_key

# Frontend (.env.local in demo_app/frontend/)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Troubleshooting

### Backend won't start

- Verify Gurobi is installed and licensed: `python -c "import gurobipy"`
- Check GEMINI_API_KEY is set in `.env` file
- Ensure Wikipedia data exists: `ls data/wikipedia/checkpoints/`
- Check Python version: `python --version` (3.9+ required)

### Frontend won't start

- Run `npm install` in `demo_app/frontend/`
- Check Node.js version: `node --version` (18+ required)
- Verify backend is running at `http://localhost:8000`

### Experiments failing

- Activate virtual environment: `venv\Scripts\activate`
- Check all dependencies installed: `pip install -r requirements.txt`
- Verify Gurobi license: `python -c "import gurobipy; print('OK')"`
- Ensure Wikipedia data is present in `data/wikipedia/checkpoints/`

### Out of memory errors

- Reduce batch size in experiments
- Use ChromaDB vector database instead of in-memory embeddings
- Close other applications to free RAM

## Performance Tips

1. **Use ChromaDB**: Run `data/wikipedia/create_vector_db.py` for 10-100x faster queries
2. **Limit candidates**: QUBO works well with 50-200 candidates (automatically filtered)
3. **Adjust k**: Start with k=5 for demos, increase to k=10 for comprehensive results
4. **Gurobi settings**: Use "balanced" preset for good speed/quality tradeoff

## Citation

If you use this work, please cite:

```
@article{quantum-rag-2024,
  title={Quantum-Inspired Retrieval-Augmented Generation using QUBO Optimization},
  author={Your Name},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Contact

For questions or issues, please open a GitHub issue or contact vanja.zdravkovic@new.ox.ac.uk.
