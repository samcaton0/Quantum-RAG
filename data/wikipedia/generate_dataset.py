"""
Wikipedia Dataset Generator

Generates aspect-based Wikipedia dataset with controlled redundancy for RAG testing.
Creates: chunks.jsonl, embeddings.npz
"""
import json
import argparse
import random
import uuid
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from core.embedding import EmbeddingGenerator

# Wikipedia API (minimal inline implementation)
try:
    import wikipedia
except ImportError:
    print("Please install: pip install wikipedia-api")
    sys.exit(1)


def fetch_article(title: str) -> Dict | None:
    """Fetch Wikipedia article and extract sections."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {
            'title': page.title,
            'content': page.content,
            'sections': page.sections[:5]  # Limit to 5 aspects
        }
    except:
        return None


def create_chunks_for_article(article: Dict, prompt_id: str, max_redundancy: int) -> List[Dict]:
    """Create prompt, gold base, redundant, and noise chunks for an article."""
    chunks = []

    # Create prompt chunk
    aspect_names = article['sections'][:5]
    prompt_text = f"Provide a comprehensive overview of {article['title']}, covering key aspects such as {', '.join(aspect_names)}."
    chunks.append({
        'chunk_id': str(uuid.uuid4()),
        'text': prompt_text,
        'chunk_type': 'prompt',
        'prompt_id': prompt_id,
        'article_title': article['title'],
        'aspect_id': -1,
        'aspect_name': 'prompt',
        'redundancy_index': -1
    })

    # Create gold base + redundant chunks for each aspect
    paragraphs = article['content'].split('\n\n')
    for aspect_id, aspect_name in enumerate(aspect_names):
        # Find paragraphs mentioning this aspect
        aspect_paragraphs = [p for p in paragraphs if aspect_name.lower() in p.lower() and len(p) > 100]
        if not aspect_paragraphs:
            continue

        # Create base + redundant versions
        for redundancy_idx in range(max_redundancy + 1):
            # Accumulate more text for higher redundancy
            num_paragraphs = min(redundancy_idx + 1, len(aspect_paragraphs))
            combined_text = '\n'.join(aspect_paragraphs[:num_paragraphs])

            chunk_type = 'gold_base' if redundancy_idx == 0 else 'gold_redundant'
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'text': combined_text[:1000],  # Limit length
                'chunk_type': chunk_type,
                'prompt_id': prompt_id,
                'article_title': article['title'],
                'aspect_id': aspect_id,
                'aspect_name': aspect_name,
                'redundancy_index': redundancy_idx
            })

    return chunks


def create_noise_chunks(articles: List[Dict], num_noise: int = 25) -> List[Dict]:
    """Create noise chunks from unrelated articles."""
    noise_chunks = []
    all_paragraphs = []

    for article in articles:
        paragraphs = [p for p in article['content'].split('\n\n') if len(p) > 100]
        all_paragraphs.extend([(p, article['title']) for p in paragraphs])

    # Sample random paragraphs
    random.shuffle(all_paragraphs)
    for i, (text, title) in enumerate(all_paragraphs[:num_noise]):
        noise_chunks.append({
            'chunk_id': str(uuid.uuid4()),
            'text': text[:1000],
            'chunk_type': 'noise',
            'prompt_id': 'global',  # Shared across all prompts
            'article_title': title,
            'aspect_id': -1,
            'aspect_name': 'noise',
            'redundancy_index': -1
        })

    return noise_chunks


def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia aspect dataset')
    parser.add_argument('--num-articles', type=int, default=100, help='Number of prompt articles')
    parser.add_argument('--max-redundancy', type=int, default=5, help='Max redundancy level')
    parser.add_argument('--test', action='store_true', help='Test mode: 5 articles only')
    args = parser.parse_args()

    num_articles = 5 if args.test else args.num_articles
    max_redundancy = args.max_redundancy

    print(f"Generating dataset: {num_articles} articles, redundancy 0-{max_redundancy}")

    # Load article titles
    article_list = script_dir / 'wiki_articles.txt'
    if not article_list.exists():
        print(f"Error: {article_list} not found")
        sys.exit(1)

    with open(article_list) as f:
        titles = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    random.seed(42)
    random.shuffle(titles)

    # Stage 1: Fetch articles
    print("\n=== Fetching Articles ===")
    prompt_articles = []
    noise_articles = []

    for title in tqdm(titles, desc="Fetching"):
        if len(prompt_articles) >= num_articles and len(noise_articles) >= num_articles:
            break

        article = fetch_article(title)
        if article and len(article['sections']) >= 3:
            if len(prompt_articles) < num_articles:
                prompt_articles.append(article)
            elif len(noise_articles) < num_articles:
                noise_articles.append(article)

    print(f"Fetched: {len(prompt_articles)} prompt, {len(noise_articles)} noise articles")

    # Stage 2: Create chunks
    print("\n=== Creating Chunks ===")
    all_chunks = []

    # Create noise pool (shared across prompts)
    noise_chunks = create_noise_chunks(noise_articles, num_noise=25)

    # Create prompt + gold chunks
    for article in tqdm(prompt_articles, desc="Processing"):
        prompt_id = str(uuid.uuid4())
        chunks = create_chunks_for_article(article, prompt_id, max_redundancy)

        # Add noise to each prompt
        for noise_chunk in noise_chunks:
            noise_copy = noise_chunk.copy()
            noise_copy['chunk_id'] = str(uuid.uuid4())
            noise_copy['prompt_id'] = prompt_id
            chunks.append(noise_copy)

        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} total chunks")

    # Stage 3: Generate embeddings
    print("\n=== Generating Embeddings ===")
    embedder = EmbeddingGenerator()
    texts = [c['text'] for c in all_chunks]
    embeddings = embedder.embed(texts)

    # Stage 4: Save to disk
    print("\n=== Saving Dataset ===")
    output_dir = script_dir / 'checkpoints'
    output_dir.mkdir(exist_ok=True)

    # Save chunks as JSONL
    with open(output_dir / 'chunks.jsonl', 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')

    # Save embeddings as NPZ (keyed by chunk_id)
    embedding_dict = {chunk['chunk_id']: emb for chunk, emb in zip(all_chunks, embeddings)}
    np.savez(output_dir / 'embeddings.npz', **embedding_dict)

    # Summary
    chunk_types = {}
    for chunk in all_chunks:
        chunk_types[chunk['chunk_type']] = chunk_types.get(chunk['chunk_type'], 0) + 1

    print("\n=== Dataset Complete ===")
    print(f"Total chunks: {len(all_chunks)}")
    for ctype, count in chunk_types.items():
        print(f"  {ctype}: {count}")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
