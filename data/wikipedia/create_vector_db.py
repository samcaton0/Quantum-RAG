"""
Create ChromaDB vector database from Wikipedia checkpoint files.

This script loads the pre-computed chunks and embeddings from the checkpoint
directory and populates a ChromaDB vector store for fast similarity search.
"""
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import chromadb


def main():
    # Paths
    script_dir = Path(__file__).parent.resolve()
    checkpoints_dir = script_dir / 'checkpoints'
    chunks_file = checkpoints_dir / 'chunks.jsonl'
    embeddings_file = checkpoints_dir / 'embeddings.npz'
    db_path = script_dir / 'chroma_db'

    print(f"Loading chunks from {chunks_file}...")
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embeddings from {embeddings_file}...")
    embeddings_npz = np.load(embeddings_file)
    embeddings = {key: embeddings_npz[key] for key in embeddings_npz.keys()}
    print(f"Loaded {len(embeddings)} embeddings")

    # Create ChromaDB client
    print(f"Creating ChromaDB at {db_path}...")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    # Delete existing collection if it exists
    try:
        client.delete_collection("wikipedia_chunks")
        print("  Deleted existing collection")
    except:
        pass

    # Create new collection
    collection = client.get_or_create_collection(
        name="wikipedia_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare chunks for bulk insert
    print("Inserting chunks into ChromaDB...")
    batch_size = 500

    for i in tqdm(range(0, len(chunks), batch_size), desc="Batches"):
        batch_chunks = chunks[i:i + batch_size]

        ids = []
        documents = []
        embeddings_batch = []
        metadatas = []

        for chunk in batch_chunks:
            chunk_id = chunk['chunk_id']
            if chunk_id not in embeddings:
                continue

            ids.append(chunk_id)
            documents.append(chunk['text'])
            embeddings_batch.append(embeddings[chunk_id].tolist())
            metadatas.append({
                'chunk_type': chunk.get('chunk_type', ''),
                'prompt_id': chunk.get('prompt_id', ''),
                'article_title': chunk.get('article_title', ''),
                'aspect_id': str(chunk.get('aspect_id', -1)),
                'aspect_name': chunk.get('aspect_name', ''),
                'redundancy_index': str(chunk.get('redundancy_index', -1)),
                'source': chunk.get('article_title', 'wikipedia')
            })

        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_batch,
                metadatas=metadatas
            )

    # Verify
    count = collection.count()
    print(f"\n✓ Successfully created vector database with {count} chunks")
    print(f"  Location: {db_path}")
    print(f"  Collection: wikipedia_chunks")

    # Test query
    print("\nTesting query...")
    test_chunk = chunks[0]
    test_embedding = embeddings[test_chunk['chunk_id']]
    results = collection.query(
        query_embeddings=[test_embedding.tolist()],
        n_results=5,
        include=["metadatas", "distances"]
    )
    print(f"  Retrieved {len(results['ids'][0])} results")
    if results['ids'][0]:
        print(f"  Top result: {results['metadatas'][0][0].get('article_title', 'N/A')}")

    print("\n✓ Vector database ready for use!")


if __name__ == '__main__':
    main()
