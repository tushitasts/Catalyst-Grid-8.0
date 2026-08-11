"""
Build FAISS + BM25 indexes from policy documents.
Run once:  python -c "from rag.build_index import build_indexes; import settings; build_indexes(settings.POLICIES_DIR, settings.INDEXES_DIR, settings.EMBEDDING_MODEL)"
Or via:    python run_triage.py --build-index
"""
import os
import pickle
import numpy as np


def load_policy_docs(policies_dir: str) -> list:
    """Load all .md files from the policies directory."""
    docs = []
    for fname in sorted(os.listdir(policies_dir)):
        if fname.endswith('.md'):
            filepath = os.path.join(policies_dir, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            docs.append({'filename': fname, 'content': content})
    return docs


def chunk_documents(docs: list, max_chunk_size: int = 500) -> list:
    """Split documents into semantic chunks by ## headers."""
    chunks = []
    for doc in docs:
        sections = doc['content'].split('\n## ')
        for i, section in enumerate(sections):
            text = section.strip() if i == 0 else ('## ' + section.strip())

            if len(text) > max_chunk_size:
                # Split long sections by double-newline paragraphs
                paragraphs = text.split('\n\n')
                current = ''
                for para in paragraphs:
                    if len(current) + len(para) > max_chunk_size and current:
                        chunks.append({
                            'text': current.strip(),
                            'source': doc['filename'],
                            'section': current.strip().split('\n')[0][:80],
                        })
                        current = para
                    else:
                        current = (current + '\n\n' + para) if current else para
                if current.strip():
                    chunks.append({
                        'text': current.strip(),
                        'source': doc['filename'],
                        'section': current.strip().split('\n')[0][:80],
                    })
            elif text:
                chunks.append({
                    'text': text,
                    'source': doc['filename'],
                    'section': text.split('\n')[0][:80],
                })
    return chunks


def build_indexes(policies_dir: str, indexes_dir: str, embedding_model_name: str):
    """Build and persist FAISS + BM25 indexes from policy documents."""
    from sentence_transformers import SentenceTransformer
    import faiss
    from rank_bm25 import BM25Okapi

    os.makedirs(indexes_dir, exist_ok=True)

    # ── Load & chunk ──────────────────────────────────────────────────────────
    print("[build_index] Loading policy documents...")
    docs = load_policy_docs(policies_dir)
    print(f"  Loaded {len(docs)} documents.")

    print("[build_index] Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks.")

    # ── BM25 ──────────────────────────────────────────────────────────────────
    print("[build_index] Building BM25 index...")
    tokenized = [chunk['text'].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(os.path.join(indexes_dir, 'bm25.pkl'), 'wb') as f:
        pickle.dump(bm25, f)

    # ── FAISS ─────────────────────────────────────────────────────────────────
    print(f"[build_index] Loading embedding model: {embedding_model_name} ...")
    model = SentenceTransformer(embedding_model_name)

    print("[build_index] Encoding chunks...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)           # cosine similarity (on L2-normed vecs)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, os.path.join(indexes_dir, 'faiss.index'))

    # ── Chunks metadata ──────────────────────────────────────────────────────
    with open(os.path.join(indexes_dir, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)

    print(f"[build_index] Done! Saved to {indexes_dir}/")
    print(f"  FAISS index : {index.ntotal} vectors, dim={dim}")
    print(f"  BM25 index  : {len(tokenized)} documents")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import settings
    build_indexes(settings.POLICIES_DIR, settings.INDEXES_DIR, settings.EMBEDDING_MODEL)
