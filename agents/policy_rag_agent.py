"""
Policy RAG Agent — Hybrid BM25 + FAISS retrieval with Reciprocal Rank Fusion.
Retrieves the most relevant policy chunks for a given return case.
"""
import os
import pickle
import numpy as np


class PolicyRAGAgent:
    """Tool: Retrieves top-k relevant policy clauses via hybrid search."""

    def __init__(self, indexes_dir: str, embedding_model_name: str):
        from sentence_transformers import SentenceTransformer
        import faiss as _faiss  # aliased so it's not confused with self.faiss_index

        if not os.path.exists(os.path.join(indexes_dir, 'faiss.index')):
            raise FileNotFoundError(
                f"RAG indexes not found in {indexes_dir}. "
                "Run:  python run_triage.py --build-index"
            )

        print("[PolicyRAGAgent] Loading indexes...")
        with open(os.path.join(indexes_dir, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)

        with open(os.path.join(indexes_dir, 'bm25.pkl'), 'rb') as f:
            self.bm25 = pickle.load(f)

        self.faiss_index = _faiss.read_index(
            os.path.join(indexes_dir, 'faiss.index')
        )
        self.encoder = SentenceTransformer(embedding_model_name)
        print(f"[PolicyRAGAgent] Ready. {len(self.chunks)} chunks indexed.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, case_data: dict, top_k: int = 5,
            bm25_n: int = 20, faiss_n: int = 20, rrf_k: int = 60) -> list:
        """
        Retrieve relevant policy chunks for a return case.

        Returns
        -------
        list[dict]  — top-k chunks, each with keys: text, source, section, rrf_score
        """
        raw = case_data['raw']

        # Build a natural-language query from the case context
        query = (
            f"{raw.get('category', '')} {raw.get('reason_category', '')} "
            f"{raw.get('return_type', '')} return "
            f"{'non-returnable' if raw.get('is_non_returnable') else 'returnable'} "
            f"{'outside return window' if not raw.get('within_return_window') else 'within return window'} "
            f"order value {raw.get('order_value', '')} "
            f"days left {raw.get('days_left_to_return', '')}"
        )

        # ── BM25 search ──────────────────────────────────────────────────────
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = np.argsort(bm25_scores)[::-1][:bm25_n]

        # ── FAISS (semantic) search ───────────────────────────────────────────
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        faiss_scores, faiss_indices = self.faiss_index.search(q_emb, faiss_n)
        faiss_top = faiss_indices[0]

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_top):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(faiss_top):
            if idx >= 0:  # FAISS pads with -1
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in sorted_results[:top_k]:
            chunk = self.chunks[idx]
            results.append({
                'text': chunk['text'],
                'source': chunk['source'],
                'section': chunk['section'],
                'rrf_score': round(score, 5),
            })

        return results
