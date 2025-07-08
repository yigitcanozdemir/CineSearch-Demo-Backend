import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time


class SimilarityCalculator:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def calculate_similarity(
        self, query: str, filtered_data: pd.DataFrame, top_k: int = 10
    ) -> Dict[str, Any]:
        if filtered_data.empty:
            return {
                "status": "⚠️ No results found with current filters.",
                "results": [],
                "search_time": 0,
                "total_candidates": 0,
            }

        start_time = time.time()

        query_embedding = self.model.encode([query])
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32)

        document_embeddings = filtered_data["embedding"].tolist()
        document_embeddings = torch.tensor(
            np.array(document_embeddings), dtype=torch.float32
        )

        similarities = self.model.similarity(query_embedding, document_embeddings)
        similarities = similarities[0]

        hybrid_scores = self._calculate_hybrid_score(
            similarities, filtered_data, similarity_weight=0.6, rating_weight=0.4
        )

        top_indices = (
            torch.topk(hybrid_scores, min(top_k, len(hybrid_scores)))
            .indices.cpu()
            .numpy()
        )

        results = []
        for idx in top_indices:
            original_idx = filtered_data.iloc[idx].name
            row = filtered_data.iloc[idx]

            result = {
                "tconst": row["tconst"],
                "title": row["primaryTitle"],
                "type": row["titleType"],
                "year": row["startYear"],
                "rating": row["averageRating"],
                "votes": row["numVotes"],
                "genres": row["genres"],
                "overview": (
                    row["overview"][:200] + "..."
                    if len(row["overview"]) > 200
                    else row["overview"]
                ),
                "similarity_score": float(similarities[idx]),
                "hybrid_score": float(hybrid_scores[idx]),
                "final_score": float(row.get("final_score", 0)),
            }
            results.append(result)

        end_time = time.time()
        search_time = end_time - start_time

        return {
            "status": "✅ Search completed successfully.",
            "results": results,
            "search_time": search_time,
            "total_candidates": len(filtered_data),
            "query_embedding_shape": query_embedding.shape,
        }

    def _calculate_hybrid_score(
        self,
        similarities: torch.Tensor,
        data: pd.DataFrame,
        similarity_weight: float = 0.6,
        rating_weight: float = 0.4,
    ) -> torch.Tensor:

        sim_normalized = (similarities - similarities.min()) / (
            similarities.max() - similarities.min() + 1e-8
        )

        ratings = torch.tensor(data["averageRating"].values, dtype=torch.float32)
        rating_normalized = (ratings - ratings.min()) / (
            ratings.max() - ratings.min() + 1e-8
        )

        if "finalScore" in data.columns:
            final_scores = torch.tensor(data["finalScore"].values, dtype=torch.float32)
            final_normalized = (final_scores - final_scores.min()) / (
                final_scores.max() - final_scores.min() + 1e-8
            )

            hybrid_score = (
                similarity_weight * sim_normalized
                + rating_weight * rating_normalized
                + 0.2 * final_normalized
            )
        else:
            hybrid_score = (
                similarity_weight * sim_normalized + rating_weight * rating_normalized
            )

        return hybrid_score
