import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time
from config import QUALITY_LEVELS


class SimilarityCalculator:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def combined_and_score(self, similarity_matrix, alpha=10):

        smooth_min = -torch.logsumexp(-alpha * similarity_matrix, dim=0) / alpha
        return smooth_min

    def calculate_similarity(
        self, features: str, filtered_data: pd.DataFrame, top_k: int = 40
    ) -> Dict[str, Any]:
        if filtered_data.empty:
            return {
                "status": "No results found with current filters.",
                "results": [],
                "search_time": 0,
                "total_candidates": 0,
            }

        start_time = time.time()
        positive_themes = features.positive_themes
        negative_themes = features.negative_themes

        positive_query_embeddings_np = self.model.encode(
            positive_themes, convert_to_numpy=True
        )

        positive_query_embeddings = torch.tensor(
            positive_query_embeddings_np, dtype=torch.float32
        )

        if (
            positive_query_embeddings.dim() > 1
            and positive_query_embeddings.shape[0] > 1
        ):
            avg_positive = torch.mean(positive_query_embeddings, dim=0, keepdim=True)
        else:
            avg_positive = positive_query_embeddings
        document_embeddings = torch.tensor(
            np.array(filtered_data["embedding"].tolist()), dtype=torch.float32
        )

        if negative_themes is not None and len(negative_themes) > 0:

            negative_query_embeddings_np = self.model.encode(
                negative_themes, convert_to_numpy=True
            )
            negative_query_embeddings = torch.tensor(
                negative_query_embeddings_np, dtype=torch.float32
            )

            if (
                negative_query_embeddings.dim() > 1
                and negative_query_embeddings.shape[0] > 1
            ):
                avg_negative = torch.mean(
                    negative_query_embeddings, dim=0, keepdim=True
                )
            else:
                avg_negative = negative_query_embeddings
            positive_weight = 1.0
            negative_influence = 0.6 # Setting this value to 1 is so harsh so I just used smaller value
            combined_embedding = (positive_weight * avg_positive) - (
                negative_influence * avg_negative
            )

        else:
            combined_embedding = avg_positive

        similarities = self.model.similarity(combined_embedding, document_embeddings)
        similarities = similarities[0]

        quality_config = QUALITY_LEVELS.get(features.quality_level, {})
        rating_weight = quality_config.get("rating_weight")
        hybrid_scores = self._calculate_hybrid_score(
            similarities,
            filtered_data,
            similarity_weight=1,
            rating_weight=rating_weight,
            genre_weight=0.3,
        )

        top_indices = (
            torch.topk(hybrid_scores, min(top_k, len(hybrid_scores)))
            .indices.cpu()
            .numpy()
        )
        results = []
        for idx in top_indices:
            row = filtered_data.iloc[idx]

            result = {
                "tconst": row["tconst"],
                "title": row["primaryTitle"],
                "type": row["titleType"],
                "year": row["startYear"],
                "rating": row["averageRating"],
                "runtimeMinutes": row.get("runtimeMinutes", None),
                "votes": row["numVotes"],
                "genres": row["genres"],
                "overview": row["overview"],
                "similarity_score": float(similarities[idx]),
                "hybrid_score": float(hybrid_scores[idx]),
                "final_score": row["finalScore"],
                "genre_score": row["genreScore"],
                "poster_url": row["poster_url"],
                "country_of_origin": row["country_of_origin"],
            }
            results.append(result)

        end_time = time.time()
        search_time = end_time - start_time

        return {
            "status": "Search completed successfully.",
            "results": results,
            "search_time": search_time,
            "total_candidates": len(filtered_data),
            "query_embedding_shape": combined_embedding.shape,
        }

    def _calculate_hybrid_score(
        self,
        similarities: torch.Tensor,
        data: pd.DataFrame,
        similarity_weight: float = 1,
        rating_weight: float = 0.1,
        genre_weight: float = 0.3,
    ) -> torch.Tensor:

        if "finalScore" in data.columns:
            final_scores = torch.tensor(data["finalScore"].values, dtype=torch.float32)
            genre_score = torch.tensor(data["genreScore"].values, dtype=torch.float32)
            final_normalized = (final_scores - final_scores.min()) / (
                final_scores.max() - final_scores.min() + 1e-8
            )

            total_weight = similarity_weight + rating_weight + genre_weight
            hybrid_score = (
                similarity_weight * similarities
                + rating_weight * final_normalized
                + genre_weight * genre_score
            ) / total_weight

        return hybrid_score
