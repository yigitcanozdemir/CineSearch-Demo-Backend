import pandas as pd
from models.pydantic_schemas import Features
from typing import List, Optional
import re
from config import QUALITY_LEVELS

class MovieFilter:
    def __init__(self):
        pass

    def apply_filters(self, data: pd.DataFrame, features: Features) -> pd.DataFrame:
        filtered_data = data.copy()
        

        if features.movie_or_series != "both":
            filtered_data = self._filter_by_type(
                filtered_data, features.movie_or_series
            )

        if features.genres or features.negative_genres:
            filtered_data["genreScore"] = filtered_data["genres"].apply(
                lambda g: self.calculate_genre_score(
                    g, 
                    features.genres or [], 
                    features.negative_genres or []
                )
            )
        else:
            
            filtered_data["genreScore"] = 0.0

        if features.date_range:
            filtered_data = self._filter_by_date_range(
                filtered_data, features.date_range
            )

        if features.quality_level:
            filtered_data = self._filter_by_quality(filtered_data, features.quality_level)

        if (
            features.min_runtime_minutes is not None
            or features.max_runtime_minutes is not None
        ):
            filtered_data = self._filter_by_runtime(
                filtered_data,
                features.min_runtime_minutes,
                features.max_runtime_minutes,
            )

        return filtered_data

    def _filter_by_runtime(
        self, data: pd.DataFrame, min_runtime: Optional[int], max_runtime: Optional[int]
    ) -> pd.DataFrame:

        data = data.dropna(subset=["runtimeMinutes"])
        data["runtimeMinutes"] = pd.to_numeric(
            data["runtimeMinutes"], errors="coerce"
        ).astype("Int64")

        data = data.dropna(subset=["runtimeMinutes"])
        if min_runtime is not None:
            data = data[data["runtimeMinutes"] >= min_runtime]

        if max_runtime is not None:
            data = data[data["runtimeMinutes"] <= max_runtime]

        return data

    def _filter_by_type(self, data: pd.DataFrame, movie_or_series: str) -> pd.DataFrame:
        if movie_or_series == "movie":
            movie_types = ["movie", "tvMovie", "video"]
            return data[data["titleType"].isin(movie_types)]
        elif movie_or_series == "tvSeries":
            series_types = ["tvSeries", "tvMiniSeries"]
            return data[data["titleType"].isin(series_types)]
        else:
            all_types = ["movie", "tvSeries", "tvMiniSeries", "tvMovie", "video"]
            return data[data["titleType"].isin(all_types)]

    def calculate_genre_score(self, row_genres: str, 
                                target_genres: List[str], 
                                negative_genres: List[str]) -> float:
            if not row_genres or pd.isna(row_genres):
                return 0.0
            
            try:
                row_genre_list = [g.strip().lower() for g in row_genres.split(",")]
                target_genre_list = [g.lower() for g in target_genres]
                negative_genre_list = [g.lower() for g in negative_genres]

                positive_matches = sum(1 for g in row_genre_list if g in target_genre_list)
                
                negative_matches = sum(1 for g in row_genre_list if g in negative_genre_list)

                score = 0.0
                if target_genres:
                    score = positive_matches / len(target_genre_list)
                elif positive_matches > 0:
                    score = 1.0 

                score -= (negative_matches * 0.5) 
                return max(0.0, score)
            except (AttributeError, TypeError):
                return 0.0


    def _filter_by_genres(self, data: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
        if not genres:
            return data

        def count_genre_matches(row_genres, target_genres):
            if pd.isna(row_genres):
                return 0

            row_genre_list = [g.strip().lower() for g in row_genres.split(",")]
            target_genre_list = [g.lower() for g in target_genres]

            matches = sum(
                1
                for target_genre in target_genre_list
                if any(target_genre in row_genre for row_genre in row_genre_list)
            )
            return matches

        data_with_matches = data.copy()
        data_with_matches["genre_matches"] = data_with_matches["genres"].apply(
            lambda g: count_genre_matches(g, genres)
        )

        filtered_2plus = data_with_matches[data_with_matches["genre_matches"] >= 2]

        if len(filtered_2plus) >= 20:
            print(f"Using 2+ genre matches: {len(filtered_2plus)} results")
            return filtered_2plus.drop("genre_matches", axis=1)

        filtered_1plus = data_with_matches[data_with_matches["genre_matches"] >= 1]
        print(f"Using 1+ genre matches: {len(filtered_1plus)} results")

        return filtered_1plus.drop("genre_matches", axis=1)

    def _filter_by_date_range(
        self, data: pd.DataFrame, date_range: List[int]
    ) -> pd.DataFrame:
        start_year, end_year = date_range
        return data[
            (data["startYear"].astype(int) >= start_year)
            & (data["startYear"].astype(int) <= end_year)
        ]

    def _filter_by_quality(
        self, data: pd.DataFrame, quality_level: str
    ) -> pd.DataFrame:
        if not quality_level or quality_level == "any":
            return data

        config = QUALITY_LEVELS.get(quality_level)
        if config:
            condition = pd.Series(True, index=data.index)
            if "min_rating" in config:
                condition &= (data["averageRating"] >= config["min_rating"])
            if "max_rating" in config:
                condition &= (data["averageRating"] <= config["max_rating"])
            if "min_votes" in config:
                condition &= (data["numVotes"] >= config["min_votes"])
            if "max_votes" in config:
                condition &= (data["numVotes"] <= config["max_votes"])
            return data[condition]

        return data
