import pandas as pd
from models.pydantic_schemas import Features
from typing import List, Optional
import re


class MovieFilter:
    def __init__(self):
        pass

    def apply_filters(self, data: pd.DataFrame, features: Features) -> pd.DataFrame:
        filtered_data = data.copy()

        if features.movie_or_series != "both":
            filtered_data = self._filter_by_type(
                filtered_data, features.movie_or_series
            )

        if features.genres:
            filtered_data = self._filter_by_genres(filtered_data, features.genres)

        if features.date_range:
            filtered_data = self._filter_by_date_range(
                filtered_data, features.date_range
            )

        if features.min_rating is not None:
            filtered_data = self._filter_by_rating(filtered_data, features.min_rating)

        if features.min_votes is not None:
            filtered_data = self._filter_by_votes(filtered_data, features.min_votes)

        if features.negative_keywords:
            filtered_data = self._filter_by_negative_keywords(
                filtered_data, features.negative_keywords
            )

        if features.quality_level != "any":
            filtered_data = self._filter_by_quality(
                filtered_data, features.quality_level
            )

        return filtered_data

    def _filter_by_type(self, data: pd.DataFrame, movie_or_series: str) -> pd.DataFrame:
        return data[data["titleType"] == movie_or_series]

    def _filter_by_genres(self, data: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
        return data[
            data["genres"].apply(
                lambda g: any(genre in g.split(",") for genre in genres)
            )
        ]

    def _filter_by_date_range(
        self, data: pd.DataFrame, date_range: List[int]
    ) -> pd.DataFrame:
        start_year, end_year = date_range
        return data[
            (data["startYear"].astype(int) >= start_year)
            & (data["startYear"].astype(int) <= end_year)
        ]

    def _filter_by_rating(self, data: pd.DataFrame, min_rating: float) -> pd.DataFrame:
        return data[data["averageRating"] >= min_rating]

    def _filter_by_votes(self, data: pd.DataFrame, min_votes: int) -> pd.DataFrame:
        return data[data["numVotes"] >= min_votes]

    def _filter_by_negative_keywords(
        self, data: pd.DataFrame, negative_keywords: List[str]
    ) -> pd.DataFrame:
        for keyword in negative_keywords:
            data = data[~data["overview"].str.contains(keyword, case=False, na=False)]
        return data

    def _filter_by_quality(
        self, data: pd.DataFrame, quality_level: str
    ) -> pd.DataFrame:
        if quality_level == "legendary":
            return data[(data["averageRating"] >= 8.5) & (data["numVotes"] >= 100000)]
        elif quality_level == "classic":
            return data[(data["averageRating"] >= 7.5) & (data["numVotes"] >= 50000)]
        elif quality_level == "popular":
            return data[(data["averageRating"] >= 6.5) & (data["numVotes"] >= 10000)]
        return data
