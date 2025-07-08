from pydantic import BaseModel, Field
from typing import Literal, Optional
from config import GENRE_LIST


class Features(BaseModel):
    movie_or_series: Literal["movie", "tvSeries"] = Field(
        description="Specify if the user wants a movie or a TV series"
    )
    genres: list[GENRE_LIST] = Field(
        description="List of genres from the predefined list"
    )
    quality_level: str = Field(
        description="Quality expectation: legendary, classic, popular, any"
    )
    themes: list[str] = Field(
        description="Actual thematic content (not quality descriptors)"
    )
    date_range: list[int] = Field(description="Date range [min_year, max_year] (note: min_year is 1900, max_year is 2025)")
    negative_keywords: list[str] = Field(description="List of negative keywords")
    production_region: list[str] = Field(description="Production region")
    min_rating: Optional[float] = Field(
        description="Minimum rating expectation", default=None
    )
    min_votes: Optional[int] = Field(
        description="Minimum number of votes", default=None
    )
