from pydantic import BaseModel, Field
from typing import Literal, Optional
from config import GENRE_LIST, COUNTRY_LIST


class Features(BaseModel):
    movie_or_series: Literal["movie", "tvSeries", "both"] = Field(
        description="Specify if the user wants a movie or a TV series or both"
    )
    genres: list[GENRE_LIST] = Field(
        description="List of genres from the predefined list"
    )
    negative_genres: list[GENRE_LIST] = Field(
        description="Unwanted list of genres from the predefined list"
    )
    quality_level: str = Field(
        default="any",
        description="Quality expectation: legendary, classic, popular, niche, cult, mainstream, any",
    )
    positive_themes: Optional[str] = Field(
        description="Themes that should be present in the results",
    )
    negative_themes: Optional[str] = Field(
        description="Themes that should be avoided in the results"
    )
    date_range: list[int] = Field(
        description="Date range [min_year, max_year] (note: min_year is 1900, max_year is 2025)"
    )
    min_runtime_minutes: Optional[int] = Field(
        description="Preferred minumum runtimes as minutes", default=None
    )
    max_runtime_minutes: Optional[int] = Field(
        description="Preferred maximum runtimes as minutes", default=None
    )
    country_of_origin: list[COUNTRY_LIST] = Field(
        description="Preferred country of production"
    )
    dont_wanted_countrys: list[COUNTRY_LIST] = Field(
        description="Unwanted country of production"
    )
    prompt_title: str = Field(description="A short and meaningful title for the prompt")
