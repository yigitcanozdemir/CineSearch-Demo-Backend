import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

GENRE_LIST = Literal[
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Musical",
    "Music",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
    "Biography",
    "Sport",
    "Film-Noir",
    "Talk-Show",
    "Game-Show",
    "News",
    "Short",
    "Adult",
    "Reality-TV",
]


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DATA_FILE = "data/demo_data.parquet"

    DEFAULT_RECOMMENDATIONS = 10
    MIN_SIMILARITY_THRESHOLD = 0.1

    SIMILARITY_WEIGHT = 0.6
    RATING_WEIGHT = 0.4

    THEME = "soft"
    TITLE = "🎬 AI Movie & TV Series Recommender"
