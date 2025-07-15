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
QUALITY_LEVELS = {
        "legendary": {"min_rating": 8.5, "min_votes": 100000,"rating_weight":0.3},
        "classic": {"min_rating": 7.5, "min_votes": 50000,"rating_weight":0.25},
        "popular": {"min_rating": 6.5, "min_votes": 10000,"rating_weight":0.2},
        "niche": {"min_rating": 7.0, "max_votes": 50000,"rating_weight":-0.1}, 
        "cult": {"min_rating": 6.0, "max_votes": 25000,"rating_weight":-0.15}, 
        "mainstream": {"min_rating": 5.5, "min_votes": 10000,"rating_weight":0.2},
        "any": {"rating_weight": 0.1}
    }

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DATA_FILE = "data/demo_data.parquet"

    THEME = "soft"
    TITLE = "🎬 AI Movie & TV Series Recommender"