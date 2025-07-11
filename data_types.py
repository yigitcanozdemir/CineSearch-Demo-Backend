from typing import TypedDict, List, Any

class RecommendationItem(TypedDict):
    tconst: str
    titleType: str 
    primaryTitle: str 
    startYear: int
    runtimeMinutes: int | None
    genres: str
    averageRating: float 
    numVotes: float 
    overview: str 
    similarity_score: float
    hybrid_score: float 
    finalScore: float 
    genre_score: float