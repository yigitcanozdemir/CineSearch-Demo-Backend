import pandas as pd
import time
from openai import OpenAI
from config import Config
from models.pydantic_schemas import Features
from components.similarity import SimilarityCalculator
from components.filters import MovieFilter
from sentence_transformers import SentenceTransformer
from components.tmdb_api import TMDBApi

class RecommendationEngine:
    def __init__(self):
        self.config = Config()
        self.model = SentenceTransformer(
            self.config.EMBEDDING_MODEL, trust_remote_code=True
        )
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        self.data = pd.read_parquet(self.config.DATA_FILE)

        self.similarity_calc = SimilarityCalculator(self.model)
        self.filter = MovieFilter()
        self.tmdb_api = TMDBApi()

        print(f"✅ Recommendation engine initialized with {len(self.data)} items.")

    def get_recommendations(self, user_query: str, top_k: int = 10):

        if not user_query.strip():
            return "⚠️ Please enter some text.", None

        try:
            features = self._parse_user_query(user_query)

            filtered_data = self.filter.apply_filters(self.data, features)

            search_results = self.similarity_calc.calculate_similarity(
                features.themes, filtered_data, top_k
            )
            if search_results["results"]:
                    print(f"🔍 First result keys: {search_results['results'][0].keys()}")
                    
                    for i, result in enumerate(search_results["results"]):
                        print(f"🔍 Result {i}: tconst = {result.get('tconst', 'NOT FOUND')}")
                        
                    search_results["results"] = self.tmdb_api.get_multiple_posters_by_imdb(
                        search_results["results"]
                    )

            formatted_results = self._format_results(search_results)

            return formatted_results, self._create_results_dataframe(search_results)

        except Exception as e:
            return f"❌ Error: {str(e)}", None

    def _parse_user_query(self, query: str) -> Features:
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI that converts user requests into structured movie/TV-series features. ONLY extract genres that are explicitly mentioned by the user. Do not infer or add additional genres unless clearly stated.",
                    },
                    {"role": "user", "content": query},
                ],
                response_format=Features,
            )
            
            
            response_model = response.choices[0].message.parsed

            print(type(response_model))
            print(response_model.model_dump_json(indent=2))
            print(f"✅ Parsed features: {response_model}")
            return response_model
        
        
        
        except Exception as e:
            print(f"❌ Error parsing user query: {str(e)}")
            return Features(
                movie_or_series="both",
                genres=[],
                quality_level="any",
                themes=[],
                date_range=[2000, 2025],
                negative_keywords=[],
                production_region=[],
            )

    def _format_results(self, search_results: dict) -> str:
        if not search_results["results"]:
            return search_results["status"]

        output = []
        output.append(f"🎬 {search_results['status']}")
        output.append(
            f"🔍 Search completed in {search_results['search_time']:.4f} seconds"
        )
        output.append(
            f"📊 Found {len(search_results['results'])} results from {search_results['total_candidates']} candidates"
        )
        output.append("=" * 50)

        for i, result in enumerate(search_results["results"], 1):
            output.append(f"{i}. **{result['title']}** ({result['year']})")
            output.append(f"   📝 Type: {result['type'].title()}")
            output.append(
                f"   ⭐ Rating: {result['rating']}/10 ({result['votes']:,} votes)"
            )
            output.append(f"   🎭 Genres: {result['genres']}")
            output.append(f"   📊 Similarity: {result['similarity_score']:.4f}")
            output.append(f"   🏆 Hybrid Score: {result['hybrid_score']:.4f}")
            output.append(f"   📄 {result['overview']}")
            output.append("")

        return "\n".join(output)

    def _create_results_dataframe(self, search_results: dict) -> pd.DataFrame:
        if not search_results["results"]:
            return pd.DataFrame()

        df_data = []
        for result in search_results["results"]:
            df_data.append(
                {
                    "Title": result["title"],
                    "Type": result["type"],
                    "Year": result["year"],
                    "Rating": result["rating"],
                    "Votes": result["votes"],
                    "Genres": result["genres"],
                    "Similarity": f"{result['similarity_score']:.4f}",
                    "Hybrid Score": f"{result['hybrid_score']:.4f}",
                    "Overview": result["overview"],
                }
            )
        print(df_data)
        return pd.DataFrame(df_data)
