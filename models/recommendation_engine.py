import pandas as pd
import time
from openai import OpenAI
from config import Config
from models.pydantic_schemas import Features
from components.similarity import SimilarityCalculator
from components.filters import MovieFilter
from sentence_transformers import SentenceTransformer
from components.tmdb_api import TMDBApi
import traceback
import sys


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

    def get_recommendations(self, user_query: str, top_k: int = 40):
        print(f"🚀 Starting recommendation process for query: '{user_query}'")
        if not user_query.strip():
            return "⚠️ Please enter some text.", None

        try:
            print("📝 Parsing user query...")
            start_time = time.time()
            features = self._parse_user_query(user_query)
            parse_time = time.time() - start_time
            print(f"✅ Query parsed in {parse_time:.4f} seconds")

            print("🔍 Applying filters...")
            start_time = time.time()
            filtered_data = self.filter.apply_filters(self.data, features)
            filter_time = time.time() - start_time
            print(f"✅ Filters applied in {filter_time:.4f} seconds")
            print(f"🔍 Filtered data contains {len(filtered_data)} items.")
            print("🔧 Preparing query input...")
            query_input = features.themes + features.named_entities
            if not query_input:
                query_input = [user_query]
            query_text = " ".join(query_input)
            print(f"📝 Query text for embedding: '{query_text}'")
            print("🧮 Starting similarity calculation...")
            start_time = time.time()
            try:
                search_results = self.similarity_calc.calculate_similarity(
                    query_text, filtered_data, top_k
                )
                similarity_time = time.time() - start_time
                print(
                    f"✅ Similarity calculation completed in {similarity_time:.4f} seconds"
                )

            except Exception as similarity_error:
                print(f"❌ Error in similarity calculation: {str(similarity_error)}")
                print(f"📊 Traceback: {traceback.format_exc()}")

                # Try with smaller batch or different approach
                print("🔄 Attempting recovery with smaller dataset...")
                if len(filtered_data) > 1000:
                    # Try with smaller subset
                    smaller_data = filtered_data.sample(n=1000, random_state=42)
                    search_results = self.similarity_calc.calculate_similarity(
                        query_text, smaller_data, top_k
                    )
                    print("✅ Recovery successful with smaller dataset")
                else:
                    raise similarity_error

            print(f"🔍 Found {len(search_results['results'])} results.")
            print("📋 Formatting results...")
            start_time = time.time()
            formatted_results = self._format_results(search_results)
            format_time = time.time() - start_time
            print(f"✅ Results formatted in {format_time:.4f} seconds")

            # Create dataframe
            print("📊 Creating results dataframe...")
            start_time = time.time()
            results_df = self._create_results_dataframe(search_results)
            df_time = time.time() - start_time
            print(f"✅ Dataframe created in {df_time:.4f} seconds")

            print("🎉 Recommendation process completed successfully!")
            return formatted_results, results_df

        except Exception as e:
            print(f"❌ Critical error in recommendation process: {str(e)}")
            print(f"📊 Full traceback: {traceback.format_exc()}")
            print(f"🔍 Exception type: {type(e).__name__}")

            # Memory usage check
            try:
                import psutil

                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                print(f"💾 Current memory usage: {memory_usage:.2f} MB")
            except:
                pass

            return f"❌ Error: {str(e)}", None

    def _parse_user_query(self, query: str) -> Features:
        try:
            print(f"📤 Sending query to OpenAI: '{query}'")
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI that converts user requests into structured movie/TV-series features.

                                    GENRE EXTRACTION RULES: 
                                    1. If user mentions a specific movie/show, include the ACTUAL genres of that content, if you do not sure include 1 or 2 genres.
                                    2. Prioritize the most common/popular genres for the referenced content
                                    3. If genres are mentioned directly select genre from the given list

                                    THEMES EXTRACTION RULES:
                                    1. Include 1–5 inferred narrative or stylistic themes (e.g., "vigilantes", "dark humor", "philosophical").
                                    2. ALWAYS include at least 1-3 meaningful `themes` even if the user query seems entity-focused.
                                    3. **CRITICAL: Always preserve specific contextual keywords from the user query in themes.**
                                    4. Themes will use for embedding and semantic search, so they should be general enough to capture the essence of the query but specific enough to find relevant content.
                                    
                                    Examples:
                                    - If user mentions "Vietnam War" → include "Vietnam" in themes
                                    - If user mentions "World War 2" → include "World War 2" or "WWII" in themes
                                    - If user mentions "Cold War" → include "Cold War" in themes
                                    - If user mentions specific historical events, locations, or periods → include them in themes
                                    - If user mentions specific concepts like "zombies", "aliens", "time travel" → include them in themes

                                    KEYWORD PRESERVATION RULES:
                                    - Extract and preserve important specific nouns, proper nouns, and key concepts from the user query
                                    - These should be added to themes alongside the general thematic elements
                                    - This ensures semantic search finds content specifically about those topics, not just similar themes

                                    OTHER RULES:
                                    If the user query includes known franchises or brands (like "Marvel", "Harry Potter", "Studio Ghibli", "Christopher Nolan"), enrich the `themes` section to reflect the core narrative/stylistic elements of that universe.
                                    For example:
                                    - "Marvel" → themes like "superheroes", "team-based heroes", "interconnected universe", "identity crisis", "comic-book style", "good vs evil", "post-credit twists"
                                    - "Ghibli" → "magical realism", "nature vs industry", "childhood wonder", "silent protagonists"
                                    - "Tarantino" → "violence with style", "non-linear narrative", "retro soundtracks", "revenge", "pop culture references"
                                    Always include these enriched thematic representations when entity is present.
                                    Do NOT leave `themes` empty unless the query is entirely meaningless or gibberish.

                                    NOTE: If the user query different language, you should translate it to English first and parse.
                                    ALSO NOTE: This are examples do it for other franchises or brands its template""",
                    },
                    {"role": "user", "content": query},
                ],
                response_format=Features,
            )

            response_model = response.choices[0].message.parsed
            print(f"📥 OpenAI response received successfully")
            print(f"🔍 Response type: {type(response_model)}")
            print(f"📋 Response content: {response_model.model_dump_json(indent=2)}")
            return response_model
        except Exception as e:
            print(f"❌ Error parsing user query: {str(e)}")
            print(f"📊 Parse error traceback: {traceback.format_exc()}")
            return Features(
                movie_or_series="both",
                genres=[],
                quality_level="any",
                themes=[query],  # Include original query as theme for fallback
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
                    "ImdbId": result["tconst"],
                    "Title": result["title"],
                    "Type": result["type"],
                    "Year": result["year"],
                    "Rating": result["rating"],
                    "RuntimeMinutes": result["runtimeMinutes"],
                    "Votes": result["votes"],
                    "Genres": result["genres"],
                    "Similarity": f"{result['similarity_score']:.4f}",
                    "Hybrid Score": f"{result['hybrid_score']:.4f}",
                    "Overview": result["overview"],
                    "Final Score": f"{result['final_score']:.4f}",
                    "Genre Score": f"{result['genre_score']:.4f}"
                }
            )
        print(df_data)
        return pd.DataFrame(df_data)
