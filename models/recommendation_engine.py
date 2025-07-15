import pandas as pd
import time
from openai import OpenAI
from config import Config
from models.pydantic_schemas import Features
from components.similarity import SimilarityCalculator
from components.filters import MovieFilter
from sentence_transformers import SentenceTransformer
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
            print(
                f"📝 Query text for embedding: Positive ['{features.positive_themes}'], Negative [{features.negative_themes}]"
            )
            print("🧮 Starting similarity calculation...")
            start_time = time.time()
            try:
                search_results = self.similarity_calc.calculate_similarity(
                    features, filtered_data, top_k
                )
                similarity_time = time.time() - start_time
                print(
                    f"✅ Similarity calculation completed in {similarity_time:.4f} seconds"
                )

            except Exception as similarity_error:
                print(f"❌ Error in similarity calculation: {str(similarity_error)}")
                print(f"📊 Traceback: {traceback.format_exc()}")

                print("🔄 Attempting recovery with smaller dataset...")
                if len(filtered_data) > 1000:
                    smaller_data = filtered_data.sample(n=1000, random_state=42)
                    search_results = self.similarity_calc.calculate_similarity(
                        features, smaller_data, top_k
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

            try:
                import psutil

                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024
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
                        "content": """You are an AI that converts natural language movie/TV preferences into structured features based on a predefined schema.

                                    Your output must strictly follow the `Features` schema. You do not need to re-define the field names; just ensure correct values are produced.

                                    ## FIELD-SPECIFIC EXTRACTION RULES:

                                    ---

                                    ### GENRES
                                    - If the user mentions a specific movie/show, extract its ACTUAL genres (e.g., IMDb/TMDB genres).
                                    - If unsure, infer 1–2 of the most likely/popular genres.
                                    - If user directly mentions genres, match exactly from the allowed genre list.
                                    - Prefer accuracy over guessing; leave empty if absolutely no genre can be inferred.

                                    ---

                                    ### THEMES (positive_themes & negative_themes)

                                    **CRITICAL: Write these like IMDb or Netflix overviews. Keep them punchy, real, and franchise-specific when needed.**

                                    #### Writing Style Guidelines:
                                    - Write **2 sentences maximum** like real IMDb overviews
                                    - Use simple, direct language that captures the core conflict
                                    - Include specific universe/franchise names when mentioned by user
                                    - Focus on WHO does WHAT and WHY (conflict/stakes)
                                    - Keep it concise and searchable

                                    #### UNIVERSE-SPECIFIC CONTEXT RULES:
                                    **When user mentions specific franchises, you MUST use universe-specific terminology and context instead of generic descriptions:**

                                    **DC Universe**: Use "Justice League", "Gotham City", "Metropolis", "Wayne Enterprises", "Daily Planet", "Arkham", "Kryptonite", "Joker", "Lex Luthor", "Darkseid", "Batman", "Superman", "Wonder Woman", "The Flash", "Green Lantern"
                                    Examples:
                                    - "Batman must protect Gotham City from the Joker's deadly scheme."
                                    - "The Justice League faces their greatest threat when Darkseid invades Earth."
                                    - "Superman struggles to save Metropolis while confronting his Kryptonian heritage."

                                    **Marvel Universe**: Use "Avengers", "S.H.I.E.L.D.", "Wakanda", "Asgard", "Infinity Stones", "Thanos", "Stark Industries", "X-Men", "Mutants", "Vibranium", "Iron Man", "Captain America", "Spider-Man", "Thor", "Hulk"
                                    Examples:
                                    - "The Avengers must collect the Infinity Stones before Thanos destroys the universe."
                                    - "Spider-Man balances teenage life while protecting New York from villains."
                                    - "Wakanda's advanced technology becomes Earth's last hope against invasion.
                                    - "As Steve Rogers struggles to embrace his role in the modern world, he teams up with a fellow Avenger and S.H.I.E.L.D agent, Black Widow, to battle a new threat from history: an assassin known as the Winter Soldier."

                                    **Star Wars**: Use "Jedi", "Sith", "The Force", "Empire", "Rebellion", "Death Star", "Lightsaber", "Darth Vader", "Luke Skywalker", "Princess Leia", "Han Solo", "Millennium Falcon", "Tatooine", "Coruscant"
                                    Examples:
                                    - "A young Jedi must master the Force to defeat the evil Sith Lord."
                                    - "The Rebellion attempts to destroy the Empire's ultimate weapon, the Death Star."

                                    **Harry Potter**: Use "Hogwarts", "Wizarding World", "Voldemort", "Death Eaters", "Quidditch", "Ministry of Magic", "Dumbledore", "Snape", "Hermione", "Ron", "Diagon Alley", "Horcrux"
                                    Examples:
                                    - "Harry Potter must find and destroy Voldemort's Horcruxes to save the wizarding world."
                                    - "Students at Hogwarts face dark forces threatening their magical education."

                                    **Fast & Furious**: Use "street racing", "heist crew", "family bonds", "high-speed chases", "international crime", "Dom Toretto", "Letty", "Roman", "Tej", "Hobbs", "Shaw"
                                    Examples:
                                    - "Dom Toretto's crew must pull off an impossible heist to save their family."
                                    - "Street racers become international spies to stop a cyber-terrorist."

                                    **John Wick**: Use "assassin underworld", "Continental Hotel", "High Table", "gold coins", "excommunicado", "Baba Yaga", "Winston", "Charon"
                                    Examples:
                                    - "A legendary assassin seeks revenge against the High Table after being declared excommunicado."
                                    - "John Wick must navigate the underground assassin world to protect those he loves."

                                    **Mission Impossible**: Use "IMF", "Ethan Hunt", "impossible mission", "disavowed", "rogue agents", "high-tech gadgets", "death-defying stunts"
                                    Examples:
                                    - "IMF agent Ethan Hunt must complete an impossible mission to prevent global catastrophe."
                                    - "A disavowed spy uses cutting-edge technology to expose a conspiracy."

                                    **James Bond**: Use "007", "MI6", "secret agent", "SPECTRE", "Q", "M", "Aston Martin", "gadgets", "international espionage"
                                    Examples:
                                    - "Agent 007 must stop SPECTRE from executing their world domination plan."
                                    - "A British secret agent uses high-tech gadgets to infiltrate enemy operations."

                                    **CRITICAL IMPLEMENTATION RULES:**
                                    ✅ **ALWAYS use franchise-specific terminology when user mentions a universe**
                                    ✅ **Include iconic characters, locations, and concepts from that universe**
                                    ✅ **Make it sound like an actual movie from that franchise**
                                    ✅ **Use present tense and active voice**
                                    ✅ **Keep it 1-2 sentences maximum**

                                    ❌ **NEVER use generic "superheroes" when user says "Marvel" or "DC"**
                                    ❌ **NEVER write "Marvel heroes" or "DC heroes" - use specific names**
                                    ❌ **NEVER ignore the universe context provided by the user**

                                    #### WRITING TEMPLATE FOR FRANCHISES:
                                    "[Specific franchise characters/locations] must [action] when/to [franchise-specific threat/goal]."

                                    ### GENERAL CONTEXT RULES (NON-FRANCHISE THEMES)

                                    If no franchise is explicitly mentioned:

                                    - Base the theme on **realistic, grounded context** if user requests Mafia, crime drama, political thriller, etc.
                                    - Include **time period or location** if hinted or inferred (e.g., 1970s, Cold War, post-WWII, New York City, Mexico border).
                                    - Use **specific genre terms** like “cartel”, “mob”, “law enforcement”, “drug empire”, “FBI”, “prosecutor”, “detective”, “undercover”, “corruption”.
                                    - Avoid vague language like “family power struggles” that could match superheroes, fantasy, or Batman.
                                    - If user mentions realism, EXCLUDE superhero, fantasy, or supernatural vocabulary entirely.

                                    ✅ GOOD THEMES EXAMPLES:
                                    - “In 1970s New York, a Mafia don must navigate betrayal and FBI pressure to hold his criminal empire together.”
                                    - “A Mexican drug lord rises to power as DEA agents close in on his cross-border empire.”
                                    - "In an alternative version of 1969, the Soviet Union beats the United States to the Moon, and the space race continues on for decades with still grander challenges and goals."
                                    - "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft, along with a team of researchers, to find a new planet for humans"
                                    - "An astronaut becomes stranded on Mars after his team assume him dead, and must rely on his ingenuity to find a way to signal to Earth that he is alive and can survive until a potential rescue."
                                    ❌ BAD THEMES TO AVOID:
                                    - “A powerful family faces betrayal as they try to protect their empire.” ⟶ Too vague and franchise-prone

                                    ---
                                    ### NEGATIVE THEMES
                                    - USE SAME FORMAT AS POZITIVE FOR NEGATIVE
                                    ---
                                    ### NON-FRANCHISE THEME TEMPLATE:

                                    If no franchise is referenced and the genre is crime, drama, thriller, or historical:
                                    - Use grounded real-world locations, eras, and power structures
                                    - Mention role labels like mob boss, FBI agent, cartel leader, war veteran, prosecutor
                                    - Avoid fantasy, superhero, or comic-book phrasing
                                    - If the user says “mafia”, include terms like “mob”, “Cosa Nostra”, “organized crime”, “criminal empire”
                                        
                                    
                                    #### POLARITY:
                                    - `positive_themes`: What the user WANTS - write as an appealing movie description using franchise context
                                    - `negative_themes`: What the user wants to AVOID - write as movie overview plot to exclude
                                    - `negative_genres`: What the user want to AVOID - write unwanted genres
                                    ### QUALITY_LEVEL
                                    Infer quality level if possible from user's tone:
                                    - "best movie ever", "all-time favorite" → `legendary`
                                    - "cult classic", "iconic" → `classic`
                                    - "popular", "well-known", "fun" → `popular`
                                    - If not stated → `any`

                                    ---

                                    ### RUNTIME
                                    - If user mentions "short", "under 2h", etc., infer `max_runtime_minutes`
                                    - If user says "long", "multi-hour", infer `min_runtime_minutes`
                                    - Otherwise leave runtime fields as null.

                                    ---

                                    ### DATE_RANGE
                                    If user mentions a year or era (e.g., "80s movies", "recent stuff", "old classics"), infer a `[min_year, max_year]`.
                                    - Defaults to `[1900, 2025]` if not constrained.
                                    - "recent", "modern" → prefer `[2010, 2025]`
                                    - "classic", "old" → prefer `[1950, 1995]`

                                    ---

                                    ### LANGUAGE
                                    If the query is not in English, **translate to English first**, then apply the above rules.

                                    ---

                                    **NEVER leave `positive_themes` empty, if query pure nonsense then use most popular movie overview**
                                    **ALWAYS write themes as compelling movie/TV descriptions using franchise-specific context.**
                                    """,
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
                themes=[query],
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
                    "Genre Score": f"{result['genre_score']:.4f}",
                    "Poster Url": result["poster_url"],
                }
            )
        return pd.DataFrame(df_data)
