import gradio as gr
from models.recommendation_engine import RecommendationEngine
import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)


def get_recommendations_api(message, engine):
    if not message:
        return []

    try:
        result = engine.get_recommendations(message)
        df = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if df is None or df.empty:
            return []
        imdb_ids = df["ImdbId"].tolist()
        recommendations = []
        for idx, (_, row) in enumerate(df.iterrows()):
            recommendations.append(
                {
                    "imdb_id": row["ImdbId"],
                    "title": row["Title"],
                    "year": row["Year"],
                    "type": row["Type"],
                    "rating": row["Rating"],
                    "runtime_minutes": row["RuntimeMinutes"],
                    "votes": row["Votes"],
                    "genres": row["Genres"],
                    "similarity": row["Similarity"],
                    "hybrid_score": row["Hybrid Score"],
                    "overview": row["Overview"],
                    "poster_url": row["Poster Url"],
                    "final_score": row["Final Score"],
                    "genre_score": row["Genre Score"],
                }
            )

        result_df = pd.DataFrame(recommendations)[
            [
                "title",
                "final_score",
                "genre_score",
                "hybrid_score",
                "similarity",
                "votes",
                "rating",
            ]
        ]
        titles = result_df["title"].tolist()
        print(titles)
        print(result_df)
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []


def create_interface(engine):
    def predict_wrapper(message):
        return get_recommendations_api(message, engine)

    iface = gr.Interface(
        fn=predict_wrapper,
        inputs=gr.Textbox(lines=1, placeholder="Type your movie query..."),
        outputs=gr.JSON(label="Recommendations"),
        title="Movie Recommendation API",
        description="Type a movie or genre, get recommendations with posters.",
        api_name="predict",
    )
    return iface
