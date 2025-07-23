import gradio as gr
from models.recommendation_engine import RecommendationEngine
import pandas as pd
import asyncio

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
        prompt_title = result[0]
        recommendations = []
        for idx, (_, row) in enumerate(df.iterrows()):
            recommendations.append(
                {
                    "imdb_id": row["tconst"],
                    "title": row["title"],
                    "year": row["year"],
                    "type": row["type"],
                    "rating": row["rating"],
                    "runtime_minutes": row["runtimeMinutes"],
                    "votes": row["votes"],
                    "genres": row["genres"],
                    "similarity": row["similarity_score"],
                    "hybrid_score": row["hybrid_score"],
                    "overview": row["overview"],
                    "poster_url": row["poster_url"],
                    "final_score": row["final_score"],
                    "genre_score": row["genre_score"],
                    "country_of_origin": row["country_of_origin"],
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
        return {"recommendations": recommendations, "prompt_title": prompt_title}
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []


def create_interface(engine):
    async def predict_wrapper(message):
        return await asyncio.to_thread(get_recommendations_api(message, engine))

    iface = gr.Interface(
        fn=predict_wrapper,
        inputs=gr.Textbox(lines=1, placeholder="Type your query..."),
        outputs=gr.JSON(label="Recommendations"),
        title="Recommendation API",
        api_name="predict",
    )
    return iface
