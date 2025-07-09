import gradio as gr
from models.recommendation_engine import RecommendationEngine
from components.imdb_poster import get_imdb_poster


def get_recommendations_api(message, engine):
    if not message:
        return []

    try:
        result = engine.get_recommendations(message)
        df = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if df is None or df.empty:
            return []

        recommendations = []
        for _, row in df.iterrows():
            poster_url = get_imdb_poster(row["ImdbId"])
            recommendations.append(
                {
                    "imdb_id": row["ImdbId"],
                    "title": row["Title"],
                    "year": row["Year"],
                    "type": row["Type"],
                    "rating": row["Rating"],
                    "votes": row["Votes"],
                    "genres": row["Genres"],
                    "similarity": row["Similarity"],
                    "hybrid_score": row["Hybrid Score"],
                    "overview": row["Overview"],
                    "poster_url": poster_url,
                }
            )
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
