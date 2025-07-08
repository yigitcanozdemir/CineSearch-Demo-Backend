import gradio as gr
from components.imdb_poster import get_imdb_poster


def get_multiple_imdb_posters(imdb_ids):
    posters = []
    for imdb_id in imdb_ids:
        poster_url = get_imdb_poster(imdb_id)
        print(f"IMDB ID: {imdb_id}, Poster URL: {poster_url}")
        posters.append({"tconst": imdb_id, "poster_url": poster_url})
    return posters


def create_interface(engine):
    def chat_function(message, history):
        if not message:
            return history, "", []

        try:
            result = engine.get_recommendations(message)
            imdb_ids = []
            df = result[1]
            if isinstance(result, tuple) and len(result) > 1:
                if hasattr(result[1], "columns") and "ImdbId" in result[1].columns:
                    imdb_ids = result[1]["ImdbId"].tolist()

            posters = get_multiple_imdb_posters(imdb_ids)

            thumbnails = [p["poster_url"] for p in posters if p["poster_url"]]

            response_text = result[0] if isinstance(result, tuple) else str(result)
            history.append([message, response_text])

            return history, "", thumbnails

        except Exception as e:
            history.append([message, f"❌ Error: {str(e)}"])
            return history, "", []

    with gr.Blocks() as demo:
        with gr.Column():
            chatbot = gr.Chatbot(height=600)
            gallery = gr.Gallery(
                label="Posters", show_label=False, columns=5, height=400
            )
            msg = gr.Textbox(placeholder="Enter your query", scale=1)

        msg.submit(chat_function, [msg, chatbot], [chatbot, msg, gallery])

    return demo
