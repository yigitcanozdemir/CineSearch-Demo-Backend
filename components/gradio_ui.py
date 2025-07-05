import gradio as gr


def create_interface(engine):
    def get_recommendations_text(query):
        """Wrapper function to safely get only the text result"""
        try:
            result = engine.get_recommendations(query)
            if isinstance(result, tuple) and len(result) >= 1:
                return result[0]
            else:
                return str(result)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    with gr.Blocks(
        theme=gr.themes.Soft(), title="TV-Series and Movie Recommend"
    ) as demo:
        gr.Markdown("# 🎬 TV-Series and Movie Recommend")

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="What you want to watch?",
                    placeholder="Define your preferences as detailed as possible.",
                    lines=3,
                )

                search_btn = gr.Button("🔍 Search", variant="primary")

            with gr.Column(scale=2):
                results_text = gr.Textbox(
                    label="Recommended Movies and TV-Series",
                    lines=20,
                    max_lines=25,
                    show_copy_button=True,
                    interactive=False,
                )

        search_btn.click(
            fn=get_recommendations_text,
            inputs=[query_input],
            outputs=[results_text],
        )

    return demo
