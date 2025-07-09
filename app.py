import gradio as gr
from models.recommendation_engine import RecommendationEngine
from components.gradio_ui import create_interface
from config import Config


def main():
    engine = RecommendationEngine()

    interface = create_interface(engine)

    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
    )


if __name__ == "__main__":
    main()
