import gradio as gr

def create_interface(engine):
    def get_recommendations_text(query):
        try:
            result = engine.get_recommendations(query)
            if isinstance(result, tuple) and len(result) >= 1:
                return result[0]
            else:
                return str(result)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_thumbnails_html(query):
        try:
            result = engine.get_recommendations(query)
            if isinstance(result, tuple) and len(result) >= 1:
                search_results = engine.get_recommendations(query)
                

                thumbnails_html = []
                thumbnails_html.append("""
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; padding: 20px; max-height: 600px; overflow-y: auto;">
                """)

                thumbnails_html.append("""
                <div style="grid-column: 1 / -1; text-align: center; padding: 20px; color: #666;">
                    Thumbnails will appear here when poster URLs are available
                </div>
                """)
                
                thumbnails_html.append("</div>")
                return "".join(thumbnails_html)
                
        except Exception as e:
            return f"<div style='color: red; padding: 20px;'>❌ Error: {str(e)}</div>"

    def get_thumbnails_from_results(query):
        """Get thumbnails from search results"""
        try:
            formatted_results, df_results = engine.get_recommendations(query)
        
            html_parts = []
            html_parts.append("""
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; padding: 20px; max-height: 600px; overflow-y: auto; background: #f8f9fa; border-radius: 8px;">
            """)
            
            for i in range(10): 
                html_parts.append(f"""
                <div style="position: relative; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s; cursor: pointer;" 
                     onmouseover="this.style.transform='scale(1.05)'" 
                     onmouseout="this.style.transform='scale(1)'">
                    <div style="width: 100%; height: 200px; background: #ddd; display: flex; align-items: center; justify-content: center; color: #666; font-size: 12px;">
                        Poster {i+1}
                    </div>
                    <div style="position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(transparent, rgba(0,0,0,0.7)); color: white; padding: 8px; font-size: 12px; text-align: center;">
                        Movie Title {i+1}
                    </div>
                </div>
                """)
            
            html_parts.append("</div>")
            return "".join(html_parts)
            
        except Exception as e:
            return f"<div style='color: red; padding: 20px;'>❌ Error: {str(e)}</div>"
        
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="TV-Series and Movie Recommend",
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
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

            with gr.Column(scale=1):
                results_text = gr.Textbox(
                    label="Detailed Results",
                    lines=20,
                    max_lines=25,
                    show_copy_button=True,
                    interactive=False,
                )
                
            with gr.Column(scale=1):
                thumbnails_display = gr.HTML(
                    label="Movie Posters",
                    value="<div style='text-align: center; padding: 40px; color: #666;'>Movie thumbnails will appear here</div>"
                )

        search_btn.click(
            fn=get_recommendations_text,
            inputs=[query_input],
            outputs=[results_text],
        )
        
        search_btn.click(
            fn=get_thumbnails_from_results,
            inputs=[query_input],
            outputs=[thumbnails_display],
        )

    return demo