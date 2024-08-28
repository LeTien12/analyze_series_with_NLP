﻿import gradio as gr
from functions import get_theme , get_character_network



def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme classification (Zero Shot Clasifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label='Themes' , value = 'friendship,hope,sacrifice,battle,self development,betrayal,love,dialogue')
                        subtitles_path = gr.Textbox(label= 'Subtitles or script Path')
                        save_path = gr.Textbox(label= 'Save Path' , value= r'D:\hoc_lap_trinh\NLP\build_ai_NLP\dataset\theme_classifier.csv')
                        get_themes_button = gr.Button("Get Themes") 
                        get_themes_button.click(get_theme , inputs=[theme_list , subtitles_path , save_path] , outputs=[plot])
        
        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Charater Network</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label='Subtutles or Script Path' , value= r'D:\hoc_lap_trinh\NLP\build_ai_NLP\dataset\Subtitles')
                        ner_path = gr.Textbox(label= 'NERs save path' ,value=r'D:\hoc_lap_trinh\NLP\build_ai_NLP\output\ners_output.csv')
                        get_themes_button = gr.Button("Get Charater Network") 
                        get_themes_button.click(get_character_network , inputs=[subtitles_path , ner_path] , outputs=[network_html])
    iface.launch(share=True)
      
    
if __name__ == '__main__':
    main()