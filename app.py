import gradio as gr
from functions import get_theme , get_character_network , classify_text
import promt



def main():
    text = promt.promt()
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
        
        # Text Classification with LLMs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output  = gr.Textbox(label= 'Text Classification Output')
                    with gr.Column():
                        text_classification_model = gr.Textbox(label='Model Path' , value= r'Tienle123/NLP')
                        text_classification_data_path = gr.Textbox(label="Data Path" , value=r'D:\hoc_lap_trinh\NLP\build_ai_NLP\dataset\jutsus.jsonl')
                        text_to_classify = gr.Textbox(label="Text Input" , value= text)
                        classify_text_button = gr.Button("Classify Text (Jutsu)")
                        classify_text_button.click(classify_text , inputs=[text_classification_model , text_classification_data_path, text_to_classify] , outputs=[text_classification_output])
    iface.launch(share=True)
      
    
if __name__ == '__main__':
    main()