import gradio as gr
import os

from dotenv import load_dotenv


load_dotenv('../.env')

from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognize , CharaterNetworkGennerator
from text_classifier import JutsuClassifier


def get_theme(theme_list , subtitles_path , save_path):
    theme_list = theme_list.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_theme = theme_classifier.get_themes(subtitles_path , save_path)
    
    #Remove dialogue from the theme list
    
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_theme[theme_list]
    output_df = output_df.sum().reset_index()
    output_df.columns = ['Theme' , 'Score']
    
    output_chart = gr.BarPlot(
        output_df,
        x = 'Theme',
        y = 'Score',
        title= 'Series Themes',
        tooltip=['Theme' , 'Score'],
        vertical= False,
        width= 500,
        height= 250
    )
    return output_chart

def get_character_network(subtitles_path , ner_path):
    ner = NamedEntityRecognize()
    ner_df = ner.get_ners(subtitles_path , ner_path)
    
    character_network_generator = CharaterNetworkGennerator()
    relationship_df = character_network_generator.generate_chasracter_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    return html

def classify_text(text_classification_model , text_classification_data_path, text_to_classify):
    jutsu_classifier = JutsuClassifier(model_path= text_classification_model , 
                                       data_path=text_classification_data_path,
                                       huggingface_token= os.getenv('HUGGINGFACE_TOKEN'))
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    
    return output


    
    
    