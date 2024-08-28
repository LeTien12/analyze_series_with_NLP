import gradio as gr
from theme_classifier.themes_classifier import ThemeClassifier


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
    
    