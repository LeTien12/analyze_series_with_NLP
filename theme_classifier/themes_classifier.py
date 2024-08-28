﻿import os
import torch
import sys
import numpy as np
import pandas as pd
import nltk
from transformers import pipeline
from nltk import sent_tokenize
from pathlib import Path
folder_path = Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path , '/..'))
from utils.load_data import load_file_subtitles

nltk.download('punkt')
class ThemeClassifier():
    def __init__(self , theme_list):
        self.model_name = 'facebook/bart-large-mnli'
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_mode(self.device)
        
    def load_mode(self , device):
        theme_classifier = pipeline("zero-shot-classification",
                      model= self.model_name,
                      device=device)
        return theme_classifier
        
    def get_themes_inference(self , script):
        script_sentences = sent_tokenize(script)
        
        #Batch Sentence
        sentence_batch_size = 20
        script_batches = []
        for index in range(0,len(script_sentences) , sentence_batch_size):
            sent = ' '.join(script_sentences[index:index+sentence_batch_size])
            script_batches.append(sent)
            
        #Run model
        theme_output = self.theme_classifier(
                        script_batches,
                        self.theme_list,
                        multi_label = True
                    )
        
        #Wrangle Out
        themes = {}
        for output in theme_output:
            for label , score in zip(output['labels'] , output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)
            
    
        themes = {k : np.mean(np.array(v))  for k , v in themes.items()}
        
        return themes
    
    def get_themes(self , dataset_path , save_path = None):
        
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            return df
        
        df = load_file_subtitles(dataset_path)
        
        output_themes = df['script'].apply(self.get_themes_inference)
        
        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df
        
        if save_path is not None:
            df.to_csv(save_path , index = False)
        return df