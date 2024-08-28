import os
import sys
import pathlib
import spacy
import pandas as pd
from ast import literal_eval
from nltk import sent_tokenize

folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path , '../'))
from utils.load_data import load_file_subtitles

class NamedEntityRecognize:
    def __init__(self) -> None:
        self.nlp_model = self.load_model()
        
    def load_model(self):
        nlp = spacy.load('en_core_web_trf')
        return nlp
    
    def get_ners_inference(self, script):
        script_sentences = sent_tokenize(script)
        
        ners_output = []
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == 'PERSON':
                    first_name = entity.text.split(" ")[0].strip()
                    ners.add(first_name)
            ners_output.append(ners)
            
        return ners_output
    
    def get_ners(self, dataset_path , save_path = None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x , str) else x)
            return df
        
        
        #load dataset
        df = load_file_subtitles(dataset_path)
        
        # Run inference
        df['ners'] = df['script'].apply(self.get_ners_inference)
        
        if save_path is not None:
            df.to_csv(save_path , index=False)
            
        return df
        
    
    
            