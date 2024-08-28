import os
import glob

import pandas as pd


def load_file_subtitles(path_data):
    subtitles_path = glob.glob(os.path.join(path_data , '*ass'))
    numbers= []
    scripts = []
    
    for path in subtitles_path:
        with open(path , 'r' ,encoding='utf-8') as file:
           
            lines = file.readlines()
            lines = lines[27:]
            lines = [','.join(line.split(',')[9:]) for line in lines]
        lines = [line.replace('\\N' , " ") for line in lines]
        script = ' '.join(lines)
        number = int(path.split('-')[-1].split('.')[0].strip())
        
        scripts.append(script)
        numbers.append(number)
    df = pd.DataFrame.from_dict({'number' : numbers , "script" : scripts})
    return df


        
