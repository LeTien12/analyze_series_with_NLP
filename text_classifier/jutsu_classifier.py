import torch
import gc
import pandas as pd
import huggingface_hub
from transformers import (AutoTokenizer , AutoModelForSequenceClassification , 
                          DataCollatorWithPadding , TrainingArguments , pipeline)
from .cleaner import Cleaner
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from datasets import Dataset
from .training_utils import get_class_weights , computer_metrics
from .custom_trainer import CustomTrainer


class JutsuClassifier():
    def __init__(self , model_path,
                 data_path,
                 text_column_name = 'text',
                 label_column_name = 'jutsu',
                 model_name = 'distilbert/distilbert-base-uncased',
                 test_size = 0.2,
                 num_labels = 3,
                 huggingface_token = None):
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.huggingface_token = huggingface_token
        
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            
        self.tokenizer = self.load_tokenizer()
        
        if not huggingface_hub.repo_exists(self.model_path):
            if self.data_path is None:
                raise ValueError("Data path is required to train the model, since the model path does not exist in huggungface hub")
            
            train_data , test_data = self.load_data()
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()
            
            
            all_data = pd.concat([train_data_df , test_data_df]).reset_index(drop = True)
            class_weights = get_class_weights(all_data)
            
            self.train_model(train_data , test_data , class_weights)
        
        self.model = self.load_model(self.model_path)
        
    def load_model(self , model_path):
        model = pipeline('text-classification' , model = model_path , return_all_scores = True)
        return model
        
    def train_model(self , train_data , test_data , class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                   num_labels = self.num_labels,
                                                                   id2label = self.labels_dict)
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir = self.model_path,
            learning_rate= 2e-4,
            per_device_train_batch_size= 8 ,
            per_device_eval_batch_size= 8,
            num_train_epochs=5,
            weight_decay= 0.01,
            eval_strategy='epoch',
            logging_strategy= 'epoch',
            push_to_hub= True,
            hub_token = self.huggingface_token
        )
        trainer = CustomTrainer(
            model= model,
            args= training_args,
            data_collator=data_collator,
            train_dataset= train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            compute_metrics= computer_metrics
        )
        
        trainer.set_divice(self.device)
        trainer.set_class_weight(class_weights)
        
        trainer.train()
        
        del trainer , model
        gc.collect()
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        
        
    def simplify_jutsu(self , jutsu):
        if 'Taijutsu' in jutsu:
            return "Taijutsu"
        if 'Genjutsu' in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
            
    def load_data(self):
        df = pd.read_json(self.data_path , lines= True)
        df['jutsu_type_simlified'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + '. ' + df['jutsu_description']
        df['jutsu'] = df['jutsu_type_simlified']
        df = df[['text' , 'jutsu']]
        df = df.dropna()
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)
        
        le = LabelEncoder()
        df['labels'] = le.fit_transform(df[self.label_column_name])
        self.labels_dict = {index : label_name for index , label_name in enumerate(le.__dict__['classes_'].tolist())}
        
        X_train , X_test  = train_test_split(df[['text_cleaned' , 'labels']],
                                                       test_size = self.test_size, random_state = 2024,
                                                       stratify = df.labels)
        
        # Conver Pandas to a hugging face dataset

        train_dataset = Dataset.from_pandas(X_train)
        test_dataset = Dataset.from_pandas(X_test)

        tokenizer_train = train_dataset.map(lambda examples: self.preprocess_function(examples) ,
                                            batched=True)

        tokenizer_test = test_dataset.map(lambda examples: self.preprocess_function(examples) ,
                                            batched=True)

        
        return tokenizer_train , tokenizer_test 
        
    def preprocess_function(self, examples):
        return self.tokenizer(examples['text_cleaned'] , truncation= True)

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        return tokenizer
       
    def postprocess(self , model_output):
        output = []
        for pred in model_output:
            labels = max(pred , key = lambda x: x['score'])['label']
            output.append(labels)
        return output
    
    def classify_jutsu(self,text):
        
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        return predictions
        