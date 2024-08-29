import re
import torch
import pandas as pd
import huggingface_hub
from datasets import Dataset
import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
import gc

from peft import LoraConfig , PeftModel
from trl import SFTConfig , SFTTrainer




class CharacterChatBot():
    def __init__(self , model_path, 
                 data_path = 'content/data/naruto.csv', 
                 huggingface_token = None):
        
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            
        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_mode(self.model_path)
        else:
            train_dataset = self.load_data()
            
            self.train(self.base_model_path , train_dataset)
            self.model = self.load_model(self.model_path)
            
    def chat(self , message , history):
        messages = []
        
        messages.append('''Your are Naruto from anime "Naruto". Your responses should reflect his personalist and speech patterns ''')
        
        for message_and_response in history:
            messages.append({'role' : 'user' , 'content' : message_and_response[0]})
            messages.append({'role' : 'assistant' , 'content' : message_and_response[1]})
        messages.append({"role":'user' , "content" : message})
        
        terimator = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output = self.model(
            messages,
            max_length = 256,
            terimator = terimator,
            do_sample = True,
            top_p = 0.9
        )
        
        output_messages = output[0]['generated_text'][-1]
        return output_messages
        
         
    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = 'nf4',
            bnb_4bit_compute_dtupe = torch.float16
        )
        
        pipeline = transformers.pipeline("text-generation",
                                         model = self.model_path,
                                         model_kwargs={
                                             "torch_dtype" : torch.float16,
                                             "quantization_config" : bnb_config,
                                         })
        return pipeline
           
    def train(self , base_model_name_or_path ,
              dataset,
              output_dir = "./results",
              per_device_train_batch_size = 1,
              gradient_accumulation_steps = 1,
              optim = 'paged_adamw_32bit',
              save_steps = 200,
              logging_steps = 10,
              learning_rate = 2e-4,
              max_grad_norm = 0.3,
              max_steps = 300,
              warmup_ratio = 0.3,
              lr_scheduler_type = 'constant'):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = 'nf4',
            bnb_4bit_compute_dtupe = torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,
                                                     quantization_config = bnb_config,
                                                     trust_remote_code = True)
        
        model.config.use_cache = False
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64
        
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout= lora_dropout,
            r = lora_r,
            bias= 'none',
            task_type = "CASUAL_LM"
        )
        training_arguments = SFTConfig(
        output_dir = output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate= learning_rate,
        fp16 = True,
        max_grad_norm= max_grad_norm,
        max_steps= max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to= 'none'
        )
        max_seq_len = 512
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            perf_config = peft_config,
            dataset_text_field='prompt',
            max_seq_length=max_seq_len,
            tokenizer=tokenizer,
            args=training_arguments
        )
        
        trainer.train()
        
        #Save model
        
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")
        
        del trainer , model
        gc.collect()
        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,
                                                          return_dict = True,
                                                          quantization_config = bnb_config,
                                                          torch_dtype = torch.float16,
                                                          device_map = self.device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        
        model = PeftModel.from_pretrained(base_model , 'final_ckpt')
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)
        
        del model , base_model
        gc.collect()
               
    def load_data(self):
        df = pd.read_csv(self.data_path)
        df.dropna(inplace=True)
        df["line"] = df['line'].apply(self.remove_paranthesis)
        df['number_of_words'] = df['line'].str.strip().str.split()
        df['number_of_words'] = df['number_of_words'].apply(lambda x : len(x))
        df['naruto_response_flag'] = 0
        df.loc[(df['name'] == 'Naruto') & (df['number_of_words'] > 5) , 'naruto_response_flag'] = 1
        indexes_to_take = list(df[(df['naruto_response_flag'] == 1) & (df.index > 0)].index)
        system_promt = '''Your are Naruto from anime "Naruto". 
        Your responses should reflect his personalist and speech patterns '''
        prompts = []

        for i in indexes_to_take:
            prompt = system_promt 
            prompt += df.iloc[i - 1]['line']
            prompt += '\n' 
            prompt += df.iloc[i]['line']
            prompts.append(prompt) 
        df = pd.DataFrame({'prompt' : prompts})
        dataset = Dataset.from_pandas(df)
        
        return dataset

        
        
        
    def remove_paranthesis(self , text):
        result = re.sub(r"\(.*?\)" , '' , text)
        return result
        
        
        