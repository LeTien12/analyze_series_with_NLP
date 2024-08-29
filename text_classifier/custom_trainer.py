import torch
import torch.nn as nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self , model ,inputs, return_outputs = False):
        labels = inputs.get('labels')
        
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits.float()
        
        loss_fct = nn.CrossEntropyLoss( weight= torch.tensor(self.class_weights , dtype=torch.float).to( device=self.device))
        loss = loss_fct(logits.view(-1 , self.model.config.num_labels) , labels.view(-1))
        
        return (loss , outputs) if return_outputs else loss
    
    def set_class_weight(self , class_weight):
        self.class_weights = class_weight
    def set_divice(self , device):
        self.device = device
        
        
        


