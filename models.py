# Basics
import torch
import numpy as np

# Model
import torch.nn as nn

# from transformers.models.modeling_roberta import *
from transformers import RobertaPreTrainedModel,RobertaModel

##----------------------------------------------------------------##
##------------------------ Bi-LSTM Model -------------------------##
##----------------------------------------------------------------##
class LSTM_Model(nn.Module):

    def __init__(self, weights,embs_np,device,dimension=128):
        super(LSTM_Model, self).__init__()
        
        self.weights = weights
        input_size = embs_np.shape[1]
        
        self.device=torch.device(device)

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_np).float()).to(self.device)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 2)

    def forward(self, text,labels):
        text_len = 128

        text_emb = self.embedding(text)

        output, _ = self.lstm(text_emb)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        output = self.fc(text_fea)
    
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights,dtype=torch.float).to(self.device))

        loss = loss_fct(output,labels)
        return loss,output

##----------------------------------------------------------------##
##----------------------- CNN-GRU Model --------------------------##
##----------------------------------------------------------------##
def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret

class CNN_GRU_Model(nn.Module):
    def __init__(self,args,vector,device):
        super(CNN_GRU_Model, self).__init__()
        self.device=torch.device(device)
        
        self.embedsize = vector.shape[1]
        self.vocab_size = vector.shape[0]
        self.conv1 = nn.Conv1d(self.embedsize,100, 2)
        self.conv2 = nn.Conv1d(self.embedsize,100, 3,padding=1)
        self.conv3 = nn.Conv1d(self.embedsize,100, 4,padding=2)
        self.maxpool1D = nn.MaxPool1d(4, stride=4)
        self.seq_model = nn.GRU(100, 100, bidirectional=False, batch_first=True)
        self.embedding = nn.Embedding(self.vocab_size, self.embedsize)
        self.embedding.weight = nn.Parameter(torch.tensor(vector.astype(np.float32), dtype=torch.float32).to(self.device))
        self.embedding.weight.requires_grad = args["train_embed"]
        self.num_labels=2
        self.weights=args['weights']
        self.out = nn.Linear(100, self.num_labels)

        
    def forward(self,x,labels=None):
        h_embedding = self.embedding(x)
        new_conv1=self.maxpool1D(self.conv1(h_embedding.permute(0,2,1)))
        new_conv2=self.maxpool1D(self.conv2(h_embedding.permute(0,2,1)))
        new_conv3=self.maxpool1D(self.conv3(h_embedding.permute(0,2,1)))
        concat=self.maxpool1D(torch.cat([new_conv1, new_conv2,new_conv3], dim=2))
        h_seq, _ = self.seq_model(concat.permute(0,2,1))
        global_h_seq=torch.squeeze(global_max_pooling(h_seq, 1, 1)) 
        output=self.out(global_h_seq)
        
        if labels is not None:
        	loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights,dtype=torch.float).to(self.device))
        	loss = loss_fct(output.view(-1, self.num_labels), labels.view(-1))
        	return loss,output
        return output
    
##----------------------------------------------------------------##
##--------------------- XLM Roberta Model ------------------------##
##----------------------------------------------------------------##
class weighted_Roberta(RobertaPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights=params['weights']
        self.roberta= RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.softmax=nn.Softmax(config.num_labels)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        attention_vals=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
            
        return (logits,)+outputs[2:]  # (loss), logits, (hidden_states), (attentions)
    
    
    
    def freeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.roberta.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.roberta.named_parameters():
            if "pooler" in name:
                param.requires_grad = True