# Basics
import torch
import numpy as np

# Model
import torch.nn as nn


class LSTM_Model(nn.Module):

    def __init__(self, weights,embs_np,dimension=128):
        super(LSTM_Model, self).__init__()
        
        self.weights = weights
        input_size = embs_np.shape[1]

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_np).float())
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 2)

    def forward(self, text,labels):
        text_len = 1024

        text_emb = self.embedding(text)

        #packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(text_emb)
        #output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        output = self.fc(text_fea)
        #text_fea = torch.squeeze(text_fea, 1)
        #text_out = torch.sigmoid(text_fea)
    
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights,dtype=torch.float))
        
        #oss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(output,labels)
        return loss,output


def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret

class CNN_GRU_Model(nn.Module):
    def __init__(self,args,vector):
        super(CNN_GRU_Model, self).__init__()
        self.embedsize = vector.shape[1]
        self.vocab_size = vector.shape[0]
        self.conv1 = nn.Conv1d(self.embedsize,100, 2)
        self.conv2 = nn.Conv1d(self.embedsize,100, 3,padding=1)
        self.conv3 = nn.Conv1d(self.embedsize,100, 4,padding=2)
        self.maxpool1D = nn.MaxPool1d(4, stride=4)
        self.seq_model = nn.GRU(100, 100, bidirectional=False, batch_first=True)
        self.embedding = nn.Embedding(self.vocab_size, self.embedsize)
        self.embedding.weight = nn.Parameter(torch.tensor(vector.astype(np.float32), dtype=torch.float32))
        self.embedding.weight.requires_grad = args["train_embed"]
        self.num_labels=2
        self.weights=args['weights']
        self.out = nn.Linear(100, self.num_labels)

        
    def forward(self,x,labels=None):
        batch_size=x.size(0)
        h_embedding = self.embedding(x)
        new_conv1=self.maxpool1D(self.conv1(h_embedding.permute(0,2,1)))
        new_conv2=self.maxpool1D(self.conv2(h_embedding.permute(0,2,1)))
        new_conv3=self.maxpool1D(self.conv3(h_embedding.permute(0,2,1)))
        concat=self.maxpool1D(torch.cat([new_conv1, new_conv2,new_conv3], dim=2))
        h_seq, _ = self.seq_model(concat.permute(0,2,1))
        global_h_seq=torch.squeeze(global_max_pooling(h_seq, 1, 1)) 
        output=self.out(global_h_seq)
        
        if labels is not None:
        	loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights,dtype=torch.float))
        	loss = loss_fct(output.view(-1, self.num_labels), labels.view(-1))
        	return loss,output
        return output