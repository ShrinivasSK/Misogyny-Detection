# Core
import random

# Basics
import numpy as np
import pandas as pd
import torch

# Metrics
from sklearn.metrics import *

# Tokeniser
from transformers import BertTokenizer

# Utility
from tqdm import tqdm

# Dataloader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Scheduler
from transformers import get_linear_schedule_with_warmup

# Optimiser
from transformers import AdamW

# Model
from transformers import BertForSequenceClassification
import torch.nn as nn

class BERT:
    def __init__(self,args):
        # fix the random
        random.seed(args['seed_val'])
        np.random.seed(args['seed_val'])
        torch.manual_seed(args['seed_val'])
        torch.cuda.manual_seed_all(args['seed_val'])
                
        self.device = torch.device(args['device'])
        self.weights=args['weights']
        
        self.tokenizer = BertTokenizer.from_pretrained(args['bert_model'])
        
    ##-----------------------------------------------------------##
    ##----------------- Utility Functions -----------------------##
    ##-----------------------------------------------------------##
    def encode(self,data,max_len):
        
        input_ids = []
        attention_masks = []
        for sent in tqdm(data):
            encoded_dict = self.tokenizer.encode_plus(
                            sent,
                            add_special_tokens =True, # for [CLS] and [SEP]
                            max_length = max_len,
                            truncation = True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt', # return pytorch tensors
            )
            input_ids.append(encoded_dict['input_ids'])

            attention_masks.append(encoded_dict['attention_mask'])
        
        return [input_ids,attention_masks]
    
    ##-----------------------------------------------------------##
    ##------------------ Dataloader -----------------------------##
    ##-----------------------------------------------------------##
    def get_dataloader(self,samples, batch_size,is_train=False):
        inputs,masks,labels = samples

        # Convert the lists into tensors.
        inputs = torch.cat(inputs, dim=0)
        masks = torch.cat(masks, dim=0)
        labels = torch.tensor(labels)

        data = TensorDataset(inputs,masks,labels)

        if(is_train==False):
            sampler = SequentialSampler(data)
        else:
            sampler = RandomSampler(data)  

        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader
    
    ##-----------------------------------------------------------##
    ##----------------- Training Utilities ----------------------##
    ##-----------------------------------------------------------## 
    def get_optimiser(self,learning_rate,model):
        return AdamW(model.parameters(),
                  lr = learning_rate, 
                  eps = 1e-8
                )
    
    def get_scheduler(self,epochs,optimiser,train_dl):
        total_steps = len(train_dl) * epochs
        return get_linear_schedule_with_warmup(optimiser, 
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = total_steps)
    
    def evalMetric(self,y_true, y_pred,prefix):
        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro')
        f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_pred))
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_c = auc(fpr, tpr)
        recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred))
        precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred))
        return dict({prefix+"accuracy": accuracy, prefix+'mF1Score': mf1Score, 
                        prefix+'f1Score': f1Score, prefix+'precision': precisionScore, 
                        prefix+'recall': recallScore})
    
    ##-----------------------------------------------------------##
    ##---------------- Different Train Loops --------------------##
    ##-----------------------------------------------------------## 
    def evaluate(self,model,loader,which):
    
        model.eval() # put model in eval mode

        total_eval_loss = 0
        nb_eval_steps = 0

        y_pred = np.zeros(shape=(0),dtype='int')
        y_true = np.empty(shape=(0),dtype='int')

        for batch in loader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad(): # do not construct compute graph
                outputs = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

            loss = outputs[0]
            logits = outputs[1]

            total_eval_loss += loss.item()

            b_y_true = b_labels.cpu().data.squeeze().numpy()

            b_y_pred = torch.max(logits,1)[1]
            b_y_pred = b_y_pred.cpu().data.squeeze().numpy()

            y_pred = np.concatenate((y_pred,b_y_pred))
            y_true = np.concatenate((y_true,b_y_true))

        metrics = self.evalMetric(y_true,y_pred,which+"_")

        # Calculate the average loss over all of the batches.
        avg_loss = total_eval_loss / len(loader)

        metrics[which+'_avg_loss'] = avg_loss

        return metrics
    
    
    def run_train_loop(self,model,train_loader,optimiser,scheduler):
        
        total_loss = 0
        model.train() # put model in train mode

        y_pred = np.zeros(shape=(0),dtype='int')
        y_true = np.empty(shape=(0),dtype='int')

        for step, batch in tqdm(enumerate(train_loader)):

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            model.zero_grad()        

            outputs = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

            logits = outputs[1]

            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
                        self.weights,dtype=torch.float))
            
            loss = loss_fct(logits,b_labels)
            
            total_loss += loss.item()

            loss.backward()

            b_y_true = b_labels.cpu().data.squeeze().numpy()

            b_y_pred = torch.max(logits,1)[1]
            b_y_pred = b_y_pred.cpu().data.squeeze().numpy()

            y_pred = np.concatenate((y_pred,b_y_pred))
            y_true = np.concatenate((y_true,b_y_true))

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimiser.step()
            
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        train_metrics = self.evalMetric(y_true,y_pred,"Train_")

        print('avg_train_loss',avg_train_loss)
        print('train_f1Score',train_metrics['Train_f1Score'])
        print('train_accuracy',train_metrics['Train_accuracy'])

        train_metrics['Train_avg_loss'] = avg_train_loss

        return train_metrics
    
    
    ##------------------------------------------------------------##
    ##----------------- Main Train Loop --------------------------##
    ##------------------------------------------------------------##
    def train(self,model,data_loaders,optimiser,scheduler,epochs):
        train_stats = []
        train_loader,val_loader,test_loader = data_loaders
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            
            print("")
            print('Training...')
            train_metrics = self.run_train_loop(model,train_loader,
                                            optimiser,scheduler)

            print("")
            print("Running Validation...") 
            val_metrics = self.evaluate(model,val_loader,"Val")
            
            print("Validation Loss: ",val_metrics['Val_avg_loss'])
            print("Validation Accuracy: ",val_metrics['Val_accuracy'])
            
            stats = {}

            stats['epoch']=epoch_i+1

            stats.update(train_metrics)
            stats.update(val_metrics)

            train_stats.append(stats)

        return train_stats
    
    ##-----------------------------------------------------------##
    ##----------------------- Main Pipeline ---------------------##
    ##-----------------------------------------------------------##
    def run(self,args,df_train,df_val,df_test):
        X_train = df_train['Text'].values
        Y_train = df_train['Label'].values
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values
        X_val = df_val['Text'].values
        Y_val = df_val['Label'].values
        
        train_data = self.encode(X_train,args['max_len'])
        val_data = self.encode(X_val,args['max_len'])
        test_data = self.encode(X_test,args['max_len'])
        
        train_data.append(Y_train)
        val_data.append(Y_val)
        test_data.append(Y_test)
        
        train_dl =self.get_dataloader(train_data,args['batch_size'],True)
        val_dl =self.get_dataloader(val_data,args['batch_size'])                          
        test_dl =self.get_dataloader(test_data,args['batch_size'])
        
        model = BertForSequenceClassification.from_pretrained(
                args['bert_model'], 
                num_labels = 2, 
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
            )
        
        optimiser = self.get_optimiser(args['learning_rate'],model)
        
        scheduler = self.get_scheduler(args['epochs'],optimiser,train_dl)
        
        train_stats = self.train(model,[train_dl,val_dl,test_dl],
                                optimiser,scheduler,args['epochs'])
        
        return train_stats
        