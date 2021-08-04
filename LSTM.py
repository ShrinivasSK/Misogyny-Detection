# Core
import random

# Basics
import numpy as np
import pandas as pd
import torch

# Utility
from tqdm import tqdm

# Dataloader
from torch.utils.data import TensorDataset, DataLoader,RandomSampler, SequentialSampler

# Optimiser
from transformers import AdamW

# Metrics
from sklearn.metrics import *

# Model
from models import LSTM_Model

class LSTM:
    def __init__(self,args):
        # fix the random
        random.seed(args['seed_val'])
        np.random.seed(args['seed_val'])
        torch.manual_seed(args['seed_val'])
        torch.cuda.manual_seed_all(args['seed_val'])
                
        self.device = torch.device(args['device'])
        
        self.vocab2idx,self.embeddings = self.load_embeddings(args['embedding_path'])

        self.model_save_path = args['model_save_path']
        self.name = args['name']
        
    ##----------------------------------------------------------##
    ##------------------- Utility Functions --------------------##
    ##----------------------------------------------------------##
    def load_embeddings(self,path):
        vocab2idx,embeddings = {},[]
        with open(path,'rt') as f:
            full_content = f.read().strip().split('\n')
        
        for i,line in tqdm(enumerate(full_content)):
            word = line.split(' ')[0]
            embedding = [float(val) for val in line.split(' ')[1:]]
            vocab2idx[word]=i
            embeddings.append(embedding)
            
        embs_np = np.array(embeddings)
        
        #embedding for '<pad>' token: 0s
        pad_emb_np = np.zeros((1,embs_np.shape[1]))   
        #embedding for '<unk>' token: mean
        unk_emb_np = np.mean(embs_np,axis=0,keepdims=True) 
        
        embs_np = np.vstack((embs_np,pad_emb_np,unk_emb_np))
        
        vocab2idx['<unk>']=400001
        vocab2idx['<pad>']=400000
        
        return vocab2idx,embs_np
    
    def encode(self,text,max_len):
        encoded=[]
        for word in text.split(' '):
            word = word.lower()
            try:
                idx = self.vocab2idx[word]
            except:
                idx = self.vocab2idx['<unk>']
            encoded.append(idx)
        if(len(encoded)<max_len):
            padding = [self.vocab2idx['<pad>']]*(max_len-len(encoded))
            encoded.extend(padding)
        else:
            encoded=encoded[:max_len]
        return encoded
    
    ##----------------------------------------------------------##
    ##---------------------- Data Loader -----------------------##
    ##----------------------------------------------------------## 
    def get_dataloader(self,X,Y,batch_size,max_len,is_train=False):
        inputs = [self.encode(text,max_len) for text in X]
        labels = Y

        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels,dtype=torch.long)

        data = TensorDataset(inputs,labels)

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
            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            with torch.no_grad(): # do not construct compute graph
                outputs = model(b_inputs,b_labels)

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
    
    
    def run_train_loop(self,model,train_loader,optimiser):
        
        total_loss = 0
        model.train() # put model in train mode

        y_pred = np.zeros(shape=(0),dtype='int')
        y_true = np.empty(shape=(0),dtype='int')

        for step, batch in tqdm(enumerate(train_loader)):

            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            model.zero_grad()        

            outputs = model(b_inputs,b_labels)

            loss = outputs[0]
            logits = outputs[1]

            total_loss += loss.item()

            loss.backward()

            b_y_true = b_labels.cpu().data.squeeze().numpy()

            b_y_pred = torch.max(logits,1)[1]
            b_y_pred = b_y_pred.cpu().data.squeeze().numpy()

            y_pred = np.concatenate((y_pred,b_y_pred))
            y_true = np.concatenate((y_true,b_y_true))

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimiser.step()

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
    def train(self,model,data_loaders,optimiser,epochs):
        train_stats = []
        train_loader,val_loader,test_loader = data_loaders
        best_mf1Score=-1.0
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            
            print("")
            print('Training...')
            train_metrics = self.run_train_loop(model,train_loader,optimiser)

            print("")
            print("Running Validation...") 
            val_metrics = self.evaluate(model,val_loader,"Val")
            
            print("Validation Loss: ",val_metrics['Val_avg_loss'])
            print("Validation Accuracy: ",val_metrics['Val_accuracy'])
            
            if(val_metrics['Val_mF1Score']>best_mf1Score):
                best_mf1Score=val_metrics['Val_mF1Score']
                torch.save(model.state_dict(), self.model_save_path+
                        '/best-cnn_gru_'+self.name+'.pt')
                test_metrics = self.evaluate(model,test_loader,"Test")

            stats = {}

            stats['epoch']=epoch_i+1

            stats.update(train_metrics)
            stats.update(val_metrics)

            train_stats.append(stats)

        return train_stats,test_metrics
    
    ##-----------------------------------------------------------##
    ##--------------------- The Pipeline ------------------------##
    ##-----------------------------------------------------------##
    def run(self,args,df_train,df_val,df_test):
        X_train = df_train['Text'].values
        Y_train = df_train['Label'].values
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values
        X_val = df_val['Text'].values
        Y_val = df_val['Label'].values
        
        train_dl = self.get_dataloader(X_train,Y_train,
                        args['batch_size'],args['max_len'],True)
        val_dl = self.get_dataloader(X_val,Y_val,args['batch_size'],
                               args['max_len'])
        test_dl = self.get_dataloader(X_test,Y_test,args['batch_size'],
                                args['max_len'])
        
        model = LSTM_Model(args['weights'],self.embeddings)
        
        optimiser =self.get_optimiser(args['learning_rate'],model)
        
        train_stats,test_metrics = self.train(model,[train_dl,val_dl,test_dl],
                            optimiser,args['epochs'])
                            
        return train_stats,test_metrics